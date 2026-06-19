"""Behavioral tests for the Runtime / ReAct loop (nanoma/runtime.py).

Every test drives the loop with a deterministic mock LLM and asserts a concrete
outcome (status, result string, history shape, emitted events, threshold alerts).
"""

import asyncio
import time
import pytest

from nanoma.core import Runtime, RuntimeConfig, Envelope
from nanoma.llm import LLMResponse, ToolCall
from nanoma.cost import UsageRecord
from nanoma.models import load_models
from nanoma.tools import build_all_tools
from nanoma.runtime import _peak_concurrency


def ev(agent, event, t):
    return {"agent": agent, "event": event, "t": t, "data": {}}


class TestPeakConcurrency:
    def test_sequential_is_one(self):
        # a born→dies, then b born→dies: never overlap.
        events = [ev("a", "agent_new", 0), ev("a", "done", 1),
                  ev("b", "agent_new", 1), ev("b", "done", 2)]
        assert _peak_concurrency(events) == 1

    def test_two_overlap_is_two(self):
        events = [ev("a", "agent_new", 0), ev("b", "agent_new", 1),
                  ev("a", "done", 2), ev("b", "done", 3)]
        assert _peak_concurrency(events) == 2

    def test_parent_plus_two_children_is_three(self):
        # The real bug: root alive while two children run in parallel → 3.
        events = [ev("root", "agent_new", 0),
                  ev("c1", "agent_new", 1), ev("c2", "agent_new", 1),
                  ev("c1", "done", 2), ev("c2", "done", 2),
                  ev("root", "done", 3)]
        assert _peak_concurrency(events) == 3

    def test_unfinished_agent_counts_to_end(self):
        events = [ev("a", "agent_new", 0), ev("b", "agent_new", 1)]  # neither ends
        assert _peak_concurrency(events) == 2

    def test_empty(self):
        assert _peak_concurrency([]) == 0


def usage(model="m", inp=10, out=5):
    return UsageRecord(input_tokens=inp, output_tokens=out, model=model)


def done_llm(result="ok"):
    async def f(messages, model, tools=None, **kw):
        return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": result})], usage=usage(model))
    return f


def mk(tmp_path, llm, **cfg):
    cfg.setdefault("budget", 10.0)
    cfg.setdefault("log_dir", None)
    cfg.setdefault("workspace_root", tmp_path / "ws")
    return Runtime(config=RuntimeConfig(**cfg), llm_call=llm)


# ─── create_agent ────────────────────────────────────────────────────────────

class TestCreateAgent:
    def test_basic_fields_and_workspace(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        a = rt.create_agent("do thing")
        assert a.id == "alpha" and a.task == "do thing" and a.status == "running"
        assert a.workspace.exists() and a.workspace.name == "alpha"
        assert a.history[0]["role"] == "system" and "do thing" in a.history[0]["content"]
        assert a.history[1] == {"role": "user", "content": "Begin working on your task now."}

    def test_parent_child_and_context_section(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        p = rt.create_agent("parent task")
        c = rt.create_agent("child task", parent=p.id, depth=1)
        assert c.id in p.children and c.parent == p.id and c.depth == 1
        assert "## Your Context" in c.history[0]["content"]
        assert p.id in c.history[0]["content"]

    def test_explicit_workspace(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        custom = tmp_path / "app"
        a = rt.create_agent("t", workspace=custom)
        assert a.workspace == custom and custom.exists()
        # prompt, shell cwd, and file tools all agree on this dir
        assert str(custom) in a.history[0]["content"]

    def test_context_limit_from_registry(self, tmp_path):
        cfg = tmp_path / "m.yaml"
        cfg.write_text("models:\n  tiny:\n    context_limit: 4096\n")
        load_models(cfg)
        rt = mk(tmp_path, done_llm(), default_model="tiny")
        a = rt.create_agent("t")
        assert a.context_limit == 4096
        load_models(__import__("pathlib").Path(__file__).parent.parent / "models.yaml")

    def test_fusion_expansion(self, tmp_path):
        cfg = tmp_path / "m.yaml"
        cfg.write_text(
            "models:\n  judge-m:\n    context_limit: 8000\n  panel-a: {}\n  panel-b: {}\n"
            "fusion:\n  fuse:\n    panel: [panel-a, panel-b]\n    judge: judge-m\n"
        )
        load_models(cfg)
        rt = mk(tmp_path, done_llm(), default_model="fuse")
        a = rt.create_agent("hard task")
        # Agent runs on the concrete judge model, not the fusion name
        assert a.model == "judge-m"
        sysmsg = a.history[0]["content"]
        assert "Model Fusion — REQUIRED" in sysmsg
        assert 'model="panel-a"' in sysmsg and 'model="panel-b"' in sysmsg
        load_models(__import__("pathlib").Path(__file__).parent.parent / "models.yaml")


# ─── run() / agent loop ──────────────────────────────────────────────────────

class TestAgentLoop:
    @pytest.mark.asyncio
    async def test_simple_done(self, tmp_path):
        rt = mk(tmp_path, done_llm("finished"))
        assert await rt.run("t") == "finished"

    @pytest.mark.asyncio
    async def test_max_turns(self, tmp_path):
        async def never_done(messages, model, tools=None, **kw):
            return LLMResponse(tool_calls=[ToolCall("g", "get_cost", {})], usage=usage(model))
        rt = mk(tmp_path, never_done, max_turns=3)
        result = await rt.run("t")
        a = rt.agents["alpha"]
        assert a.status == "done" and result == "[Max turns reached]" and a._turns == 4

    @pytest.mark.asyncio
    async def test_budget_exhausted(self, tmp_path):
        async def costly(messages, model, tools=None, **kw):
            # huge output → fallback pricing ($3/M out) blows the tiny budget
            return LLMResponse(tool_calls=[ToolCall("g", "get_cost", {})], usage=usage(model, out=1_000_000))
        rt = mk(tmp_path, costly, budget=0.01, max_turns=50)
        result = await rt.run("t")
        a = rt.agents["alpha"]
        assert a.status == "failed" and "BUDGET" in result
        assert rt.ledger.remaining() == 0

    @pytest.mark.asyncio
    async def test_content_then_done(self, tmp_path):
        calls = {"n": 0}
        async def f(messages, model, tools=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return LLMResponse(content="thinking out loud", usage=usage(model))
            if calls["n"] == 2:
                return LLMResponse(content=None, usage=usage(model))  # empty branch
            return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": "z"})], usage=usage(model))
        rt = mk(tmp_path, f, max_turns=10)
        assert await rt.run("t") == "z"
        hist = rt.agents["alpha"].history
        assert any(m.get("content") == "thinking out loud" for m in hist)
        assert any(m.get("content") == "(empty)" for m in hist)

    @pytest.mark.asyncio
    async def test_tool_execution_and_event(self, tmp_path):
        calls = {"n": 0}
        async def f(messages, model, tools=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return LLMResponse(tool_calls=[ToolCall("c", "create_file", {"path": "x.txt", "content": "hi"})], usage=usage(model))
            return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": "ok"})], usage=usage(model))
        rt = mk(tmp_path, f, max_turns=10)
        await rt.run("t")
        a = rt.agents["alpha"]
        assert (a.workspace / "x.txt").read_text() == "hi"
        tool_events = [e for e in rt._events if e["event"] == "tool_call"]
        assert any(e["data"]["tool"] == "create_file" for e in tool_events)
        assert any(m["role"] == "tool" for m in a.history)

    @pytest.mark.asyncio
    async def test_crash_is_caught(self, tmp_path):
        async def boom(messages, model, tools=None, **kw):
            raise ValueError("kaboom")
        rt = mk(tmp_path, boom)
        result = await rt.run("t")
        a = rt.agents["alpha"]
        assert a.status == "failed" and result.startswith("Crash:") and "kaboom" in result

    @pytest.mark.asyncio
    async def test_no_tool_call_injects_nudge(self, tmp_path):
        # A content-only turn (no tool call) should trigger a user nudge so a
        # verbose model is pushed to act/finish instead of stalling.
        calls = {"n": 0}
        async def f(messages, model, tools=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return LLMResponse(content="Let me think about this...", usage=usage(model))
            return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": "z"})], usage=usage(model))
        rt = mk(tmp_path, f, max_turns=10)
        assert await rt.run("t") == "z"
        hist = rt.agents["alpha"].history
        assert any(m["role"] == "user" and "No tool call" in (m.get("content") or "") for m in hist)

    @pytest.mark.asyncio
    async def test_parent_notified_on_child_done(self, tmp_path):
        async def f(messages, model, tools=None, **kw):
            sysmsg = messages[0]["content"]
            n = sum(1 for m in messages if m.get("role") == "assistant")
            if "child task" in sysmsg:
                return LLMResponse(tool_calls=[ToolCall("c", "set_status", {"status": "done", "result": "child-result"})], usage=usage(model))
            if n == 0:
                return LLMResponse(tool_calls=[ToolCall("s", "spawn", {"task": "child task"})], usage=usage(model))
            if n == 1:
                return LLMResponse(tool_calls=[ToolCall("w", "wait", {"timeout": 5})], usage=usage(model))
            return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": "parent-done"})], usage=usage(model))
        rt = mk(tmp_path, f, max_turns=10, max_agents=5)
        assert await rt.run("parent task") == "parent-done"
        # the system→parent completion notice was recorded
        assert any(frm == "system" and "alpha" == to for frm, to, _ in rt._messages_sent)

    @pytest.mark.asyncio
    async def test_immediate_interrupt_skips_remaining_tool_calls(self, tmp_path):
        calls = {"n": 0}
        async def f(messages, model, tools=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return LLMResponse(tool_calls=[
                    ToolCall("g1", "get_cost", {}),
                    ToolCall("g2", "get_cost", {}),
                ], usage=usage(model))
            return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": "ok"})], usage=usage(model))
        rt = mk(tmp_path, f, max_turns=10)
        a = rt.create_agent("t")
        a._immediate_inbox.put_nowait(Envelope("system", a.id, "STOP NOW", 3, time.time(), mode="immediate"))
        rt.start_agent(a)
        await a._task
        assert any('"interrupted": true' in (m.get("content") or "") for m in a.history)


# ─── deliver / messaging ─────────────────────────────────────────────────────

class TestDeliver:
    @pytest.mark.asyncio
    async def test_modes_route_to_correct_inbox(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        a = rt.create_agent("t")
        a.status = "idle"  # don't auto-start
        await rt.deliver(Envelope("x", a.id, "q", 1, time.time(), mode="queue"))
        # idle agent gets woken on first delivery, so check immediate/steer on a fresh idle agent
        b = rt.create_agent("t2")
        b.status = "done"  # terminal: deliver won't wake
        await rt.deliver(Envelope("x", b.id, "s", 1, time.time(), mode="steer"))
        await rt.deliver(Envelope("x", b.id, "i", 1, time.time(), mode="immediate"))
        assert not b._steer_inbox.empty() and not b._immediate_inbox.empty()

    @pytest.mark.asyncio
    async def test_unknown_recipient_is_noop(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        await rt.deliver(Envelope("x", "ghost", "hi", 1, time.time()))  # must not raise
        assert rt._messages_sent == []


# ─── threshold alerts ────────────────────────────────────────────────────────

class TestThresholds:
    def test_budget_and_turns_alerts(self, tmp_path):
        rt = mk(tmp_path, done_llm(), budget=1.0)
        a = rt.create_agent("t")
        a.quota.max_turns = 10
        a._turns = 6                      # 60% of turns
        rt.ledger.total_spent = 0.6       # 60% of budget
        rt._check_thresholds(a)
        assert "budget_0.5" in a._notified_thresholds
        assert "turns_0.5" in a._notified_thresholds
        assert not a._steer_inbox.empty()

    def test_context_and_time_alerts(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        a = rt.create_agent("t")
        a.context_limit, a.context_tokens = 1000, 600
        a.quota.time_limit = 100
        rt._start_time = time.time() - 60
        rt._check_thresholds(a)
        assert "context_0.5" in a._notified_thresholds
        assert "time_0.5" in a._notified_thresholds

    def test_alerts_fire_once(self, tmp_path):
        rt = mk(tmp_path, done_llm(), budget=1.0)
        a = rt.create_agent("t")
        rt.ledger.total_spent = 0.6
        rt._check_thresholds(a)
        first = len(a._notified_thresholds)
        rt._check_thresholds(a)            # same state → no new alerts
        assert len(a._notified_thresholds) == first

    def test_disabled_when_no_thresholds(self, tmp_path):
        rt = mk(tmp_path, done_llm(), notify_thresholds=[])
        a = rt.create_agent("t")
        rt.ledger.total_spent = 0.9
        rt._check_thresholds(a)
        assert a._notified_thresholds == set()


# ─── compression ─────────────────────────────────────────────────────────────

class TestCompress:
    @pytest.mark.asyncio
    async def test_short_history_unchanged(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        hist = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        assert await rt._compress(hist) == hist

    @pytest.mark.asyncio
    async def test_long_history_compressed(self, tmp_path):
        rt = mk(tmp_path, done_llm(), compress_keep_recent=2)
        hist = [{"role": "system", "content": "SYS"}]
        hist += [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        out = await rt._compress(hist)
        assert out[0]["content"] == "SYS"
        assert out[1]["content"].startswith("[Compressed")
        assert out[-2:] == hist[-2:]           # recent kept verbatim
        assert len(out) == 4                   # system + summary + 2 recent


# ─── _execute_tool ───────────────────────────────────────────────────────────

class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_unknown_meta_work_and_error(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        a = rt.create_agent("t")
        tools = build_all_tools()
        assert "Unknown tool" in (await rt._execute_tool(ToolCall("x", "nope", {}), a, tools))["error"]
        # meta tool
        gc = await rt._execute_tool(ToolCall("x", "get_cost", {}), a, tools)
        assert gc["agent_id"] == a.id
        # work tool
        cf = await rt._execute_tool(ToolCall("x", "create_file", {"path": "a.txt", "content": "y"}), a, tools)
        assert cf["bytes_written"] == 1
        # handler raising → wrapped error
        async def raising(*a, **k):
            raise RuntimeError("bad handler")
        tools2 = dict(tools)
        tools2["boom"] = {"handler": raising, "is_meta": False, "schema": {}}
        r = await rt._execute_tool(ToolCall("x", "boom", {}), a, tools2)
        assert "bad handler" in r["error"]


# ─── stats / status / shutdown ───────────────────────────────────────────────

class TestStatsStatusShutdown:
    @pytest.mark.asyncio
    async def test_stats_after_run(self, tmp_path):
        async def f(messages, model, tools=None, **kw):
            sysmsg = messages[0]["content"]
            n = sum(1 for m in messages if m.get("role") == "assistant")
            if "child" in sysmsg:
                return LLMResponse(tool_calls=[ToolCall("c", "set_status", {"status": "done", "result": "c"})], usage=usage(model))
            if n == 0:
                return LLMResponse(tool_calls=[ToolCall("s", "spawn", {"task": "child"})], usage=usage(model))
            if n == 1:
                return LLMResponse(tool_calls=[ToolCall("w", "wait", {"timeout": 5})], usage=usage(model))
            return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": "p"})], usage=usage(model))
        rt = mk(tmp_path, f, max_turns=10, max_agents=5)
        await rt.run("root")
        s = rt.stats()
        assert s["agents"]["total_spawned"] == 2
        assert s["tools"]["spawn_count"] == 1
        assert s["overview"]["total_tokens"] > 0
        assert len(s["per_agent"]) == 2
        assert s["communication"]["total_messages"] >= 1

    def test_stats_empty(self, tmp_path):
        assert "error" in mk(tmp_path, done_llm()).stats()

    @pytest.mark.asyncio
    async def test_status_keys(self, tmp_path):
        rt = mk(tmp_path, done_llm())
        await rt.run("t")
        st = rt.status()
        assert "alpha" in st["agents"] and "cost" in st and "scheduler" in st

    @pytest.mark.asyncio
    async def test_shutdown_cancels(self, tmp_path):
        async def slow(messages, model, tools=None, **kw):
            await asyncio.sleep(10)
            return LLMResponse(usage=usage(model))
        rt = mk(tmp_path, slow)
        a = rt.create_agent("t")
        rt.start_agent(a)
        await asyncio.sleep(0.05)
        await rt.shutdown()
        assert a.status == "done" and a._task.done()
