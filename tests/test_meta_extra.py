"""Branch coverage for nanoma/tools/meta.py beyond the happy paths."""

import time
import pytest

from nanoma.core import Runtime, RuntimeConfig
from nanoma.llm import LLMResponse, ToolCall
from nanoma.cost import UsageRecord
from nanoma.tools.meta import (
    meta_spawn, meta_kill, meta_send, meta_query, meta_wait,
    meta_get_cost, meta_submit,
)


@pytest.fixture
def rt(tmp_path):
    async def llm(messages, model, tools=None, **kw):
        return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": "ok"})],
                           usage=UsageRecord(model=model))
    return Runtime(config=RuntimeConfig(workspace_root=tmp_path / "ws", log_dir=None,
                                        budget=10.0, max_agents=50), llm_call=llm)


# ─── spawn ───────────────────────────────────────────────────────────────────

class TestSpawn:
    @pytest.mark.asyncio
    async def test_requires_task(self, rt):
        a = rt.create_agent("p")
        assert "error" in await meta_spawn({}, a, rt)

    @pytest.mark.asyncio
    async def test_max_depth(self, rt):
        a = rt.create_agent("p")
        a.depth = rt.config.max_depth
        assert "Max depth" in (await meta_spawn({"task": "c"}, a, rt))["error"]

    @pytest.mark.asyncio
    async def test_max_agents(self, tmp_path):
        async def llm(m, mo, t=None, **k):
            return LLMResponse(usage=UsageRecord(model=mo))
        r = Runtime(config=RuntimeConfig(workspace_root=tmp_path / "ws", log_dir=None, max_agents=1), llm_call=llm)
        a = r.create_agent("p")          # already at the cap of 1
        assert "Max agents" in (await meta_spawn({"task": "c"}, a, r))["error"]

    @pytest.mark.asyncio
    async def test_router_chooses_model(self, rt):
        rt.router = lambda task, remaining, allowed_models=None: "routed-model"
        a = rt.create_agent("p")
        out = await meta_spawn({"task": "c"}, a, rt)   # no explicit model
        assert rt.agents[out["agent_id"]].model == "routed-model"

    @pytest.mark.asyncio
    async def test_allowed_models_filter(self, tmp_path):
        async def llm(m, mo, t=None, **k):
            return LLMResponse(usage=UsageRecord(model=mo))
        r = Runtime(config=RuntimeConfig(workspace_root=tmp_path / "ws", log_dir=None,
                                         allowed_models=["allowed-x"]), llm_call=llm)
        a = r.create_agent("p")
        out = await meta_spawn({"task": "c", "model": "disallowed-y"}, a, r)
        assert r.agents[out["agent_id"]].model == "allowed-x"   # coerced into the allow-list

    @pytest.mark.asyncio
    async def test_delegate_terminates_parent(self, rt):
        a = rt.create_agent("p")
        out = await meta_spawn({"task": "c", "delegate": True}, a, rt)
        assert a.status == "done" and out["agent_id"] in a.result


# ─── kill / send / query ─────────────────────────────────────────────────────

class TestKillSendQuery:
    @pytest.mark.asyncio
    async def test_kill_not_found(self, rt):
        a = rt.create_agent("p")
        assert "not found" in (await meta_kill({"agent_id": "ghost"}, a, rt))["error"]

    @pytest.mark.asyncio
    async def test_send_invalid_mode(self, rt):
        a = rt.create_agent("a"); b = rt.create_agent("b")
        assert "Invalid mode" in (await meta_send({"to": b.id, "message": "x", "mode": "boom"}, a, rt))["error"]

    @pytest.mark.asyncio
    async def test_send_requires_to_and_message(self, rt):
        a = rt.create_agent("a")
        assert "error" in await meta_send({"message": "x"}, a, rt)
        assert "error" in await meta_send({"to": "b"}, a, rt)

    @pytest.mark.asyncio
    async def test_send_multicast(self, rt):
        a = rt.create_agent("a"); b = rt.create_agent("b"); c = rt.create_agent("c")
        out = await meta_send({"to": f"{b.id},{c.id}", "message": "hi", "mode": "queue"}, a, rt)
        assert out["delivered"] == 2

    @pytest.mark.asyncio
    async def test_query_single_not_found(self, rt):
        a = rt.create_agent("a")
        assert "not found" in (await meta_query({"agent_id": "ghost"}, a, rt))["error"]


# ─── wait ────────────────────────────────────────────────────────────────────

class TestWait:
    @pytest.mark.asyncio
    async def test_invalid_mode(self, rt):
        a = rt.create_agent("p")
        assert "mode must be" in (await meta_wait({"mode": "weird"}, a, rt))["error"]

    @pytest.mark.asyncio
    async def test_nothing_to_wait_for(self, rt):
        a = rt.create_agent("p")          # no children
        out = await meta_wait({}, a, rt)
        assert out["completed"] == [] and "Nothing" in out["note"]

    @pytest.mark.asyncio
    async def test_mode_any_returns_on_first(self, rt):
        p = rt.create_agent("p")
        c1 = rt.create_agent("c1", parent=p.id); c2 = rt.create_agent("c2", parent=p.id)
        p.children |= {c1.id, c2.id}
        c1.status, c1.result = "done", "r1"   # c2 still running
        out = await meta_wait({"mode": "any"}, p, rt)
        assert len(out["completed"]) == 1 and out["completed"][0]["id"] == c1.id

    @pytest.mark.asyncio
    async def test_timeout_reports_pending(self, rt):
        p = rt.create_agent("p")
        c = rt.create_agent("c", parent=p.id); p.children.add(c.id)   # never completes
        out = await meta_wait({"agent_ids": [c.id], "timeout": 0.2}, p, rt)
        assert out.get("interrupted") and out["reason"] == "timeout"
        assert out["pending"] and out["pending"][0]["id"] == c.id


# ─── get_cost / submit ───────────────────────────────────────────────────────

class TestGetCostSubmit:
    @pytest.mark.asyncio
    async def test_get_cost_includes_time_remaining(self, rt):
        a = rt.create_agent("p")
        a.quota.time_limit = 100
        rt._start_time = time.time() - 10
        out = await meta_get_cost({}, a, rt)
        assert "time_remaining" in out and out["time_remaining"] <= 90

    @pytest.mark.asyncio
    async def test_submit_errors(self, rt):
        a = rt.create_agent("p")
        assert "path required" in (await meta_submit({}, a, rt))["error"]
        assert "Not found" in (await meta_submit({"path": "nope.txt"}, a, rt))["error"]

    @pytest.mark.asyncio
    async def test_submit_copies_to_shared(self, rt):
        a = rt.create_agent("p")
        (a.workspace / "out.txt").write_text("deliverable")
        out = await meta_submit({"path": "out.txt", "description": "final"}, a, rt)
        assert out["submitted"] == "out.txt"
        assert (rt._tool_context.shared_dir / "out.txt").read_text() == "deliverable"
        assert len(a.artifacts) == 1 and a.artifacts[0].description == "final"
