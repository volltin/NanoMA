"""Orchestration runtime: agent lifecycle, the ReAct loop, messaging, and stats."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable

from nanoma import prompts
from nanoma.config import RuntimeConfig
from nanoma.cost import CostLedger
from nanoma.llm import (
    LLMResponse, Message, ToolCall, ToolDef,
    count_message_tokens, estimate_tokens, openai_compatible_call, set_log_dir,
    anthropic_messages_call, is_anthropic_model,
)
from nanoma.scheduler import Scheduler
from nanoma.state import Agent, Envelope, IdGenerator, ResourceQuota, ToolContext
from nanoma.tools import build_all_tools

logger = logging.getLogger("nanoma")

# Injected after a turn that produced no tool call, to stop a verbose model from
# burning whole round-trips on prose instead of acting (or finishing).
_NO_TOOL_NUDGE = (
    "(No tool call in your last message. Every step must call a tool. "
    'If the task is fully complete, call set_status(status="done", result=...) now. '
    "Otherwise call the next tool to make progress.)"
)


def _peak_concurrency(events: list[dict]) -> int:
    """Max number of agents alive at once, from agent_new → done/failed events.

    Each event is keyed by its own agent id (unlike "spawn", which is keyed by the
    parent). An agent with no recorded end is treated as alive through the last event.
    """
    intervals: dict[str, list] = {}
    last_t = 0.0
    for e in events:
        aid, t = e["agent"], e.get("t", 0.0)
        last_t = max(last_t, t)
        if e["event"] == "agent_new":
            intervals.setdefault(aid, [t, None])
        elif e["event"] in ("done", "failed") and aid in intervals:
            intervals[aid][1] = t
    if not intervals:
        return 0
    open_end = last_t + 1.0  # sentinel strictly after every recorded event
    ticks = []
    for start, end in intervals.values():
        ticks.append((start, 1))
        ticks.append((end if end is not None else open_end, -1))
    # Process exits (-1) before entries (+1) at the same timestamp so back-to-back
    # sequential agents don't falsely count as concurrent.
    ticks.sort(key=lambda x: (x[0], x[1]))
    peak = cur = 0
    for _, delta in ticks:
        cur += delta
        peak = max(peak, cur)
    return peak


class Runtime:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        llm_call: Callable | None = None,
        router: Callable | None = None,
        on_event: Callable | None = None,
    ):
        self.config = config or RuntimeConfig()
        self.ledger = CostLedger(total_budget=self.config.budget)
        self.agents: dict[str, Agent] = {}
        self._id_gen = IdGenerator()
        self._tool_context = ToolContext(
            shared_dir=self.config.workspace_root / self.config.shared_dir,
            workspace_root=self.config.workspace_root,
            shell_max_output=self.config.shell_max_output,
            file_read_max_chars=self.config.file_read_max_chars,
            file_list_max_entries=self.config.file_list_max_entries,
            grep_max_results=self.config.grep_max_results,
        )
        self.llm_call = llm_call or openai_compatible_call
        self.router = router
        self.scheduler = Scheduler(max_concurrent=self.config.max_concurrent_llm)
        self.on_event = on_event or (lambda e: None)
        self._start_time = time.time()
        self._events: list[dict] = []
        self._messages_sent: list[tuple[str, str, int]] = []  # (from, to, tokens)
        self._emit_lock = threading.Lock()

        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        self._tool_context.shared_dir.mkdir(parents=True, exist_ok=True)
        if self.config.log_dir:
            set_log_dir(self.config.log_dir)

    # ─── Agent lifecycle ─────────────────────────────────────────────────

    def _registry(self):
        from nanoma.models import get_registry
        try:
            return get_registry()
        except Exception:
            return None

    def create_agent(
        self,
        task: str,
        model: str | None = None,
        quota: ResourceQuota | None = None,
        parent: str | None = None,
        depth: int = 0,
        workspace: Path | None = None,
    ) -> Agent:
        agent_id = self._id_gen.next()
        model = model or self.config.default_model
        quota = quota or ResourceQuota(
            budget=self.config.budget,
            time_limit=self.config.time_limit,
            max_turns=self.config.max_turns,
        )
        # An explicit workspace lets the agent operate in an existing directory (e.g. a
        # benchmark's WORKDIR); otherwise each agent gets a private dir under the root.
        workspace = Path(workspace) if workspace is not None else (self.config.workspace_root / agent_id)
        workspace.mkdir(parents=True, exist_ok=True)

        reg = self._registry()

        # Fusion model → run as a Mixture-of-Agents orchestrator. Not a special call
        # path: the agent runs on the judge's concrete model and is told (via a prompt
        # directive) to spawn the panel as real sub-agents and synthesize their answers.
        fusion_name = None
        fusion_directive = ""
        if reg is not None:
            fdef = reg.get_fusion(model)
            if fdef:
                fusion_name = fdef.name
                fusion_directive = prompts.build_fusion_directive(reg, fdef)
                model = prompts.concrete_model(reg, fdef.judge or (fdef.panel[0] if fdef.panel else model))

        parent_context = None
        if parent and parent in self.agents:
            p = self.agents[parent]
            siblings = [sid for sid in p.children if sid in self.agents]
            sibling_info = ", ".join(f"{sid}({self.agents[sid].task[:30]})" for sid in siblings[:5])
            parent_context = {
                "parent_id": parent, "parent_task": p.task[:100],
                "siblings": sibling_info, "depth": depth,
            }

        system_prompt = prompts.build_system_prompt(
            reg, agent_id=agent_id, task=task, workspace=workspace,
            shared_dir=self._tool_context.shared_dir,
            time_limit=self.config.time_limit, parent_context=parent_context,
        )
        if fusion_directive:
            system_prompt += fusion_directive

        agent = Agent(
            id=agent_id, task=task, model=model, quota=quota,
            parent=parent, depth=depth, workspace=workspace,
            history=[
                {"role": "system", "content": system_prompt},
                # Kickoff user turn: a fresh agent has no user message, which
                # Anthropic-backed providers reject ("field messages is required").
                {"role": "user", "content": "Begin working on your task now."},
            ],
        )
        if reg is not None:
            try:
                agent.context_limit = reg.context_limit(model)
            except Exception:
                pass

        self.agents[agent_id] = agent
        agent._created_at = time.time()
        if parent and parent in self.agents:
            self.agents[parent].children.add(agent_id)

        self._emit(agent_id, "agent_new", {
            "task": task, "model": model, "fusion": fusion_name,
            "budget": quota.budget if math.isfinite(quota.budget) else None,
            "parent": parent, "depth": depth,
        })
        return agent

    def start_agent(self, agent: Agent):
        agent._task = asyncio.ensure_future(self._agent_loop(agent))

    async def run(self, task: str, model: str | None = None, workspace: Path | None = None) -> str:
        """Run a single root agent to completion, then shut down any stragglers.

        If ``workspace`` is given, the root agent operates in that directory (its shell
        CWD, file tools, and system prompt all agree) instead of a private sub-dir.
        """
        root = self.create_agent(task, model=model, workspace=workspace)
        self.start_agent(root)
        await root._task
        running = [a for a in self.agents.values()
                   if a.status in ("running", "idle") and a.id != root.id]
        if running:
            await self.shutdown()
        return root.result or ""

    # ─── Messaging ───────────────────────────────────────────────────────

    async def deliver(self, envelope: Envelope):
        agent = self.agents.get(envelope.to_id)
        if not agent:
            return
        self._messages_sent.append((envelope.from_id, envelope.to_id, envelope.tokens))
        if envelope.mode == "immediate":
            agent._immediate_inbox.put_nowait(envelope)
        elif envelope.mode == "steer":
            agent._steer_inbox.put_nowait(envelope)
        else:
            agent._queue_inbox.put_nowait(envelope)
        if agent.status == "idle":
            agent.status = "running"
            if agent._task is None or agent._task.done():
                self.start_agent(agent)

    async def _model_call(self, model: str, messages: list[Message], tools: list[ToolDef] | None) -> LLMResponse:
        """Resolve a model alias to a concrete name and call the LLM.

        The single place model strings are resolved before inference. Fusion models
        are already expanded to a concrete judge model at create_agent time.
        """
        from nanoma.models import get_registry
        resolved = get_registry().resolve(model)
        # Route Claude-family models to the native Anthropic endpoint (for prompt
        # caching) when enabled — but never override an injected llm_call (e.g. tests).
        call = self.llm_call
        if (self.config.anthropic_native and call is openai_compatible_call
                and is_anthropic_model(resolved)):
            call = anthropic_messages_call
        return await call(
            messages, resolved, tools,
            temperature=self.config.temperature, retry_config=self.config.retry,
        )

    # ─── ReAct loop ──────────────────────────────────────────────────────

    async def _agent_loop(self, agent: Agent):
        all_tools = build_all_tools()
        tool_schemas = [t["schema"] for t in all_tools.values()]

        try:
            while agent.status == "running":
                agent._turns += 1
                agent._last_active = time.time()

                if agent._turns > agent.quota.max_turns:
                    agent.status = "done"
                    agent.result = agent.result or "[Max turns reached]"
                    break

                if agent.quota.time_limit > 0:
                    elapsed = time.time() - self._start_time
                    if elapsed >= agent.quota.time_limit:
                        agent.status = "done"
                        agent.result = agent.result or f"[Time limit at {elapsed:.0f}s]"
                        break

                if self.ledger.remaining() <= 0:
                    agent.status = "failed"
                    agent.result = agent.result or "[GLOBAL BUDGET EXHAUSTED]"
                    self._emit(agent.id, "failed", {
                        "reason": "budget_exhausted", "status": "failed",
                        "turns": agent._turns, "result": agent.result,
                    })
                    break

                self._check_thresholds(agent)

                self._inject_messages(agent, agent._queue_inbox)
                self._inject_messages(agent, agent._steer_inbox)

                # Context compression
                agent.context_tokens = count_message_tokens(agent.history)
                if agent.context_tokens > int(agent.context_limit * self.config.context_compress_ratio):
                    agent.history = await self._compress(agent.history)
                    agent.context_tokens = count_message_tokens(agent.history)

                # LLM call
                await self.scheduler.acquire()
                try:
                    response = await self._model_call(agent.model, agent.history, tool_schemas)
                finally:
                    self.scheduler.release()

                cost = self.ledger.record(agent.id, response.usage)
                agent.tokens_consumed += response.usage.total_tokens
                agent.quota.budget -= cost
                self._emit(agent.id, "llm_done", {
                    "tokens": response.usage.total_tokens,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cached": response.usage.cached_input_tokens,
                    "cost": round(cost, 6), "model": agent.model,
                    "tool_calls": [tc.name for tc in response.tool_calls] if response.tool_calls else [],
                    "has_content": bool(response.content),
                    "content_preview": (response.content or "")[:200],
                })

                if response.tool_calls:
                    agent.history.append({
                        "role": "assistant", "content": response.content,
                        "tool_calls": [
                            {"id": tc.id, "type": "function",
                             "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                            for tc in response.tool_calls
                        ],
                    })
                    for i, tc in enumerate(response.tool_calls):
                        # Immediate-interrupt check: abandon remaining calls this turn
                        if not agent._immediate_inbox.empty():
                            self._inject_messages(agent, agent._immediate_inbox)
                            for skipped in response.tool_calls[i:]:
                                agent.history.append({
                                    "role": "tool", "tool_call_id": skipped.id,
                                    "content": json.dumps({"interrupted": True}),
                                })
                            break

                        result = await self._execute_tool(tc, agent, all_tools)
                        result_json = json.dumps(result, ensure_ascii=False, default=str)
                        agent.history.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": result_json[:8000],
                        })
                        self._emit(agent.id, "tool_call", {
                            "tool": tc.name, "args": tc.arguments,
                            "result": result_json[:5000], "result_full_len": len(result_json),
                        })

                    self._inject_messages(agent, agent._steer_inbox)

                else:
                    # No tool call this turn. A chatty model can stall here, burning
                    # whole LLM round-trips on prose. Record the turn, then nudge it to
                    # act (or finish) so the next turn does real work instead of rambling.
                    agent.history.append({"role": "assistant", "content": response.content or "(empty)"})
                    agent.history.append({"role": "user", "content": _NO_TOOL_NUDGE})

                if agent.status != "running":
                    break
                if not self.ledger.can_afford(0):
                    agent.status = "failed"
                    agent.result = "Budget exhausted"
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Agent {agent.id} crashed: {e}")
            agent.status = "failed"
            agent.result = f"Crash: {e}"
        finally:
            if agent.status in ("done", "failed"):
                if agent.parent and agent.parent in self.agents:
                    death_msg = f"[Agent {agent.id} finished: {agent.status}] {(agent.result or '')[:200]}"
                    msg_tokens = estimate_tokens(death_msg)
                    await self.deliver(Envelope(
                        from_id="system", to_id=agent.parent, content=death_msg,
                        tokens=msg_tokens, timestamp=time.time(), mode="steer",
                    ))
                    self._emit(agent.parent, "send", {
                        "from": "system", "to": agent.parent,
                        "mode": "steer", "tokens": msg_tokens, "message": death_msg,
                    })
                self._emit(agent.id, agent.status, {
                    "status": agent.status, "turns": agent._turns,
                    "tokens": agent.tokens_consumed, "result": (agent.result or "")[:500],
                    "artifacts": [a.path for a in agent.artifacts],
                })

    # ─── Turn helpers ────────────────────────────────────────────────────

    def _check_thresholds(self, agent: Agent):
        """Inject resource alerts (time/turns/budget/context) when usage crosses a
        configured threshold, delivered as a system→agent steer message."""
        thresholds = self.config.notify_thresholds
        if not thresholds:
            return
        elapsed = time.time() - self._start_time
        alerts = []

        if agent.quota.time_limit > 0:
            frac = elapsed / agent.quota.time_limit
            for t in thresholds:
                key = f"time_{t}"
                if frac >= t and key not in agent._notified_thresholds:
                    agent._notified_thresholds.add(key)
                    alerts.append(f"⏱️ TIME {int(t*100)}% used — {max(0, agent.quota.time_limit - elapsed):.0f}s remaining")

        turn_frac = agent._turns / agent.quota.max_turns
        for t in thresholds:
            key = f"turns_{t}"
            if turn_frac >= t and key not in agent._notified_thresholds:
                agent._notified_thresholds.add(key)
                alerts.append(f"🔄 TURNS {int(t*100)}% used — {agent.quota.max_turns - agent._turns} turns remaining")

        total_budget = self.ledger.total_budget
        if total_budget > 0:
            spent_frac = max(0.0, min(1.0, self.ledger.total_spent / total_budget))
            for t in thresholds:
                key = f"budget_{t}"
                if spent_frac >= t and key not in agent._notified_thresholds:
                    agent._notified_thresholds.add(key)
                    alerts.append(f"BUDGET {int(t*100)}% used — ${self.ledger.remaining():.4f} remaining of ${total_budget:.4f}")

        if agent.context_limit > 0 and agent.context_tokens > 0:
            ctx_frac = agent.context_tokens / agent.context_limit
            for t in thresholds:
                key = f"context_{t}"
                if ctx_frac >= t and key not in agent._notified_thresholds:
                    agent._notified_thresholds.add(key)
                    alerts.append(f"🧠 CONTEXT {int(t*100)}% full — {int((1 - ctx_frac) * 100)}% capacity left. Wrap up soon; old turns are auto-compacted.")

        if alerts:
            notice = "[Resource Alert]\n" + "\n".join(alerts)
            agent._steer_inbox.put_nowait(Envelope(
                from_id="system", to_id=agent.id, content=notice,
                tokens=estimate_tokens(notice), timestamp=time.time(), mode="steer",
            ))
            self._emit(agent.id, "send", {
                "from": "system", "to": agent.id, "mode": "steer",
                "tokens": estimate_tokens(notice), "message": notice,
            })

    def _inject_messages(self, agent: Agent, queue: asyncio.Queue[Envelope]) -> list[Envelope]:
        msgs = []
        while not queue.empty():
            try:
                msgs.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        msgs.sort(key=lambda m: m.priority, reverse=True)
        for msg in msgs:
            agent.history.append({"role": "user", "content": f"[Message from {msg.from_id}]: {msg.content}"})
        return msgs

    async def _execute_tool(self, tc: ToolCall, agent: Agent, tools: dict) -> Any:
        tool_info = tools.get(tc.name)
        if not tool_info:
            return {"error": f"Unknown tool: {tc.name}"}
        handler = tool_info["handler"]
        try:
            if tool_info.get("is_meta"):
                return await handler(tc.arguments, agent, self)
            return await handler(tc.arguments, agent.workspace, self._tool_context)
        except Exception as e:
            return {"error": str(e)}

    async def _compress(self, history: list[Message], keep_recent: int | None = None) -> list[Message]:
        """Compress old messages into a single summary, keeping recent ones intact."""
        keep_recent = keep_recent or self.config.compress_keep_recent
        if len(history) <= keep_recent + 2:
            return history
        system = history[0]
        old = history[1:-keep_recent]
        recent = history[-keep_recent:]
        max_msgs = self.config.compress_max_messages
        max_chars = self.config.compress_max_chars
        parts = []
        for msg in (old[-max_msgs:] if max_msgs > 0 else old):
            content = msg.get("content") or ""
            if max_chars > 0:
                content = content[:max_chars]
            if content:
                parts.append(f"[{msg.get('role', '')}]: {content}")
        summary = {"role": "assistant", "content": f"[Compressed {len(old)} messages]\n" + "\n".join(parts)}
        return [system, summary] + recent

    # ─── Events ──────────────────────────────────────────────────────────

    def _emit(self, agent_id: str, event_type: str, data: dict | None = None):
        now = time.time()
        event = {"agent": agent_id, "event": event_type, "data": data or {}, "t": now}
        self._events.append(event)
        self.on_event(event)
        if self.config.log_dir:
            # Persist the same shape the on_event callback receives, plus timing.
            # (Key is "event" — consistent with self._events and on_event consumers.)
            trace_event = {
                "event": event_type, "agent": agent_id,
                "ts": now, "rel_ts": round(now - self._start_time, 3),
                "data": data or {},
            }
            try:
                line = json.dumps(trace_event, ensure_ascii=False, default=str, allow_nan=False) + "\n"
                with self._emit_lock:
                    with open(self.config.log_dir / "events.jsonl", "a") as f:
                        f.write(line)
            except Exception:
                pass

    # ─── Public API ──────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "agents": {
                aid: {"status": a.status, "task": a.task[:80], "turns": a._turns}
                for aid, a in self.agents.items()
            },
            "cost": self.ledger.summary(),
            "scheduler": self.scheduler.stats,
        }

    def stats(self) -> dict[str, Any]:
        """Comprehensive post-run statistics."""
        elapsed = time.time() - self._start_time
        agents = list(self.agents.values())
        n = len(agents)
        if n == 0:
            return {"error": "no agents"}

        total_tokens = sum(a.tokens_consumed for a in agents)
        token_per_agent = [a.tokens_consumed for a in agents]

        statuses: dict[str, int] = {}
        for a in agents:
            statuses[a.status] = statuses.get(a.status, 0) + 1
        max_depth = max((a.depth for a in agents), default=0)
        turns_list = [a._turns for a in agents]

        # Peak concurrency from agent birth (agent_new) → death (done/failed).
        # NB: "spawn" events are keyed by the *parent*, so they cannot be used here.
        peak_concurrent = _peak_concurrency(self._events)

        tool_counts: dict[str, int] = {}
        for e in self._events:
            if e["event"] == "tool_call":
                name = e["data"].get("tool", "?")
                tool_counts[name] = tool_counts.get(name, 0) + 1
        tool_sorted = sorted(tool_counts.items(), key=lambda x: -x[1])

        # Communication graph
        comm_edges: dict[tuple[str, str], int] = {}
        agents_who_sent: set[str] = set()
        agents_who_received: set[str] = set()
        out_degree: dict[str, set] = {}
        in_degree: dict[str, set] = {}
        for frm, to, _ in self._messages_sent:
            comm_edges[(frm, to)] = comm_edges.get((frm, to), 0) + 1
            agents_who_sent.add(frm)
            agents_who_received.add(to)
            out_degree.setdefault(frm, set()).add(to)
            in_degree.setdefault(to, set()).add(frm)
        busiest_sender = max(out_degree.items(), key=lambda x: len(x[1]), default=("none", set()))
        busiest_receiver = max(in_degree.items(), key=lambda x: len(x[1]), default=("none", set()))

        cost = self.ledger.total_spent
        return {
            "overview": {
                "elapsed_seconds": round(elapsed, 1),
                "total_cost_usd": round(cost, 4),
                "total_tokens": total_tokens,
                "tokens_per_dollar": int(total_tokens / cost) if cost > 0 else 0,
                "cache_hit_rate": round(self.ledger.cache_hit_rate(), 4),
            },
            "agents": {
                "total_spawned": n,
                "peak_concurrent": peak_concurrent,
                "max_depth": max_depth,
                "status_breakdown": statuses,
                "avg_turns": round(sum(turns_list) / n, 1),
                "max_turns": max(turns_list),
                "min_turns": min(turns_list),
                "avg_tokens": int(total_tokens / n),
                "max_tokens_agent": max(agents, key=lambda a: a.tokens_consumed).id,
                "max_tokens_value": max(token_per_agent),
            },
            "tools": {
                "total_calls": sum(tool_counts.values()),
                "unique_tools_used": len(tool_counts),
                "top_tools": tool_sorted[:10],
                "spawn_count": tool_counts.get("spawn", 0),
                "send_count": tool_counts.get("send", 0),
                "query_count": tool_counts.get("query", 0),
                "wait_count": tool_counts.get("wait", 0),
            },
            "communication": {
                "total_messages": len(self._messages_sent),
                "total_message_tokens": sum(t for _, _, t in self._messages_sent),
                "unique_edges": len(comm_edges),
                "agents_who_sent": len(agents_who_sent),
                "agents_who_received": len(agents_who_received),
                "max_out_degree": max((len(v) for v in out_degree.values()), default=0),
                "max_in_degree": max((len(v) for v in in_degree.values()), default=0),
                "busiest_sender": f"{busiest_sender[0]} → {len(busiest_sender[1])} recipients",
                "busiest_receiver": f"{busiest_receiver[0]} ← {len(busiest_receiver[1])} senders",
                "top_edges": sorted(comm_edges.items(), key=lambda x: -x[1])[:5],
            },
            "per_agent": [
                {
                    "id": a.id, "status": a.status, "depth": a.depth,
                    "turns": a._turns, "tokens": a.tokens_consumed,
                    "children": len(a.children),
                }
                for a in sorted(agents, key=lambda a: -a.tokens_consumed)
            ],
        }

    async def shutdown(self):
        """Force-terminate all running agents."""
        for a in self.agents.values():
            if a.status in ("running", "idle"):
                a.status = "done"
            if a._task and not a._task.done():
                a._task.cancel()
        tasks = [a._task for a in self.agents.values() if a._task and not a._task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
