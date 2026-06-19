"""Meta-tools: spawn, kill, send, query, wait, get_cost, set_status, submit."""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.tools.spec import Tool, arg, registry

if TYPE_CHECKING:
    from nanoma.core import Agent, Runtime


# ─── spawn ───────────────────────────────────────────────────────────────────

async def meta_spawn(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Create a child agent."""
    from nanoma.core import ResourceQuota

    task = args.get("task", "")
    model = args.get("model")
    delegate = args.get("delegate", False)

    if not task:
        return {"error": "task is required"}
    if agent.depth + 1 > runtime.config.max_depth:
        return {"error": f"Max depth ({runtime.config.max_depth}) exceeded"}
    if len(runtime.agents) >= runtime.config.max_agents:
        return {"error": f"Max agents ({runtime.config.max_agents}) reached"}

    # Model selection
    if not model:
        if runtime.router:
            model = runtime.router(task, runtime.ledger.remaining(), allowed_models=runtime.config.allowed_models)
        else:
            model = runtime.config.default_model
    if runtime.config.allowed_models:
        # Compare on resolved names so aliases satisfy the allow-list too.
        try:
            from nanoma.models import get_registry
            reg = get_registry()
            allowed_resolved = {reg.resolve(a) for a in runtime.config.allowed_models}
            if reg.resolve(model) not in allowed_resolved:
                model = runtime.config.allowed_models[0]
        except Exception:
            if model not in runtime.config.allowed_models:
                model = runtime.config.allowed_models[0]

    child_quota = ResourceQuota(budget=float("inf"), time_limit=agent.quota.time_limit, max_turns=agent.quota.max_turns)
    child = runtime.create_agent(task=task, model=model, quota=child_quota, parent=agent.id, depth=agent.depth + 1)
    runtime.start_agent(child)

    # Emit spawn event from parent's perspective (trace)
    runtime._emit(agent.id, "spawn", {"child": child.id, "task": task[:100], "model": model})

    if delegate:
        agent.status = "done"
        agent.result = f"[Delegated to {child.id}]"

    return {"agent_id": child.id, "model": model}


# ─── kill ────────────────────────────────────────────────────────────────────

async def meta_kill(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Terminate an agent."""
    target_id = args.get("agent_id", "")
    target = runtime.agents.get(target_id)
    if not target:
        return {"error": f"Agent '{target_id}' not found"}

    # Permission: self or descendant
    if target_id != agent.id and not _is_descendant(target_id, agent.id, runtime):
        return {"error": "Cannot kill — not a descendant"}

    target.status = "done"
    if target._task and not target._task.done():
        target._task.cancel()

    return {"killed": target_id}


# ─── send ────────────────────────────────────────────────────────────────────

async def meta_send(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Send a message to agent(s). No broadcast — must specify IDs."""
    from nanoma.core import Envelope
    from nanoma.llm import estimate_tokens

    to = args.get("to", "")
    message = args.get("message", "")
    mode = args.get("mode", "queue")

    if not to:
        return {"error": "'to' required (agent_id or list of IDs)"}
    if not message:
        return {"error": "'message' required"}
    if mode not in ("immediate", "steer", "queue"):
        return {"error": f"Invalid mode '{mode}'"}

    recipients = [r.strip() for r in to.split(",")] if isinstance(to, str) else to
    msg_tokens = estimate_tokens(message)
    delivered = 0

    for rid in recipients:
        if rid not in runtime.agents:
            continue
        await runtime.deliver(Envelope(
            from_id=agent.id, to_id=rid, content=message,
            tokens=msg_tokens, timestamp=time.time(), mode=mode,
        ))
        delivered += 1
        # Emit send event (trace)
        runtime._emit(agent.id, "send", {
            "from": agent.id, "to": rid, "mode": mode,
            "tokens": msg_tokens, "message": message,
            "message_chars": len(message),
        })

    return {"delivered": delivered, "tokens": msg_tokens, "mode": mode}


# ─── query ───────────────────────────────────────────────────────────────────

async def meta_query(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Query agents. messages=N for last N messages, messages=-1 for all."""
    target_id = args.get("agent_id")
    messages_n = args.get("messages", 0)  # 0=meta only, N=last N, -1=all

    if target_id:
        target = runtime.agents.get(target_id)
        if not target:
            return {"error": f"Agent '{target_id}' not found"}
        result: dict[str, Any] = {
            "id": target.id,
            "status": target.status,
            "task": target.task[:100],
            "model": target.model,
            "parent": target.parent,
            "children": list(target.children),
            "result": target.result if target.result else None,
            "artifacts": [a.path for a in target.artifacts],
        }
        # Include messages if requested
        if messages_n != 0:
            history = target.history[1:]  # skip system prompt
            if messages_n > 0:
                history = history[-messages_n:]
            # Serialize compactly
            result["messages"] = [
                {"role": m.get("role", ""), "content": (m.get("content") or "")[:500]}
                for m in history if m.get("role") in ("user", "assistant")
            ]
        return result

    # List all agents (lightweight)
    agents_list = []
    for a in runtime.agents.values():
        agents_list.append({
            "id": a.id, "status": a.status, "task": a.task[:80],
        })
    return {"agents": agents_list, "count": len(agents_list)}


# ─── wait ────────────────────────────────────────────────────────────────────

async def meta_wait(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Block until agents finish. mode='all' waits for everyone, mode='any' returns on first completion."""
    target_ids = args.get("agent_ids", [])
    timeout = args.get("timeout", 300.0)
    mode = args.get("mode", "all")  # "all" or "any"

    if mode not in ("all", "any"):
        return {"error": "mode must be 'all' or 'any'"}

    if not target_ids:
        target_ids = list(agent.children)
    if not target_ids:
        return {"completed": [], "pending": [], "note": "Nothing to wait for"}

    # Cap timeout vs time limit
    if agent.quota.time_limit > 0:
        remaining = agent.quota.time_limit - (time.time() - runtime._start_time)
        timeout = min(timeout, max(0, remaining * 0.9))  # leave 10% margin for cleanup

    completed = []
    interrupted = False
    reason = None

    try:
        async with asyncio.timeout(timeout):
            while True:
                # Only an *immediate* message aborts a wait. Steer/queue messages
                # (including the "[Agent X finished]" notices for the very children
                # we're waiting on) must NOT interrupt — otherwise mode="all" returns
                # after the first child and the caller has to re-issue wait N times.
                # Those messages stay queued and are injected after wait returns.
                if not agent._immediate_inbox.empty():
                    interrupted = True
                    reason = "immediate_message"
                    break

                for tid in target_ids:
                    t = runtime.agents.get(tid)
                    if t and t.status in ("done", "failed"):
                        if not any(c["id"] == tid for c in completed):
                            completed.append({
                                "id": tid, "status": t.status,
                                "result": (t.result or "")[:500],
                                "artifacts": [a.path for a in t.artifacts],
                            })

                # mode="any": return as soon as at least one completes
                if mode == "any" and completed:
                    break
                # mode="all": wait for every target
                if mode == "all":
                    all_done = all(
                        any(c["id"] == tid for c in completed)
                        for tid in target_ids
                    )
                    if all_done:
                        break

                await asyncio.sleep(0.5)
    except (asyncio.TimeoutError, TimeoutError):
        interrupted = True
        reason = "timeout"

    # Snapshot pending agents
    pending = []
    for tid in target_ids:
        if not any(c["id"] == tid for c in completed):
            t = runtime.agents.get(tid)
            pending.append({"id": tid, "status": t.status if t else "not_found"})

    result = {"completed": completed, "pending": pending}
    if interrupted:
        result["interrupted"] = True
        result["reason"] = reason
    return result


# ─── get_cost ────────────────────────────────────────────────────────────────

async def meta_get_cost(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Get resource status."""
    elapsed = time.time() - runtime._start_time
    context_pct = round(agent.context_tokens / max(1, agent.context_limit) * 100, 1)
    result = {
        "agent_id": agent.id,
        "spawned_by": agent.parent,
        "sub_agents": list(agent.children),
        "context_tokens": agent.context_tokens,
        "context_limit": agent.context_limit,
        "context_usage_pct": context_pct,
        "tokens_consumed": agent.tokens_consumed,
        "turns_used": agent._turns,
        "max_turns": agent.quota.max_turns,
        "budget_total": runtime.ledger.total_budget,
        "budget_spent": round(runtime.ledger.total_spent, 6),
        "budget_remaining": round(runtime.ledger.remaining(), 6),
        "elapsed_seconds": round(elapsed, 1),
        "total_agents": len(runtime.agents),
    }
    if agent.quota.time_limit > 0:
        result["time_remaining"] = round(max(0, agent.quota.time_limit - elapsed), 1)
    return result


# ─── set_status ──────────────────────────────────────────────────────────────

async def meta_set_status(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Mark done or idle."""
    status = args.get("status", "done")
    result = args.get("result", "")
    if status not in ("done", "idle"):
        return {"error": "status must be 'done' or 'idle'"}
    agent.status = status
    if result:
        agent.result = result
    return {"status": status}


# ─── submit ──────────────────────────────────────────────────────────────────

async def meta_submit(args: dict[str, Any], agent: "Agent", runtime: "Runtime") -> dict[str, Any]:
    """Submit a file as artifact."""
    from nanoma.core import Artifact
    path_str = args.get("path", "")
    if not path_str:
        return {"error": "path required"}
    path = Path(path_str)
    if not path.is_absolute():
        path = agent.workspace / path
    if not path.exists():
        return {"error": f"Not found: {path}"}

    # Copy to shared
    shared = runtime._tool_context.shared_dir
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, shared / path.name)

    artifact = Artifact(path=path_str, absolute_path=path, description=args.get("description", ""), agent_id=agent.id)
    agent.artifacts.append(artifact)
    return {"submitted": path_str, "shared_copy": str(shared / path.name)}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_descendant(target_id: str, ancestor_id: str, runtime: "Runtime") -> bool:
    visited = set()
    stack = list(runtime.agents[ancestor_id].children) if ancestor_id in runtime.agents else []
    while stack:
        current = stack.pop()
        if current == target_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        if current in runtime.agents:
            stack.extend(runtime.agents[current].children)
    return False


# ─── Registry ────────────────────────────────────────────────────────────────

def _m(name: str, description: str, handler, *args) -> Tool:
    """Helper: a meta tool (is_meta=True) with structured arg specs."""
    return Tool(name, description, handler, args=list(args), is_meta=True)


META_TOOLS: dict[str, Tool] = registry(
    _m("spawn",
       "Create a new agent that starts immediately and runs in parallel. The child agent gets its "
       "own workspace, fresh context, and the same tools. It cannot see your conversation history — "
       "include all necessary context in the task description. You become its parent and receive a "
       "notification when it finishes.",
       meta_spawn,
       arg("task", str, "Complete task description for the new agent. Include all context it needs — it cannot see your history."),
       arg("model", str, "LLM model for the child (omit to use the default). Accepts a concrete model id, a tier alias (e.g. 'nano', 'mini', 'pro'), or a fusion model (e.g. 'fusion' / 'fusion-quality') — a fusion model spawns a panel of models that each solve the task and a judge that synthesizes their answers, for higher quality on hard tasks. See models.yaml for available names.", required=False),
       arg("delegate", bool, "If true, you terminate immediately and the child inherits your role", default=False)),

    _m("kill",
       "Terminate an agent. You can only kill yourself or your descendants (children, grandchildren, "
       "etc). The agent stops immediately.",
       meta_kill,
       arg("agent_id", str, "ID of the agent to terminate")),

    _m("send",
       "Send a message to one or more agents. The message appears in their conversation as a user "
       "message. Use to communicate results, give instructions, or coordinate. Modes: 'queue' "
       "(delivered next turn, default), 'steer' (delivered after current tool calls), 'immediate' "
       "(interrupts current processing). Sending to an 'idle' agent wakes it up.",
       meta_send,
       arg("to", str, "Recipient agent ID. For multicast, pass comma-separated IDs."),
       arg("message", str, "Message content to deliver"),
       arg("mode", str, "Delivery priority: queue (next turn), steer (after current tools), immediate (interrupts)",
           enum=["immediate", "steer", "queue"], default="queue")),

    _m("query",
       "Discover agents in the system. Without agent_id: returns list of all agents with {id, status, "
       "task}. With agent_id: returns detailed info including parent, children, result, artifacts. Use "
       "messages=N to peek at an agent's recent conversation (N messages, or -1 for all).",
       meta_query,
       arg("agent_id", str, "Query a specific agent for detailed info (omit to list all)", required=False),
       arg("messages", int, "Include last N messages from agent's history (0=none, -1=all)", default=0)),

    _m("wait",
       "Block execution until agents finish. Returns {completed: [{id, status, result, artifacts}], "
       "pending: [{id, status}]}. mode='all' blocks until every target finishes (child-completion "
       "notices do NOT cut it short); mode='any' returns as soon as one finishes. Only an 'immediate' "
       "message or the timeout ends it early. Use mode='any' in a loop to process results as they "
       "arrive incrementally.",
       meta_wait,
       arg("agent_ids", list, "Which agents to wait for (default: all your children)", items=str, required=False),
       arg("timeout", float, "Max seconds to wait before returning", default=300),
       arg("mode", str, "'all' = block until every agent finishes, 'any' = return as soon as one finishes",
           enum=["all", "any"], default="all")),

    _m("get_cost",
       "Get your identity and resource status. Returns: your agent_id, parent, children list, "
       "context_tokens/limit, turns used/max, elapsed time, and remaining budget.",
       meta_get_cost),

    _m("set_status",
       "Change your lifecycle status. 'done': terminate and notify parent with your result (parent "
       "receives '[Agent X finished: done] result'). 'idle': pause your loop and sleep — you wake "
       "automatically when any message arrives in your inbox.",
       meta_set_status,
       arg("status", str, "'done' = terminate, 'idle' = sleep until messaged", enum=["done", "idle"]),
       arg("result", str, "Summary of your output (sent to parent on 'done')", required=False)),

    _m("submit",
       "Mark a file as a final deliverable/artifact. The file is copied to the shared/ directory so "
       "all agents and the user can access it. Use for outputs that represent completed work.",
       meta_submit,
       arg("path", str, "Path to the file to submit (relative to your workspace)"),
       arg("description", str, "Brief description of what this deliverable is", required=False)),
)
