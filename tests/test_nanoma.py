"""Tests for NanoMA framework — no real LLM calls, fully deterministic."""

import pytest

from nanoma.core import Envelope, ResourceQuota, Runtime, RuntimeConfig, ToolContext
from nanoma.cost import CostLedger, UsageRecord
from nanoma.llm import LLMResponse, ToolCall, estimate_tokens, count_message_tokens
from nanoma.tools.meta import (
    meta_spawn, meta_kill, meta_send, meta_query, meta_wait,
    meta_get_cost, meta_set_status, meta_submit,
)
from nanoma.models import load_models
from nanoma.scheduler import Scheduler


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_workspace(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "shared").mkdir()
    return ws


@pytest.fixture
def runtime(tmp_workspace):
    """Runtime with a mock LLM that immediately calls set_status(done)."""
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="tc1", name="set_status", arguments={"status": "done", "result": "test done"})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    config = RuntimeConfig(
        workspace_root=tmp_workspace,
        budget=10.0,
        max_agents=50,
        log_dir=None,
    )
    return Runtime(config=config, llm_call=mock_llm)


@pytest.fixture
def runtime_multi_turn(tmp_workspace):
    """Runtime where LLM does 3 turns then quits."""
    call_count = {"n": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            return LLMResponse(
                tool_calls=[ToolCall(id=f"tc{call_count['n']}", name="set_status",
                                     arguments={"status": "done", "result": f"done after {call_count['n']} turns"})],
                usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
            )
        return LLMResponse(
            tool_calls=[ToolCall(id=f"tc{call_count['n']}", name="get_cost", arguments={})],
            usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None)
    return Runtime(config=config, llm_call=mock_llm)


# ─── Test: Basic agent lifecycle ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_basic_run(runtime):
    """Agent starts, calls set_status(done), terminates."""
    result = await runtime.run("Say hello")
    assert result == "test done"
    assert len(runtime.agents) == 1
    agent = list(runtime.agents.values())[0]
    assert agent.status == "done"
    assert agent._turns >= 1


@pytest.mark.asyncio
async def test_multi_turn(runtime_multi_turn):
    """Agent runs multiple turns before completing."""
    result = await runtime_multi_turn.run("Do something complex")
    assert "done after 3 turns" in result


# ─── Test: ID generation ─────────────────────────────────────────────────────

def test_id_generation():
    from nanoma.core import IdGenerator
    gen = IdGenerator()
    ids = [gen.next() for _ in range(30)]
    assert ids[0] == "alpha"
    assert ids[25] == "zulu"
    assert ids[26] == "alpha-1"
    assert len(set(ids)) == 30  # all unique


# ─── Test: ResourceQuota ─────────────────────────────────────────────────────

def test_quota_defaults():
    q = ResourceQuota(budget=10.0, time_limit=60.0, max_turns=100)
    assert q.budget == 10.0
    assert q.time_limit == 60.0
    assert q.max_turns == 100


# ─── Test: CostLedger ────────────────────────────────────────────────────────

def test_ledger():
    ledger = CostLedger(total_budget=5.0)
    assert ledger.remaining() == 5.0
    assert ledger.can_afford(3.0)
    usage = UsageRecord(input_tokens=1000, output_tokens=500, model="test")
    ledger.record("agent-1", usage)
    assert ledger.total_spent > 0
    assert "agent-1" in ledger.per_agent


# ─── Test: Scheduler ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduler():
    s = Scheduler(max_concurrent=2)
    assert s.stats["available"] == 2
    await s.acquire()
    assert s.stats["active"] == 1
    assert s.stats["available"] == 1
    s.release()
    assert s.stats["active"] == 0


# ─── Test: Message delivery ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_message_delivery(runtime):
    agent = runtime.create_agent("test task")
    env = Envelope(from_id="other", to_id=agent.id, content="hello", tokens=5, timestamp=0.0, mode="queue")
    await runtime.deliver(env)
    assert not agent._queue_inbox.empty()


@pytest.mark.asyncio
async def test_idle_wake(runtime):
    """Idle agent wakes on message."""
    agent = runtime.create_agent("test")
    agent.status = "idle"
    env = Envelope(from_id="x", to_id=agent.id, content="wake up", tokens=5, timestamp=0.0, mode="queue")
    await runtime.deliver(env)
    assert agent.status == "running"


# ─── Test: Meta tools ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_meta_spawn(runtime):
    parent = runtime.create_agent("parent task")
    result = await meta_spawn({"task": "child task"}, parent, runtime)
    assert "agent_id" in result
    assert result["agent_id"] in runtime.agents
    # v0.9.0: no per-agent budget deduction on spawn (global budget model)
    child = runtime.agents[result["agent_id"]]
    assert child.parent == parent.id
    assert child.id in parent.children


@pytest.mark.asyncio
async def test_meta_spawn_max_depth(runtime):
    """Spawn fails when max depth is exceeded."""
    parent = runtime.create_agent("parent")
    parent.depth = runtime.config.max_depth  # already at max
    result = await meta_spawn({"task": "child"}, parent, runtime)
    assert "error" in result


@pytest.mark.asyncio
async def test_meta_kill(runtime):
    parent = runtime.create_agent("parent")
    child = runtime.create_agent("child", parent=parent.id)
    parent.children.add(child.id)
    result = await meta_kill({"agent_id": child.id}, parent, runtime)
    assert result["killed"] == child.id
    assert child.status == "done"


@pytest.mark.asyncio
async def test_meta_kill_permission(runtime):
    a = runtime.create_agent("a")
    b = runtime.create_agent("b")
    result = await meta_kill({"agent_id": b.id}, a, runtime)
    assert "error" in result  # can't kill non-descendant


@pytest.mark.asyncio
async def test_meta_send(runtime):
    a = runtime.create_agent("sender")
    b = runtime.create_agent("receiver")
    result = await meta_send({"to": b.id, "message": "hello", "mode": "queue"}, a, runtime)
    assert result["delivered"] == 1
    assert not b._queue_inbox.empty()


@pytest.mark.asyncio
async def test_meta_send_no_broadcast(runtime):
    """No broadcast support — must specify IDs."""
    a = runtime.create_agent("sender")
    # '*' is treated as a literal agent_id which won't exist
    result = await meta_send({"to": "*", "message": "hi"}, a, runtime)
    assert result["delivered"] == 0


@pytest.mark.asyncio
async def test_meta_query_all(runtime):
    runtime.create_agent("task A")
    runtime.create_agent("task B")
    a = runtime.create_agent("querier")
    result = await meta_query({}, a, runtime)
    assert result["count"] == 3
    assert all("task" in x for x in result["agents"])


@pytest.mark.asyncio
async def test_meta_query_single(runtime):
    a = runtime.create_agent("target")
    b = runtime.create_agent("querier")
    result = await meta_query({"agent_id": a.id}, b, runtime)
    assert result["task"] == "target"
    assert "messages" not in result  # messages=0 by default


@pytest.mark.asyncio
async def test_meta_query_with_messages(runtime):
    a = runtime.create_agent("target")
    a.history.append({"role": "user", "content": "msg1"})
    a.history.append({"role": "assistant", "content": "reply1"})
    a.history.append({"role": "user", "content": "msg2"})
    a.history.append({"role": "assistant", "content": "reply2"})
    b = runtime.create_agent("querier")
    # Last 2 messages
    result = await meta_query({"agent_id": a.id, "messages": 2}, b, runtime)
    assert "messages" in result
    assert len(result["messages"]) == 2


@pytest.mark.asyncio
async def test_meta_query_all_messages(runtime):
    a = runtime.create_agent("target")
    a.history.append({"role": "user", "content": "msg1"})
    a.history.append({"role": "assistant", "content": "reply1"})
    b = runtime.create_agent("querier")
    result = await meta_query({"agent_id": a.id, "messages": -1}, b, runtime)
    assert "messages" in result
    assert len(result["messages"]) >= 2


@pytest.mark.asyncio
async def test_meta_get_cost(runtime):
    a = runtime.create_agent("worker")
    a._turns = 5
    a.tokens_consumed = 1000
    result = await meta_get_cost({}, a, runtime)
    assert result["turns_used"] == 5
    assert result["tokens_consumed"] == 1000
    assert "budget_remaining" in result
    assert "budget_total" in result
    assert result["budget_total"] == runtime.ledger.total_budget
    assert "total_agents" in result


@pytest.mark.asyncio
async def test_meta_set_status(runtime):
    a = runtime.create_agent("worker")
    await meta_set_status({"status": "done", "result": "finished!"}, a, runtime)
    assert a.status == "done"
    assert a.result == "finished!"


@pytest.mark.asyncio
async def test_meta_submit(runtime):
    a = runtime.create_agent("worker")
    # Create a file in workspace
    test_file = a.workspace / "output.txt"
    test_file.write_text("result data")
    result = await meta_submit({"path": "output.txt", "description": "final output"}, a, runtime)
    assert result["submitted"] == "output.txt"
    assert len(a.artifacts) == 1
    # Check shared copy exists
    shared = runtime._tool_context.shared_dir / "output.txt"
    assert shared.exists()


@pytest.mark.asyncio
async def test_meta_wait_immediate(runtime):
    """Wait returns immediately when children already done."""
    parent = runtime.create_agent("parent")
    child = runtime.create_agent("child", parent=parent.id)
    parent.children.add(child.id)
    child.status = "done"
    child.result = "child result"
    result = await meta_wait({}, parent, runtime)
    assert len(result["completed"]) == 1
    assert result["completed"][0]["status"] == "done"


# ─── Test: Work tools ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_file_write_read(tmp_workspace):
    from nanoma.tools import tool_create_file, tool_read_file_advanced
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()

    # Write
    result = await tool_create_file({"path": "test.txt", "content": "hello world"}, ws, ctx)
    assert result["bytes_written"] == 11

    # Read
    result = await tool_read_file_advanced({"path": "test.txt"}, ws, ctx)
    assert "hello world" in result["content"]


@pytest.mark.asyncio
async def test_tool_file_read_sandbox(tmp_workspace):
    from nanoma.tools import tool_read_file_advanced
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_read_file_advanced({"path": "/etc/passwd"}, ws, ctx)
    assert "error" in result  # outside workspace


@pytest.mark.asyncio
async def test_tool_shell(tmp_workspace):
    from nanoma.tools import tool_shell
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "echo hello"}, ws, ctx)
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_tool_shell_timeout(tmp_workspace):
    from nanoma.tools import tool_shell
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    result = await tool_shell({"command": "sleep 10", "timeout": 1}, ws, ctx)
    assert result["exit_code"] == -1
    assert "Timeout" in result["stderr"]


@pytest.mark.asyncio
async def test_tool_grep(tmp_workspace):
    from nanoma.tools import tool_grep_search
    ctx = ToolContext(shared_dir=tmp_workspace / "shared", workspace_root=tmp_workspace)
    ws = tmp_workspace / "agent"
    ws.mkdir()
    (ws / "code.py").write_text("def hello():\n    return 42\n")
    result = await tool_grep_search({"query": "hello"}, ws, ctx)
    assert result["count"] >= 1
    assert any("hello" in m["content"] for m in result["matches"])


# ─── Test: Model registry ────────────────────────────────────────────────────

def test_model_registry(tmp_path):
    config = tmp_path / "models.yaml"
    config.write_text("""
models:
  test-model:
    provider: test
    context_limit: 64000
    pricing:
      input: 1.0
      cached_input: 0.1
      output: 2.0
    tier: cheap
""")
    reg = load_models(config)
    m = reg.get("test-model")
    assert m is not None
    assert m.context_limit == 64000
    assert reg.pricing("test-model") == (1.0, 0.1, 2.0)
    assert reg.route(1.0) == "test-model"


# ─── Test: Token estimation ──────────────────────────────────────────────────

def test_estimate_tokens():
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("a" * 400) == 100


def test_count_message_tokens():
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello there"},
    ]
    tokens = count_message_tokens(msgs)
    assert tokens > 0


# ─── Test: Full integration (spawn + message + wait) ─────────────────────────

@pytest.mark.asyncio
async def test_spawn_and_wait(tmp_workspace):
    """Parent spawns child, child finishes, parent gets result."""
    turn_count = {"parent": 0, "child": 0}

    async def mock_llm(messages, model, tools=None, **kwargs):
        # Detect if this is a child (task contains "child")
        system = messages[0]["content"] if messages else ""
        if "child task" in system:
            turn_count["child"] += 1
            return LLMResponse(
                tool_calls=[ToolCall(id="c1", name="set_status", arguments={"status": "done", "result": "child done"})],
                usage=UsageRecord(input_tokens=50, output_tokens=30, model=model),
            )
        else:
            turn_count["parent"] += 1
            if turn_count["parent"] == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(id="p1", name="spawn", arguments={"task": "child task"})],
                    usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
                )
            elif turn_count["parent"] == 2:
                return LLMResponse(
                    tool_calls=[ToolCall(id="p2", name="wait", arguments={"timeout": 5})],
                    usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
                )
            else:
                return LLMResponse(
                    tool_calls=[ToolCall(id="p3", name="set_status", arguments={"status": "done", "result": "parent done"})],
                    usage=UsageRecord(input_tokens=100, output_tokens=50, model=model),
                )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None)
    rt = Runtime(config=config, llm_call=mock_llm)
    result = await rt.run("parent task")
    assert result == "parent done"
    assert len(rt.agents) == 2


@pytest.mark.asyncio
async def test_query_discovery(tmp_workspace):
    """Agents can discover each other (id, status, task) via query()."""
    async def mock_llm(messages, model, tools=None, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="t1", name="set_status", arguments={"status": "done", "result": "ok"})],
            usage=UsageRecord(input_tokens=50, output_tokens=30, model=model),
        )

    config = RuntimeConfig(workspace_root=tmp_workspace, budget=10.0, log_dir=None)
    rt = Runtime(config=config, llm_call=mock_llm)
    rt.create_agent("worker A")
    rt.create_agent("worker B")
    c = rt.create_agent("coordinator")
    result = await meta_query({}, c, rt)
    tasks = [x["task"] for x in result["agents"]]
    assert "worker A" in tasks
    assert "worker B" in tasks
