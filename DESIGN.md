# NanoMA Design Principles

## The One Rule: Orthogonal Minimalism

> **Shell is the universal escape hatch. A dedicated tool exists only when it provides
> an interaction model that shell cannot reliably replicate.**

### Admission Criteria (satisfy ANY ONE to keep a tool)

1. **Reliability** — Shell cannot do this reliably due to escaping, quoting, or state issues.
   (e.g., `create_file` — file content with backticks/dollars breaks heredocs)

2. **Atomicity** — The operation requires multi-step transactional semantics.
   (e.g., `replace_string` — 4-tier matching cascade is impossible in sed)

3. **Coordination** — The operation requires access to runtime internals (other agents, budgets, queues).
   (e.g., `spawn`, `send`, `wait` — these operate on the runtime graph, not the filesystem)

### Elimination Criterion

If `shell("one-liner")` produces equally usable output → the tool is redundant → remove it.

### Cost Formula

```
Each tool ≈ 100 tokens of schema × N agents = N×100 tokens/turn
Fewer tools → less deliberation overhead → faster, cheaper, more reliable decisions
```

---

## Tool Architecture (3 Layers)

```
Layer 3: META (8 tools)
  Coordination primitives operating on runtime internals.
  spawn, kill, send, query, wait, get_cost, set_status, submit

Layer 2: WORKSPACE (6 tools)
  Structured operations where shell fails at reliability/atomicity.
  read_file, create_file, append_file, replace_string, multi_replace, grep

Layer 1: SHELL (1 tool)
  Universal primitive. Anything not covered above.
  shell(command, timeout)
```

### Implementation: one package, generated schemas

All tool code lives in the **`nanoma/tools/`** package:

```
nanoma/tools/
  spec.py        Tool · arg() · obj() — structured schema generation
  shell.py       Layer 1
  file_ops.py / file_edit.py / grep.py          Layer 2 (workspace tools)
  meta.py        Layer 3 (+ META_TOOLS)
  registry.py    WORK_TOOLS (shell + workspace) · build_all_tools()
```

Tools are **declared**, not hand-serialized. Instead of maintaining nested OpenAI
function-schema JSON by hand, each tool is a `Tool` with typed `arg(...)` specs and the
schema is generated:

```python
Tool("create_file", "Create a new file…", tool_create_file,
     arg("path", str, "File path"),
     arg("content", str, "File content"),
     arg("overwrite", bool, "Overwrite existing", default=False))
```

`arg()` maps Python types → JSON types, `default=` marks a param optional, `enum=` constrains
values, and `obj(...)`/`items=` build nested array/object schemas. `Tool` is dict-compatible
(`tool["handler"]`, `tool["schema"]`, `tool.get("is_meta")`) so the runtime's dispatch is
untouched. `build_all_tools()` (= `WORK_TOOLS` + `META_TOOLS`) is the single source of truth
for the registry every agent sees.

### Why these 6 workspace tools?

| Tool | Why shell can't | Layer justification |
|------|----------------|---------------------|
| `read_file` | Paginated line-numbered reading with structural metadata | Structured I/O |
| `create_file` | Content with `` ` ``, `$`, `\`, `EOF` breaks heredocs | Reliability |
| `append_file` | Same escaping issues as create | Reliability |
| `replace_string` | 4-tier cascading match (exact→trimmed→indent→normalized) | Atomicity |
| `multi_replace` | All-or-nothing batch replacement | Atomicity |
| `grep` | Structured JSON results with match positions, auto-ignore dirs | Structured I/O |

Everything else (V4A patching, code outlines, cross-workspace copies, context reset, self-bios,
batched calls) was dropped: shell + the shared filesystem + native multi-tool-calls + automatic
context compression already cover them.

### What agents use shell for (explicitly)

These were previously dedicated tools, now delegated to shell:

```bash
mkdir -p path              # was: ws_create_directory
rm -rf path               # was: ws_delete_file
mv old new                # was: ws_rename_file
ls -la path               # was: ws_list_dir
find . -name '*.py'       # was: ws_file_search
tree -L 3                 # was: ws_project_structure
```

---

## Multi-Agent Coordination Design

The 8 meta tools implement **uniform infrastructure** — every agent has identical capabilities.
The orchestration pattern emerges entirely from the task prompt.

### Composability Examples

| Pattern | Composition |
|---------|-------------|
| Orchestrator-Workers | `spawn` × N → `wait(mode="all")` → aggregate |
| Map-Reduce | `spawn` × N → `wait(mode="all")` → read `shared/` → reduce |
| Streaming Pipeline | `spawn` → `send(mode="steer")` chain |
| Debate/Tournament | `spawn` × N → `query(messages=N)` → judge → `kill` losers |
| Event-Driven Service | `set_status("idle")` → `send` wakes → process → `set_status("idle")` |
| Hierarchical Delegation | `spawn(delegate=True)` cascading |
| Iterative Refinement | `spawn` → `wait` → evaluate → `send` feedback → repeat |

### Key Insight

8 primitives × free composition = unbounded topologies.
No framework code needed — the prompt IS the orchestration logic.

---

## Model Layer: Aliases & Fusion

The model layer follows the same minimalism rule as tools: **a model is just a string**, and
anything that improves model selection is expressed *as a model string* rather than as new tools
or new agent-facing API surface.

```
model string ──► registry.resolve() ──► concrete model ──► single LLM call
                                     └─► fusion model   ──► becomes a Mixture-of-Agents
                                                            orchestrator (spawns real agents)
```

### Aliases — naming, not plumbing

Aliases (`mini → deepseek/deepseek-v4-flash:nitro`, provider-agnostic tiers `nano`/`mini`/`air`/`pro`/`max`)
are pure indirection in `models.yaml`.
They resolve uniformly in `get / pricing / context_limit / route`, and once at call time in
`Runtime._model_call`, so every consumer (`--model`, `default_model`, `spawn(model=...)`) gets the
benefit for free. Cycles are detected and broken. This also removed a real bug: the default model
name no longer has to match the exact provider key.

### Fusion — a model that *is* the multi-agent framework

Fusion (a panel of different models + a synthesizing judge) is conceptually identical
to the **Mixture-of-Agents** preset that already ships (`presets/13_mixture_of_agents.md`). So the
guiding decision was: **do not build a second mechanism for running many models — reuse the one
NanoMA already has (agents).**

A first instinct (and an earlier draft) was to expand a fusion model *below the ReAct loop* into raw
parallel panel + judge LLM calls. That was rejected: it is a parallel, hidden way to run many models
that bypasses the agent graph — exactly the kind of "second mode" this project avoids. It also throws
away everything the framework already gives multi-model runs for free.

Instead, a fusion model is **sugar for an orchestration**:

| Concern | Design choice | Why |
|---------|---------------|-----|
| What a fusion model is | An agent on the **judge** model, with a Mixture-of-Agents directive injected at `create_agent` | No engine call-path branch; it's a normal agent |
| The panel | **Real spawned sub-agents**, one per panel model | Observable in the trace/comm-map; own cost line; can use tools, be multi-turn |
| The judge | The fusion agent itself synthesizes after `wait()` | Honors the configured judge model |
| Mechanism | Ordinary `spawn` / `wait` / `query` / `submit` | Zero new primitives; "the prompt is the orchestration" |
| Recursion | Panel/judge resolved to **concrete** models (+ `max_depth`) | A fusion agent never spawns more fusion agents |

This yields two faces of the *same* procedure:

- **Slug (auto):** `--model fusion` / `spawn(model="fusion-quality")` — the directive is injected
  automatically. Smooth "swap the string."
- **Preset (manual):** `presets/13_mixture_of_agents.md` — the same steps as an editable prompt.

Nothing in the engine special-cases an inference for fusion; turn it on by writing `--model fusion`,
and the rest of the system treats the result as ordinary agents.

---

## Code Layout

The runtime is split into small, single-responsibility modules; `core.py` is a thin facade
re-exporting the public types so `from nanoma.core import …` stays stable.

| Module | Responsibility |
|--------|----------------|
| `state.py` | Data: `Agent`, `Envelope`, `ResourceQuota`, `Artifact`, `ToolContext`, `IdGenerator` |
| `config.py` | `RuntimeConfig` |
| `prompts.py` | Pure prompt construction: system prompt, model menu, fusion directive |
| `runtime.py` | `Runtime` — agent lifecycle, the ReAct loop, messaging, compression, stats |
| `core.py` | Public facade (re-exports) |
| `llm.py` | OpenAI-compatible client, token counting, retries, `default_router` |
| `models.py` | `ModelRegistry` (aliases & fusion, `models.yaml`) |
| `cost.py` · `scheduler.py` · `sandbox.py` | Budget ledger · concurrency limiter · shell exec |
| `tools/` | `spec` (Tool/arg schema generation), `shell`, `file_ops`, `file_edit`, `grep`, `meta`, `registry` |
| `main.py` · `task.py` · `calibrate.py` | `nanoma` CLI · `nanoma-run` · model preflight |
