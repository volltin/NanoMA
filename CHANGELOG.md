# Changelog

All notable changes to NanoMA are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/).

## v0.10.0

### Added — Model Aliases

- **`aliases:` section in `models.yaml`** — short, stable names that resolve to a
  concrete model, a fusion model, or another alias (e.g. `mini → deepseek/deepseek-v4-flash:nitro`).
- Generic, provider-agnostic capability tiers ship by default: `nano` / `mini` / `air` / `pro` / `max`.
- Aliases work **anywhere a model string is accepted**: `--model`, `RuntimeConfig.default_model`,
  and `spawn(model=...)`. Resolution is centralized in `ModelRegistry.resolve()` (with cycle protection).
- Per-model inline aliases are also supported (`aliases: [short, other]` under a model).
- `ModelRegistry.get()` / `pricing()` / `context_limit()` / `route()` are now alias-aware.
- Fixes a long-standing mismatch where the default model `deepseek-v4-flash` did not match the
  registry key `deepseek/deepseek-v4-flash:nitro`.

### Added — Model Fusion (mix of models, built on the agent framework)

- **`fusion:` section in `models.yaml`** — define a *fusion model*: a **panel** of models each
  solves the task, then a **judge** model synthesizes their answers (consensus / contradictions /
  unique insights / blind spots).
- **Fusion is the multi-agent framework, not a second mechanism.** A fusion model is sugar for the
  [Mixture-of-Agents](presets/13_mixture_of_agents.md) pattern: using one creates an agent on the
  judge model with a Mixture-of-Agents directive injected, which **spawns the panel as real
  sub-agents** (via the ordinary `spawn`/`wait`/`query`/`submit` tools) and synthesizes their
  answers. The engine special-cases nothing at inference time.
- **No new tool, no new call path.** Works anywhere a model string is accepted — `--model fusion`,
  `default_model="fusion-quality"`, `spawn(model="fusion-budget")`.
- **Two faces of the same procedure**: the **slug** (auto-injects the directive) and the editable
  **preset** `presets/13_mixture_of_agents.md` (upgraded to assign a different model per proposer).
- **Fully observable** — each panel model is a real agent in the trace (`logs/events.jsonl`) and
  communication map,
  with its own cost line; budgets stay accurate via the existing per-agent accounting.
- **Bounded recursion** — panel/judge names are resolved to concrete models, so a fusion agent never
  spawns more fusion agents (also capped by `max_depth`).
- Ships three presets: `fusion-quality` (frontier panel), `fusion-budget` (cheap diverse panel),
  `fusion-self` (same model ×2 + synthesis).
- The agent system prompt lists available aliases and fusion models so agents can pick a model when
  spawning sub-agents.

### Changed

- **Claude prompt caching via a native Anthropic request path (`anthropic_native`).** OpenAI-compatible
  proxies strip Anthropic `cache_control`, so Claude models get no prompt caching through
  `/chat/completions`. When enabled, NanoMA routes Claude-family models (`claude`/`opus`/`sonnet`/`haiku`)
  through the native `/messages` endpoint — converting the request to Anthropic format (tool calls →
  `tool_use`, tool results coalesced into one user turn) and placing cache breakpoints on the stable
  prefix (system + tools) and the latest turn. Enable via `RuntimeConfig(anthropic_native=True)`,
  `nanoma --anthropic-native`, or front-matter `anthropic_native: true`.
- **`stats()` reports `cache_hit_rate`** and the cost ledger tracks input/cached/output token totals,
  making prompt-cache effectiveness visible without parsing per-call logs.
- **`Runtime.run(task, workspace=...)` / `create_agent(workspace=...)`.** The root agent can operate in
  an existing directory (shell CWD, file tools, and the system-prompt path all agree) instead of a
  private sub-dir — useful for running against an external task environment.
- **Finishing protocol in the system prompt.** Every agent (root included) is told that prose is not
  delivered and that the only way to finish is `set_status(status="done", result=...)`. Previously this
  reached sub-agents only, so a verbose root agent could waste no-op turns after finishing the work.
- **No-op turn nudge (safety net).** A turn that returns prose with no tool call gets a one-line nudge
  to act or call `set_status`.
- **Trace key consistency.** `logs/events.jsonl` records use the `event` key (was `type`), matching the
  `on_event` callback and `Runtime._events`, so one parser works for both the live stream and the file.
- **Fixed `stats().agents.peak_concurrent`.** It was computed from `spawn` events (keyed by the parent),
  so parallel sub-agents were undercounted; now computed from each agent's own `agent_new` →
  `done`/`failed` interval.
- **Removed the HTML trace viewer** (`nanoma/viewer.py` + `viewer.html`). Runs still write a structured
  trace to `logs/events.jsonl` plus per-call LLM logs; inspect or parse it directly.
- **Pruned the tool set 22 → 15 (Orthogonal Minimalism).** Removed tools whose behavior is already
  covered by shell, the shared filesystem, native multi-tool-calls, or automatic context compression:
  - `transfer` → write/read via the shared dir (`$SHARED`).
  - `set_bio` (+ the `Agent.bio` field) → `query()` shows id/status/task; advertise via `$SHARED`.
  - `batch` → the function-calling API already runs multiple tool calls per turn.
  - `apply_patch` → `create_file` + `multi_replace` + `shell rm`.
  - `rebirth` → automatic context compression already keeps agents under their window.
  - `code_outline`, `read_symbol` → `grep` + `read_file`.

  Kept (15): shell · create_file · append_file · read_file · replace_string · multi_replace · grep ·
  spawn · send · wait · query · kill · set_status · get_cost · submit. `set_status` stays — it's the
  lifecycle primitive (terminate/idle), which a file cannot replicate.
- `nanoma --model` default is now the `mini` tier alias; added `--models PATH` to load a custom registry.
- Changelog moved out of `README.md` into this file.
- **Refactored the 950-line `core.py` monolith into focused modules** — `state.py` (Agent /
  Envelope / ResourceQuota / Artifact / ToolContext / IdGenerator), `config.py` (RuntimeConfig),
  `prompts.py` (system-prompt / model-menu / fusion-directive construction), and `runtime.py`
  (the `Runtime` class). `core.py` is now a thin facade re-exporting the public types, so every
  `from nanoma.core import …` keeps working. Behavior and semantics are unchanged.
- **All tool code now lives in a dedicated `nanoma/tools/` package** (was scattered as
  `nanoma/{tools,file_ops,file_edit,code_search,meta}.py`):
  `tools/{spec,shell,file_ops,file_edit,grep,meta,registry}.py` (the search module is now just
  `grep.py`, since `code_outline`/`read_symbol` were dropped).
- **Tool schemas are generated from structured declarations, not hand-written JSON.** New
  `nanoma/tools/spec.py` provides `Tool` + `arg()` + `obj()`; each tool declares typed argument
  specs and `Tool.schema` produces the OpenAI function schema. No more maintaining nested
  `{"type": "object", "properties": {...}}` dicts by hand. `Tool` stays dict-compatible
  (`tool["handler"]`, `tool["schema"]`, `tool.get("is_meta")`) so runtime dispatch is unchanged.
- **Dropped the `ws_` prefix from workspace tool names** (`ws_create_file` → `create_file`, etc.);
  shell and meta names unchanged.
- **Workspace tools are now first-class core tools, not a plugin.** The old
  `nanoma/plugins/workspace_tools/` package is removed; `nanoma.tools.build_all_tools()` is the single
  source of truth for the full agent tool set (work + meta).

### Added — `nanoma-run` Markdown task runner

- Run a task by writing **one Markdown file** with optional YAML front-matter for config
  (`model`, `models`, `budget`, `max_agents`, `time_limit`, `base_url`, …); the body is the prompt.
- Self-contained per-task folder: `nanoma-run <folder>` uses `<folder>/task.md` and writes
  `result.md`, `logs/`, and `workspace/` alongside it. Model config can come from the global default
  or be written at the top of the file.

### Added — preflight model calibration

- `nanoma-run` now probes every model a run will use (the default model, or a fusion's full
  panel + judge) **before** starting, using the real request shape (system+user messages, the real
  tool schemas, configured `temperature`). Unusable models (bad name, auth, message-shape, tool-schema
  rejection, deprecated params) **abort the run before any budget is spent**.
- Flags: `--calibrate-only` (probe and exit), `--no-calibrate` (skip); front-matter `calibrate: false`.
- New module `nanoma/calibrate.py` (`calibrate`, `probe_model`, `models_for_run`) — importable.

### Improved — precise, paginated output for large tool results

- Large tool output is never dumped into context unbounded. Output tools return a **bounded
  preview**, save the **full** text to a file, and report **precisely** what was shown: total
  chars/lines, the exact shown range, and the next offset to continue from.
- `shell`: a long stream returns a clean preview plus `<stream>_truncated`, `_total_chars`,
  `_total_lines`, `_shown_chars`, `_shown_lines`, `_file`, and a precise `_note`
  (e.g. *"showed chars 0–300 of 1892 (lines 1–102 of 500) … read more with read_file(offset=103)"*).
  Previously it appended an inline `"...(truncated N chars)"` marker into the data.
- `read_file`: now also returns `total_chars` and `has_more`, applies a char budget
  (`file_read_max_chars`, default 50 000) on top of the line limit, and emits a precise `note` with
  the next offset; a single over-long line is flagged so paging can't loop.
- New shared helper `nanoma/tools/output.py` (`clip_text`, `truncation_note`) encodes the rule once,
  so future output tools (e.g. a web `fetch`) behave the same: the agent always learns exactly how
  big a result is and can read the whole thing or just the slice it needs.

### Improved — multi-agent ergonomics

- **`wait(mode="all")` now actually waits for all.** Child-completion notices are delivered as
  `steer` messages, and `wait` used to treat *any* inbox message as an interrupt — so a parent
  waiting on 3 children returned after the first and had to re-issue `wait` per child (and later
  calls lost track of earlier completions). Now only an **`immediate`** message (or the timeout)
  ends a wait early; `steer`/`queue` messages stay queued and are injected afterward. Default `wait`
  timeout raised 120 → 300s.
- **No more redundant "report to parent."** The sub-agent system prompt told agents to `send` their
  result to the spawner *and then* `set_status("done")` — but `done` already delivers the result to
  the parent, so every sub-agent wasted a turn. Guidance is now just `set_status("done", result=...)`;
  `send()` is for mid-task coordination only.
- **`create_file` overwrite hint** is now structured (`hint` + `existing_bytes`) — agents that
  scaffold with `cargo new`/`npm init` and then write the stub file recover in one step.

### Fixed — provider compatibility (Anthropic / reasoning models)

- Agents now start with a **kickoff user turn** in addition to the system prompt. A system-only
  request is rejected by Anthropic-backed providers (`field messages is required`, HTTP 500).
- `temperature` is **omitted unless explicitly set** (`RuntimeConfig.temperature`, default `None`).
  Several models reject it (e.g. Claude Opus via Bedrock: *"temperature is deprecated for this
  model"*, HTTP 400; OpenAI o-series). Previously NanoMA always sent `temperature: 0.7`.

## v0.9.2

- **Redesign: Orthogonal Minimalism** — reduced from 37 tools to 22 by eliminating shell-redundant tools
- **Core principle**: a tool exists only when shell cannot reliably replicate its interaction model
- `grep` now uses subprocess (ripgrep → grep → Python fallback) for 10-100x performance
- `find_block_end` correctly skips braces inside string literals and comments
- Added Rust and Go symbol patterns to code outline
- Fixed `batch` meta tool to include workspace tools (was missing `WORKSPACE_TOOLS`)
- Shell output files use UUID naming (no more hash collisions)
- `multi_replace` error messages explicitly state file is unchanged on failure
- Removed: `ws_file_search`, `ws_list_dir`, `ws_project_structure`, `ws_create_directory`, `ws_delete_file`, `ws_rename_file`, `ws_search_symbols`, `ws_list_usages`, `ws_tool_search`, `ws_call_from_file`, skills system (5 tools)
- Removed: `file_read`, `file_write`, `file_list`, `grep` from basic WORK_TOOLS (superseded by ws_* or shell)

## v0.9.1

- Initial release: core runtime, meta tools, 30 presets, trace viewer
