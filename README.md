# NanoMA

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue?logo=python&logoColor=white)](pyproject.toml)
![Version](https://img.shields.io/badge/version-0.9.2-7c3aed)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A minimal multi-agent harness for research.** ~2000 lines of Python.

NanoMA provides the thinnest possible runtime for studying multi-agent LLM coordination.
It is deliberately unopinionated — the framework supplies primitives (spawn, send, wait, kill, shared filesystem),
and the orchestration pattern emerges entirely from the prompt you give the root agent.

## Design Principle: Orthogonal Minimalism

> **Shell is the universal escape hatch. A dedicated tool exists only when it provides
> an interaction model that shell cannot reliably replicate.**

Every tool earns its place by satisfying one of three criteria:
1. **Reliability** — shell can't do it without escaping gymnastics
2. **Atomicity** — the operation needs multi-step transactional semantics
3. **Coordination** — the operation requires access to runtime internals

If `shell("one-liner")` produces equally usable output → the tool is redundant → it doesn't exist.

See [DESIGN.md](DESIGN.md) for the full rationale.

## Quick Start

```bash
pip install -e .
export NANOMA_API_KEY="sk-..."
export NANOMA_LLM_BASE_URL="https://openrouter.ai/api/v1"

# Run with a preset orchestration pattern:
nanoma "$(cat presets/01_orchestrator_workers.md | sed 's/{task}/Build a REST API/')" --budget 5.0

# Or run raw — let the agent decide its own strategy:
nanoma "Build a linked list in C with tests" --budget 5.0 --max-agents 20
```

## Architecture: 3 Layers, 22 Tools

```
Layer 3: META (12 tools) — coordination primitives on runtime internals
Layer 2: WORKSPACE (9 tools) — structured ops where shell fails
Layer 1: SHELL (1 tool) — universal primitive for everything else
```

### Meta Tools (Coordination)

| Tool | What it does |
|------|-------------|
| `spawn(task)` | Create a new agent (runs in parallel immediately) |
| `send(to, message)` | Send a message to any agent |
| `wait(ids, mode)` | Block until agents finish (`mode="all"` or `"any"`) |
| `query()` | Discover all agents and their status |
| `kill(id)` | Terminate an agent |
| `transfer(src, to)` | Copy files between workspaces |
| `set_status("done")` | Finish and report result to parent |
| `set_status("idle")` | Sleep until messaged |
| `set_bio(bio)` | Advertise your role to others |
| `rebirth(summary)` | Reset context to save memory |
| `submit(path)` | Mark a file as deliverable |
| `batch(path)` | Execute tool calls from a JSON file |

### Workspace Tools (Structured I/O)

| Tool | Why shell can't | What it does |
|------|----------------|-------------|
| `ws_create_file` | Heredoc escaping | Create files with any content reliably |
| `ws_append_file` | Same | Append to files without escaping issues |
| `ws_read_file` | No structured pagination | Read with line numbers, offset, limit |
| `ws_replace_string` | sed can't 4-tier match | Find-replace with cascading match strategies |
| `ws_multi_replace` | No atomicity | Batch replace, all-or-nothing |
| `ws_apply_patch` | No V4A format | Context-based multi-file patching |
| `ws_grep` | No structured output | Search with file/line/content JSON results |
| `ws_code_outline` | No CLI equivalent | Symbol tree with block boundaries |
| `ws_read_symbol` | Needs outline knowledge | Extract a specific function/class by name |

### Shell (Universal Primitive)

| Tool | What it does |
|------|-------------|
| `shell(command)` | Execute any command (stdout, stderr, exit_code) |

Agents use shell for everything else: `mkdir -p`, `rm -rf`, `mv`, `ls`, `find`, `tree`, `git`, `pip`, `curl`, etc.

## Design Principles

- **Uniform infrastructure** — every agent has the same loop, same tools. Differentiation emerges from task prompts, not from code.
- **Flat topology** — any agent can message any other. The framework imposes no hierarchy — but agents can self-organize into any pattern.
- **Stigmergy** — agents coordinate through a shared/ filesystem and direct messages.
- **Global budget** — one shared pool. When it runs out, everyone stops.
- **Fresh context** — each spawned agent gets a clean context window.
- **Orthogonal minimalism** — 22 tools total. If shell can do it, there's no dedicated tool.

## Security Notice

**NanoMA has NO sandboxing for shell commands.** Agents execute shell commands with the same privileges as the host process. File operations in workspace tools are sandboxed to the workspace root via path validation, but shell bypasses this.

**Always run NanoMA in a disposable environment** (container, VM, or dedicated machine).

## Presets (30 Orchestration Patterns)

The `presets/` directory contains 30 prompt templates implementing known MA patterns.
Each is a pure prompt — no code changes needed:

| # | Pattern | Topology |
|---|---------|----------|
| 01 | Orchestrator-Workers | Star |
| 02 | Evaluator-Optimizer | Loop |
| 03 | Prompt Chaining | Chain |
| 04 | Router | Fan-out |
| 05 | Parallelization | Fan-out/in |
| 06 | Debate | Star+Judge |
| 07 | Plan-and-Execute | Planner↔Executor |
| 08 | Map-Reduce | N→1 |
| 09 | Hierarchical Delegation | Tree |
| 10 | Swarm | Mesh |
| 11–30 | ... | See `presets/` |

Usage: `nanoma "$(cat presets/06_debate.md | sed 's/{task}/your question/')" --budget 2.0`

## Programmatic API

```python
import asyncio
from pathlib import Path
from nanoma import Runtime, RuntimeConfig

async def main():
    config = RuntimeConfig(
        budget=5.0,
        max_agents=20,
        max_turns=50,
        default_model="deepseek/deepseek-v4-flash",
        workspace_root=Path("./workspace"),
        log_dir=Path("./logs"),
    )
    rt = Runtime(config=config)
    result = await rt.run("Your task here")
    print(result)
    print(rt.stats())

asyncio.run(main())
```

## Trace Viewer

Every run produces a trace in `logs/events.jsonl`. View it in real-time:

```bash
python nanoma/viewer.py ./logs 8900
# Open http://localhost:8900
```

## Configuration

```python
RuntimeConfig(
    budget=10.0,              # global budget in USD
    max_agents=1000,          # max total agents
    max_depth=100,            # max spawn depth
    max_concurrent_llm=50,    # parallel LLM calls
    time_limit=0,             # seconds, 0 = unlimited
    max_turns=200,            # per agent
    default_model="deepseek-v4-flash",
    # Truncation (0 = unlimited for any of these)
    shell_max_output=10000,
    file_read_max_chars=50000,
    file_list_max_entries=500,
    grep_max_results=100,
    # Context compression
    compress_keep_recent=6,
    compress_max_messages=40,
    compress_max_chars=300,    # 0 = keep full content
)
```

## Changelog

### v0.9.2

- **Redesign: Orthogonal Minimalism** — reduced from 37 tools to 22 by eliminating shell-redundant tools
- **Core principle**: a tool exists only when shell cannot reliably replicate its interaction model
- `ws_grep` now uses subprocess (ripgrep → grep → Python fallback) for 10-100x performance
- `find_block_end` correctly skips braces inside string literals and comments
- Added Rust and Go symbol patterns to code outline
- Fixed `batch` meta tool to include workspace tools (was missing `WORKSPACE_TOOLS`)
- Shell output files use UUID naming (no more hash collisions)
- `ws_multi_replace` error messages explicitly state file is unchanged on failure
- Removed: `ws_file_search`, `ws_list_dir`, `ws_project_structure`, `ws_create_directory`, `ws_delete_file`, `ws_rename_file`, `ws_search_symbols`, `ws_list_usages`, `ws_tool_search`, `ws_call_from_file`, skills system (5 tools)
- Removed: `file_read`, `file_write`, `file_list`, `grep` from basic WORK_TOOLS (superseded by ws_* or shell)

### v0.9.1

- Initial release: core runtime, meta tools, 30 presets, trace viewer

## Citation

If you use NanoMA in your research, please cite:

```bibtex
@software{he2026nanoma,
  author = {Jiyan He},
  title = {NanoMA: A Minimal Multi-Agent Harness for Research},
  year = {2026},
  url = {https://github.com/volltin/NanoMA}
}
```
