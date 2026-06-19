# NanoMA

[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue?logo=python&logoColor=white)](pyproject.toml)
![Version](https://img.shields.io/badge/version-0.10.0-7c3aed)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A minimal multi-agent harness for research.** Every agent gets the same loop and the same tools
(spawn, send, wait, kill, a shared filesystem); the orchestration pattern emerges from the prompt
you give the root agent, not from framework code.

See [DESIGN.md](DESIGN.md) for the rationale and [CHANGELOG.md](CHANGELOG.md) for history.

## Install

```bash
pip install -e .
export NANOMA_API_KEY="sk-..."
export NANOMA_LLM_BASE_URL="https://openrouter.ai/api/v1"
```

## Usage

```bash
# Run a task — the agent picks its own strategy:
nanoma "Build a linked list in C with tests" --budget 5 --max-agents 20

# Use a fusion model (panel of models + a judge):
nanoma "Survey arguments for and against carbon taxes" --model fusion --budget 5
```

Or write a task as a Markdown file with optional YAML front-matter and run it in a self-contained
folder:

```markdown
---
model: fusion     # alias / model / fusion name (omit → default)
budget: 8
---
Port the core of python-slugify to Rust and make `cargo test` pass.
```

```bash
nanoma-run my-task/   # uses my-task/task.md → writes result.md, logs/, workspace/
```

`nanoma-run` calibrates every model with a tiny real request first and aborts before spending
budget if one is unusable (`--calibrate-only` / `--no-calibrate`).

```python
import asyncio
from nanoma import Runtime, RuntimeConfig

async def main():
    rt = Runtime(config=RuntimeConfig(budget=5.0, default_model="mini"))
    print(await rt.run("Your task here"))

asyncio.run(main())
```

## Tools (15)

`shell` (the universal primitive) · 6 workspace tools (`create_file`, `append_file`, `read_file`,
`replace_string`, `multi_replace`, `grep`) · 8 meta tools (`spawn`, `send`, `wait`, `query`, `kill`,
`set_status`, `get_cost`, `submit`).

*Orthogonal Minimalism*: a dedicated tool exists only where `shell`, the shared filesystem (`$SHARED`),
or a native LLM capability can't do the job. Tools are declared structurally in `nanoma/tools/`; their
JSON schemas are generated.

## Models, aliases & fusion

Declared in `models.yaml`: concrete models, short **aliases** (`nano`/`mini`/`air`/`pro`/`max`), and
**fusion** models (a panel solves the task as sub-agents, then a judge synthesizes). Any name works
wherever a model is accepted (`--model`, `spawn(model=...)`).

## Presets & trace

`presets/` holds 30 prompt templates for known patterns (orchestrator-workers, debate, map-reduce,
mixture-of-agents, …); `{task}` is the placeholder, no code changes needed.

Every run writes a structured trace to `logs/events.jsonl` (one JSON object per event) plus per-call
LLM logs — `tail -f logs/events.jsonl` or parse it directly.

## Security

NanoMA does **not** sandbox shell — agents run commands with the host's privileges (workspace tools
are path-restricted, `shell` is not). **Run it in a disposable container or VM.**

## Development

```bash
make install   # pip install -e ".[dev]"
make all       # lint + test (run before committing)
```

## Citation

```bibtex
@software{he2026nanoma,
  author = {Jiyan He},
  title  = {NanoMA: A Minimal Multi-Agent Harness for Research},
  year   = {2026},
  url    = {https://github.com/volltin/NanoMA}
}
```
