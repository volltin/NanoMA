"""Markdown task runner — run a single ``.md`` task file in a self-contained folder.

A *task* is a Markdown file (conventionally ``task.md``) with optional YAML
front-matter for config. Everything else in the file is the task prompt.

    ---
    model: fusion                 # alias / model / fusion name; omit -> global default
    models: ./models.yaml         # optional custom registry (path relative to the task file)
    budget: 8
    max_agents: 8
    time_limit: 900
    ---

    Port the core of python-slugify to Rust and make `cargo test` pass.
    ...

Run it:

    nanoma-run path/to/task.md          # run a specific file
    nanoma-run path/to/folder           # uses <folder>/task.md

The run is self-contained in the task file's folder:

    <folder>/
      task.md       (input)
      result.md     (output, written here)
      logs/         (events.jsonl + per-call LLM logs)
      workspace/    (agent workspaces + shared/)

Config keys (all optional; CLI flags override front-matter):
  model, models, budget, max_agents, max_depth, max_turns, max_concurrent,
  time_limit, workspace, log_dir, result, base_url, api_key
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# ─── Front-matter parsing ────────────────────────────────────────────────────

def parse_task_file(text: str) -> tuple[dict[str, Any], str]:
    """Split a task file into (config, body).

    Front-matter is an optional YAML block delimited by a leading line of exactly
    ``---`` and a closing line of exactly ``---``. If absent, config is empty and
    the whole text is the body.
    """
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                fm = "\n".join(lines[1:i])
                body = "\n".join(lines[i + 1:]).strip()
                cfg = yaml.safe_load(fm) or {}
                if not isinstance(cfg, dict):
                    raise ValueError("Task front-matter must be a YAML mapping.")
                return cfg, body
        raise ValueError("Unterminated front-matter: missing closing '---'.")
    return {}, text.strip()


# ─── Run ─────────────────────────────────────────────────────────────────────

def _resolve_task_path(path: Path) -> Path:
    if path.is_dir():
        cand = path / "task.md"
        if not cand.exists():
            raise FileNotFoundError(f"No task.md in folder: {path}")
        return cand
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    return path


async def run_task(task_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    from nanoma.core import Runtime, RuntimeConfig
    from nanoma.models import load_models

    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    task_file = _resolve_task_path(Path(task_path)).resolve()
    folder = task_file.parent

    cfg, body = parse_task_file(task_file.read_text(encoding="utf-8"))
    cfg.update(overrides)
    if not body:
        raise ValueError("Task file has no body (the prompt is empty).")

    # Credentials (env preferred; front-matter may set them for convenience)
    if cfg.get("base_url"):
        os.environ["NANOMA_LLM_BASE_URL"] = str(cfg["base_url"])
    if cfg.get("api_key"):
        os.environ["NANOMA_API_KEY"] = str(cfg["api_key"])

    # Custom model registry (relative to the task file unless absolute)
    if cfg.get("models"):
        mp = Path(cfg["models"])
        if not mp.is_absolute():
            mp = folder / mp
        load_models(mp)

    workspace = Path(cfg.get("workspace") or (folder / "workspace"))
    if not workspace.is_absolute():
        workspace = folder / workspace
    log_dir = Path(cfg.get("log_dir") or (folder / "logs"))
    if not log_dir.is_absolute():
        log_dir = folder / log_dir

    rc_kwargs: dict[str, Any] = {
        "workspace_root": workspace,
        "log_dir": log_dir,
    }
    if "model" in cfg:          rc_kwargs["default_model"] = str(cfg["model"])
    if "budget" in cfg:         rc_kwargs["budget"] = float(cfg["budget"])
    if "max_agents" in cfg:     rc_kwargs["max_agents"] = int(cfg["max_agents"])
    if "max_depth" in cfg:      rc_kwargs["max_depth"] = int(cfg["max_depth"])
    if "max_turns" in cfg:      rc_kwargs["max_turns"] = int(cfg["max_turns"])
    if "max_concurrent" in cfg: rc_kwargs["max_concurrent_llm"] = int(cfg["max_concurrent"])
    if "time_limit" in cfg:     rc_kwargs["time_limit"] = float(cfg["time_limit"])
    if "anthropic_native" in cfg: rc_kwargs["anthropic_native"] = bool(cfg["anthropic_native"])

    config = RuntimeConfig(**rc_kwargs)

    # ── Preflight calibration: probe every model this run will use, in the real
    # request shape, so an unusable model fails fast (and cheap) instead of mid-run.
    do_calibrate = cfg.get("calibrate", True)
    calibrate_only = bool(cfg.get("calibrate_only", False))
    if do_calibrate or calibrate_only:
        from nanoma.calibrate import calibrate as _calibrate, models_for_run, format_results, all_ok
        from nanoma.models import get_registry
        probe_models = models_for_run(get_registry(), config.default_model)
        print(f"[calibrate] probing {len(probe_models)} model(s): {', '.join(probe_models)}", file=sys.stderr)
        results = await _calibrate(probe_models, temperature=config.temperature)
        print(format_results(results), file=sys.stderr)
        if calibrate_only:
            return {"calibration": results, "result": "", "result_path": None, "status": "calibrated"}
        if not all_ok(results):
            bad = ", ".join(r["model"] for r in results if not r.get("ok"))
            raise SystemExit(f"[calibrate] ABORT — {bad} unusable for this run. "
                             f"Fix the registry/config, or pass --no-calibrate to skip.")

    def on_event(e):
        ev, d = e["event"], e.get("data", {})
        if ev == "agent_new":
            fus = f" [fusion:{d['fusion']}]" if d.get("fusion") else ""
            print(f"  [+] {e['agent']:<8} {d.get('model','')}{fus}  d{d.get('depth',0)}  {str(d.get('task',''))[:48]}", file=sys.stderr)
        elif ev == "spawn":
            print(f"  [->] {e['agent']:<8} spawn {d.get('child')}", file=sys.stderr)
        elif ev in ("done", "failed"):
            print(f"  [{ev:>6}] {e['agent']:<8} turns={d.get('turns','?')} ${d.get('result','')[:0]}", file=sys.stderr)

    rt = Runtime(config=config, on_event=on_event)

    print(f"[nanoma-run] task:   {task_file}", file=sys.stderr)
    print(f"[nanoma-run] model:  {config.default_model}", file=sys.stderr)
    print(f"[nanoma-run] folder: {folder}", file=sys.stderr)

    t0 = time.time()
    result = await rt.run(body)
    elapsed = time.time() - t0

    stats = rt.stats()
    root = next((a for a in rt.agents.values() if a.parent is None), None)
    status = root.status if root else "unknown"
    result_path = Path(cfg.get("result") or (folder / "result.md"))
    if not result_path.is_absolute():
        result_path = folder / result_path
    _write_result(result_path, task_file, config, result, stats, rt, elapsed, status)
    print(f"[nanoma-run] result: {result_path}", file=sys.stderr)
    return {"result": result, "result_path": str(result_path), "stats": stats, "status": status}


def _write_result(path: Path, task_file: Path, config, result: str,
                  stats: dict, rt, elapsed: float, status: str) -> None:
    ov = stats.get("overview", {})
    ag = stats.get("agents", {})
    shared = config.workspace_root / config.shared_dir
    artifacts = sorted(p.name for p in shared.glob("*")) if shared.exists() else []

    lines = [
        f"# Result — {task_file.parent.name}",
        "",
        f"- Task file: `{task_file.name}`",
        f"- Model: `{config.default_model}`",
        f"- Status: **{status}**",
        f"- Elapsed: {elapsed:.1f}s",
        f"- Cost: ${ov.get('total_cost_usd', 0)} of ${config.budget} budget",
        f"- Agents: {ag.get('total_spawned', 0)} spawned "
        f"(peak {ag.get('peak_concurrent', 0)} concurrent, depth {ag.get('max_depth', 0)})",
        f"- Tokens: {ov.get('total_tokens', 0)}",
        "",
        "## Final result",
        "",
        (result or "_(empty)_").strip(),
        "",
    ]
    if artifacts:
        lines += ["## Artifacts (shared/)", ""]
        lines += [f"- `{a}`" for a in artifacts]
        lines += [""]
    # Per-agent breakdown
    lines += ["## Agents", "", "| id | model | status | turns | tokens | cost $ |", "|----|-------|--------|-------|--------|--------|"]
    for a in sorted(rt.agents.values(), key=lambda a: a.id):
        c = rt.ledger.per_agent.get(a.id, 0.0)
        lines.append(f"| {a.id} | {a.model} | {a.status} | {a._turns} | {a.tokens_consumed} | {c:.4f} |")
    lines += ["", "---", f"_Generated by nanoma-run at {time.strftime('%Y-%m-%d %H:%M:%S')}._", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def cli():
    p = argparse.ArgumentParser(
        prog="nanoma-run",
        description="Run a Markdown task file (with optional YAML front-matter) in a self-contained folder.",
    )
    p.add_argument("path", help="Path to a task .md file, or a folder containing task.md")
    p.add_argument("--model", default=None, help="Override model (alias / model / fusion name)")
    p.add_argument("--models", default=None, help="Override model registry YAML path")
    p.add_argument("--budget", type=float, default=None)
    p.add_argument("--max-agents", type=int, default=None, dest="max_agents")
    p.add_argument("--time-limit", type=float, default=None, dest="time_limit")
    p.add_argument("--no-calibrate", action="store_true", help="Skip the preflight model calibration")
    p.add_argument("--calibrate-only", action="store_true", help="Only run the model calibration, then exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    import logging
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    overrides = {
        "model": args.model, "models": args.models, "budget": args.budget,
        "max_agents": args.max_agents, "time_limit": args.time_limit,
        "calibrate": False if args.no_calibrate else None,
        "calibrate_only": True if args.calibrate_only else None,
    }
    try:
        out = asyncio.run(run_task(args.path, overrides))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    print(out["result"])


if __name__ == "__main__":
    cli()
