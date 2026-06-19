"""System-prompt construction: base prompt, model menu, and fusion directive.

Pure functions of the model registry + agent facts, so prompt wording lives in one
place and the runtime stays focused on orchestration.
"""

from __future__ import annotations

from pathlib import Path


def concrete_model(reg, name: str) -> str:
    """Resolve a model name to a concrete (non-fusion) model.

    If it resolves to a fusion model, descend to its judge (or first panel member)
    so panel/judge agents never recursively re-fuse.
    """
    resolved = reg.resolve(name)
    if reg.is_fusion(resolved):
        f = reg.get_fusion(resolved)
        if f:
            return concrete_model(reg, f.judge or (f.panel[0] if f.panel else resolved))
    return resolved


def build_models_section(reg) -> str:
    """List available aliases and fusion models for spawn(model=...). '' if none."""
    parts = []
    if reg.aliases:
        alias_str = ", ".join(f"{a} → {reg.resolve(a)}" for a in sorted(reg.aliases))
        parts.append(f"- Aliases: {alias_str}")
    if reg.fusions:
        fus = []
        for name, f in reg.fusions.items():
            judge = f.judge or (f.panel[0] if f.panel else "?")
            fus.append(f"{name} (panel: {', '.join(f.panel)}; judge: {judge})")
        parts.append(
            "- Fusion models (using one as a model runs a Mixture-of-Agents: a panel "
            "solves the task as sub-agents, then a judge synthesizes — higher quality, "
            "higher cost): " + "; ".join(fus)
        )
    if not parts:
        return ""
    return (
        "\n## Models\n"
        "When you spawn(model=...), you may use a concrete id, an alias, or a fusion model:\n"
        + "\n".join(parts)
        + "\nUse a fusion model for hard reasoning/research where quality matters; "
        "use a cheap alias for routine subtasks.\n"
    )


def build_fusion_directive(reg, fdef) -> str:
    """Mixture-of-Agents directive appended to a fusion agent's system prompt.

    The agent runs on the judge model and orchestrates the panel as real sub-agents
    via the ordinary spawn/wait/query tools.
    """
    panel = [concrete_model(reg, m) for m in fdef.panel]
    judge = concrete_model(reg, fdef.judge or (fdef.panel[0] if fdef.panel else "?"))
    spawn_lines = "\n".join(
        f'   {i}. spawn(task="<the COMPLETE task, to be solved independently and thoroughly>", model="{m}")'
        for i, m in enumerate(panel, 1)
    )
    return f"""
## Model Fusion — REQUIRED procedure (Mixture-of-Agents)
This is a *fusion* run (`{fdef.name}`). Do NOT answer the task yourself first.
Instead, fuse a panel of models and synthesize the best result. You are the JUDGE,
running model `{judge}`.

Procedure:
1. Spawn one sub-agent per panel model, each given the COMPLETE task and asked to
   solve it independently and report its answer (and write any files):
{spawn_lines}
2. wait(mode="all") until the whole panel finishes.
3. Read every panel answer (use the results they report; query(agent_id, messages=-1) to pull
   details, or read files they wrote to the shared directory).
4. Synthesize ONE superior answer: keep what the panel AGREES on (higher confidence),
   resolve CONTRADICTIONS with your own judgment, fold in UNIQUE insights, and drop
   errors / blind spots. Do not merely pick one answer or concatenate them.
5. If the deliverable is a file, submit() it. Then set_status("done", result=<synthesis>).

The whole point is combining multiple models — always consult the panel.
"""


def build_system_prompt(
    reg,
    *,
    agent_id: str,
    task: str,
    workspace: Path,
    shared_dir: Path,
    time_limit: float,
    parent_context: dict | None = None,
) -> str:
    time_info = f"\nTime limit: {time_limit:.0f}s total. Check with get_cost()." if time_limit > 0 else ""

    context_section = ""
    if parent_context:
        context_section = f"""
## Your Context
- Spawned by: agent "{parent_context['parent_id']}" (task: {parent_context['parent_task']})
- Peers working in parallel: {parent_context['siblings'] or 'none yet'}
- Depth: {parent_context['depth']}
- When you finish, call set_status("done", result="<your result>") — your spawner is notified automatically with that result. Use send() only to coordinate mid-task, not to report completion.
"""

    models_section = build_models_section(reg) if reg is not None else ""

    return f"""You are agent "{agent_id}" in a multi-agent system.

Task: {task}
Workspace: {workspace} (private to you)
Shared: {shared_dir} (visible to all agents){time_info}
{context_section}
## How to act
- Every turn must call at least one tool. Plain prose is NOT delivered to anyone and wastes a step — narrate inside a tool call's reasoning, then act.
- You finish ONLY by calling set_status(status="done", result="<your final answer/summary>"). That result is the only thing returned to your caller — explanatory messages are not. Call it the moment the task is complete; don't add an extra "summary" turn.
- If you are blocked and cannot proceed, call set_status(status="failed", result="<why>").
- Deliver file outputs by writing them to the shared directory (or submit()).

## Tool Philosophy
You have tools in 3 layers:
- shell: universal primitive. Use for mkdir, rm, mv, ls, find, tree, git, pip, curl, etc.
- workspace tools: structured operations (file create/read/edit, grep)
- meta tools: coordination (spawn, send, wait, query, kill, set_status, submit, get_cost)

When in doubt, use shell. The workspace tools exist only for operations shell can't do reliably.
Share files with other agents by writing to the shared directory ($SHARED); read theirs from there.
{models_section}"""
