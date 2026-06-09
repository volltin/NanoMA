"""Work tools: shell.

The shell tool is NanoMA's universal primitive — the "escape hatch" that covers
any operation not handled by the 9 structured workspace tools or 12 meta tools.

Design principle: everything that CAN be done via a one-liner shell command
SHOULD use shell. Dedicated tools exist only when shell cannot be reliable.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import ToolContext


async def tool_shell(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Execute a shell command in the agent's workspace directory.

    Environment variables:
        WORKSPACE — agent's private workspace path
        SHARED    — shared directory visible to all agents
    """
    from nanoma.sandbox import shell_exec
    cmd = args.get("command", "")
    timeout = args.get("timeout", 30)
    max_output = ctx.shell_max_output

    if not cmd or not cmd.strip():
        return {"error": "command is required"}

    result = await shell_exec(cmd, workspace, ctx.shared_dir, timeout)

    # Truncate large outputs and spill to file
    for key in ("stdout", "stderr"):
        val = result.get(key, "")
        if max_output > 0 and len(val) > max_output:
            # Use uuid for unique, collision-free file naming
            out_dir = workspace / ".cmd-output"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{uuid.uuid4().hex[:8]}_{key}.txt"
            out_file.write_text(val)
            result[key] = val[:max_output] + f"\n...(truncated {len(val)} chars, full: {out_file.name})"
            result[f"{key}_file"] = str(out_file.relative_to(workspace))

    return result


# --- Registry ---

WORK_TOOLS: dict[str, dict[str, Any]] = {
    "shell": {
        "handler": tool_shell,
        "schema": {"type": "function", "function": {
            "name": "shell",
            "description": "Execute a shell command in your workspace directory. Returns {stdout, stderr, exit_code}. CWD is your private workspace. $SHARED references the shared directory. Long outputs are truncated and saved to a file. Use shell for: mkdir -p, rm -rf, mv, ls, find, tree, git, pip, curl, and any other system command.",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Max execution time in seconds (default: 30)", "default": 30},
            }, "required": ["command"]},
        }},
    },
}
