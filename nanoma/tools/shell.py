"""The shell tool — NanoMA's universal primitive / escape hatch."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.tools.spec import Tool, arg
from nanoma.tools.output import clip_text, truncation_note

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

    # Bound each stream: return a preview, save the full text, report PRECISELY what
    # was shown (chars + lines + next offset) so the agent can page through the rest.
    for key in ("stdout", "stderr"):
        val = result.get(key, "")
        preview, meta = clip_text(val, max_output)
        if not meta["truncated"]:
            continue
        out_dir = workspace / ".cmd-output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{uuid.uuid4().hex[:8]}_{key}.txt"
        out_file.write_text(val)
        rel = str(out_file.relative_to(workspace))
        result[key] = preview
        result[f"{key}_truncated"] = True
        result[f"{key}_total_chars"] = meta["total_chars"]
        result[f"{key}_total_lines"] = meta["total_lines"]
        result[f"{key}_shown_chars"] = meta["shown_chars"]
        result[f"{key}_shown_lines"] = meta["shown_lines"]
        result[f"{key}_file"] = rel
        result[f"{key}_note"] = truncation_note(meta, file_ref=rel)

    return result


SHELL_TOOL = Tool(
    "shell",
    "Execute a shell command in your workspace directory. Returns {stdout, stderr, exit_code}. "
    "CWD is your private workspace. $SHARED references the shared directory. Use shell for: "
    "mkdir -p, rm -rf, mv, ls, find, tree, git, pip, curl, and any other system command. "
    "If a stream is long it is truncated to a preview and the FULL output is saved to a file; the "
    "result then also includes <stream>_truncated, _total_chars, _total_lines, _shown_chars, "
    "_shown_lines, _file and a precise _note. Read more of it with "
    "read_file(path=<stream>_file, offset=<next line>).",
    tool_shell,
    args=[
        arg("command", str, "Shell command to execute"),
        arg("timeout", int, "Max execution time in seconds (default: 30)", default=30),
    ],
)
