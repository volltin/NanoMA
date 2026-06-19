"""Workspace tools — file operations.

Implements:
- create_file: create a new file (auto-creates parent directories)
- append_file: append content to a file (create if not exists)
- read_file: read file content with offset/limit support
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from nanoma.tools.output import clip_text

if TYPE_CHECKING:
    from nanoma.state import ToolContext


async def tool_create_file(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Create a new file. Auto-creates parent directories.
    
    Args:
        path: File path (relative to workspace or absolute under workspace root)
        content: File content to write
        overwrite: If true, overwrite existing file (default: false)
    """
    file_path = args.get("path", "")
    content = args.get("content", "")
    overwrite = args.get("overwrite", False)

    if not file_path or not file_path.strip():
        return {"error": "path is required and must not be empty"}
    if content is None:
        return {"error": "content is required"}

    path = Path(file_path)
    if not path.is_absolute():
        path = workspace / path

    # Sandbox check
    try:
        path.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if path.exists() and not overwrite:
        return {
            "error": f"File already exists: {path}",
            "hint": "Re-call create_file with overwrite=true to replace it "
                    "(common after scaffolding like `cargo new` creates a stub file).",
            "existing_bytes": path.stat().st_size,
        }

    # Auto-create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}


async def tool_append_file(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Append content to a file. Creates the file if it doesn't exist.

    Args:
        path: File path
        content: Content to append
    """
    file_path = args.get("path", "")
    content = args.get("content", "")

    if not file_path or not file_path.strip():
        return {"error": "path is required"}
    if content is None:
        return {"error": "content is required"}

    path = Path(file_path)
    if not path.is_absolute():
        path = workspace / path

    try:
        path.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(content)

    return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}


async def tool_read_file_advanced(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Read file content with offset/limit support (1-indexed lines).

    Args:
        path: File path
        offset: Starting line number (1-based, optional)
        limit: Maximum number of lines to read (optional, default 2000)
    """
    file_path = args.get("path", "")
    offset = args.get("offset")
    limit = args.get("limit")

    if not file_path or not file_path.strip():
        return {"error": "path is required"}

    path = Path(file_path)
    if not path.is_absolute():
        path = workspace / path

    try:
        path.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}
    if path.is_dir():
        return {"error": f"Path is a directory, not a file: {path}"}

    content = path.read_text(encoding="utf-8", errors="replace")
    all_lines = content.split("\n")
    total_lines = len(all_lines)
    total_chars = len(content)

    # Compute reading range (line-based)
    start_line = max(1, offset) if offset else 1
    max_lines = limit if limit else 2000
    end_line = min(start_line + max_lines - 1, total_lines)

    selected_lines = all_lines[start_line - 1:end_line]
    chunk = "\n".join(selected_lines)

    # Char budget: even a bounded line range can be huge if lines are long. Clip the
    # chunk and report PRECISELY what was shown so the agent can page through the rest.
    max_chars = getattr(ctx, "file_read_max_chars", 0) or 0
    preview, meta = clip_text(chunk, max_chars)
    long_line = False
    if meta["truncated"]:
        if meta["shown_lines"] == 0:
            # A single line longer than the char budget — shown partially. Advance past
            # it so paging can't loop forever; flag it so the agent can widen the read.
            read_to = start_line
            long_line = True
        else:
            read_to = start_line - 1 + meta["shown_lines"]
    else:
        read_to = end_line

    has_more = read_to < total_lines
    result: dict[str, Any] = {
        "content": preview,
        "total_lines": total_lines,
        "total_chars": total_chars,
        "read_from": start_line,
        "read_to": read_to,
        "has_more": has_more,
    }
    if long_line:
        result["note"] = (
            f"Line {start_line} is longer than the {max_chars}-char budget and was shown "
            f"partially ({len(preview)} chars). Use shell (e.g. sed/cut) to read the rest, "
            f"or read_file(path='{file_path}', offset={start_line + 1}) to continue past it."
        )
    elif has_more:
        reason = "char limit" if meta["truncated"] else "line limit"
        result["note"] = (
            f"Showed lines {start_line}–{read_to} of {total_lines} "
            f"({len(preview)} of {total_chars} chars; stopped at {reason}). "
            f"Read more with read_file(path='{file_path}', offset={read_to + 1})."
        )
    return result
