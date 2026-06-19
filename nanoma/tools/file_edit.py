"""Workspace tools — file editing.

Implements:
- replace_string: 4-tier exact find and replace
- multi_replace: atomic batch find and replace
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import ToolContext


# ─── Core matching logic ─────────────────────────────────────────────────────

def count_occurrences(text: str, search: str) -> int:
    """Count non-overlapping occurrences of search in text."""
    if not search:
        return 0
    count = 0
    pos = 0
    while True:
        pos = text.find(search, pos)
        if pos == -1:
            break
        count += 1
        pos += len(search)
    return count


def find_and_replace(
    content: str, old_string: str, new_string: str
) -> dict[str, Any]:
    """4-tier find and replace logic.

    Tier 1: Exact match
    Tier 2: Trimmed match (ignore leading/trailing whitespace)
    Tier 3: Indent-agnostic match (strip leading whitespace per line)
    Tier 4: Normalized whitespace match (collapse all whitespace)

    Returns:
        {"success": True, "new_content": str, "tier": int} on success
        {"success": False, "error": str} on failure
    """
    # Tier 1: Exact match
    exact_count = count_occurrences(content, old_string)
    if exact_count == 1:
        return {
            "success": True,
            "new_content": content.replace(old_string, new_string, 1),
            "tier": 1,
        }
    if exact_count > 1:
        return {
            "success": False,
            "error": f"Ambiguous match: oldString appears {exact_count} times in the file. It must appear exactly once.",
        }

    # Tier 2: Trimmed match
    trimmed_old = old_string.strip()
    if trimmed_old:
        trimmed_count = count_occurrences(content, trimmed_old)
        if trimmed_count == 1:
            return {
                "success": True,
                "new_content": content.replace(trimmed_old, new_string, 1),
                "tier": 2,
            }
        if trimmed_count > 1:
            return {
                "success": False,
                "error": f"Ambiguous match (tier 2, trimmed): found {trimmed_count} occurrences.",
            }

    # Tier 3: Indent-agnostic match
    tier3_result = _try_indent_agnostic_match(content, old_string, new_string)
    if tier3_result is not None:
        if tier3_result.get("ambiguous"):
            return {
                "success": False,
                "error": "Ambiguous match (tier 3, indent-agnostic): multiple matches found.",
            }
        return {"success": True, "new_content": tier3_result["new_content"], "tier": 3}

    # Tier 4: Normalized whitespace match
    tier4_result = _try_normalized_match(content, old_string, new_string)
    if tier4_result is not None:
        if tier4_result.get("ambiguous"):
            return {
                "success": False,
                "error": "Ambiguous match (tier 4, normalized whitespace): multiple matches found.",
            }
        return {"success": True, "new_content": tier4_result["new_content"], "tier": 4}

    return {
        "success": False,
        "error": "oldString not found in the file (tried all 4 matching tiers: exact, trimmed, indent-agnostic, normalized).",
    }


def _try_indent_agnostic_match(
    content: str, old_string: str, new_string: str
) -> dict[str, Any] | None:
    """Tier 3: Match ignoring leading whitespace on each line."""
    old_lines = [line.lstrip() for line in old_string.split("\n")]
    content_lines = content.split("\n")

    matches: list[int] = []
    for i in range(len(content_lines) - len(old_lines) + 1):
        match = True
        for j in range(len(old_lines)):
            if content_lines[i + j].lstrip() != old_lines[j]:
                match = False
                break
        if match:
            matches.append(i)

    if not matches:
        return None
    if len(matches) > 1:
        return {"new_content": "", "ambiguous": True}

    # Replace matched lines
    match_start = matches[0]
    new_lines = list(content_lines)
    new_lines[match_start:match_start + len(old_lines)] = new_string.split("\n")
    return {"new_content": "\n".join(new_lines)}


def _try_normalized_match(
    content: str, old_string: str, new_string: str
) -> dict[str, Any] | None:
    """Tier 4: Match after normalizing all whitespace."""

    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    normalized_old = normalize(old_string)
    content_lines = content.split("\n")
    old_line_count = len(old_string.split("\n"))

    matches: list[tuple[int, int]] = []

    for i in range(len(content_lines) - old_line_count + 1):
        # Try windows of different sizes (allow extra empty lines)
        for length in range(old_line_count, min(old_line_count + 3, len(content_lines) - i + 1)):
            window = "\n".join(content_lines[i:i + length])
            if normalize(window) == normalized_old:
                matches.append((i, i + length))
                break

    if not matches:
        return None
    if len(matches) > 1:
        return {"new_content": "", "ambiguous": True}

    start, end = matches[0]
    new_lines = list(content_lines)
    new_lines[start:end] = new_string.split("\n")
    return {"new_content": "\n".join(new_lines)}


# ─── Tool Implementations ────────────────────────────────────────────────────

async def tool_replace_string(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Find and replace a string in a file (4-tier matching).

    Args:
        path: File path
        old_string: Exact text to find (must appear exactly once)
        new_string: Replacement text
    """
    file_path = args.get("path", "")
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")

    if not file_path or not file_path.strip():
        return {"error": "path is required"}
    if not old_string:
        return {"error": "old_string is required and must not be empty"}
    if new_string is None:
        return {"error": "new_string is required"}

    path = Path(file_path)
    if not path.is_absolute():
        path = workspace / path

    try:
        path.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}

    content = path.read_text(encoding="utf-8")
    result = find_and_replace(content, old_string, new_string)

    if not result["success"]:
        return {"error": result["error"]}

    path.write_text(result["new_content"], encoding="utf-8")
    return {"match_tier": result["tier"]}


async def tool_multi_replace_string(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Execute multiple find-replace operations atomically.

    Args:
        path: File path
        replacements: List of {"old_string": str, "new_string": str}
    """
    file_path = args.get("path", "")
    replacements = args.get("replacements", [])

    if not file_path or not file_path.strip():
        return {"error": "path is required"}
    if not replacements:
        return {"error": "replacements array must not be empty"}

    path = Path(file_path)
    if not path.is_absolute():
        path = workspace / path

    try:
        path.resolve().relative_to(ctx.workspace_root.resolve())
    except ValueError:
        return {"error": "Access denied: path is outside workspace root"}

    if not path.exists():
        return {"error": f"File not found: {path}"}

    content = path.read_text(encoding="utf-8")

    # Apply each replacement sequentially (in-memory)
    for i, repl in enumerate(replacements):
        old_str = repl.get("old_string", "")
        new_str = repl.get("new_string", "")

        if not old_str:
            return {"error": f"Replacement #{i + 1}: old_string must not be empty"}

        result = find_and_replace(content, old_str, new_str)
        if not result["success"]:
            return {"error": f"Replacement #{i + 1} failed: {result['error']}. File NOT modified (all-or-nothing)."}

        content = result["new_content"]

    # All succeeded — write to file
    path.write_text(content, encoding="utf-8")
    return {"replaced_count": len(replacements)}

