"""Work-tool registry — shell + structured workspace tools, declared via Tool/arg.

This replaces the old hand-written JSON schema dicts: each tool is a `Tool` with
typed `arg(...)` specs, and the OpenAI schema is generated from them.
"""

from __future__ import annotations

from nanoma.tools.spec import Tool, arg, obj, registry
from nanoma.tools.shell import SHELL_TOOL
from nanoma.tools.file_ops import (
    tool_create_file, tool_append_file, tool_read_file_advanced,
)
from nanoma.tools.file_edit import (
    tool_replace_string, tool_multi_replace_string,
)
from nanoma.tools.grep import tool_grep_search


WORK_TOOLS: dict[str, Tool] = registry(
    SHELL_TOOL,

    # ─── File I/O (Reliability: shell heredocs break on special chars) ────
    Tool(
        "create_file",
        "Create a new file with content. Auto-creates parent directories. Fails if file exists "
        "unless overwrite=true. Use this instead of shell heredocs when content contains backticks, "
        "$, or backslashes.",
        tool_create_file,
        args=[
            arg("path", str, "File path (relative to workspace)"),
            arg("content", str, "File content to write"),
            arg("overwrite", bool, "If true, overwrite existing file", default=False),
        ],
    ),
    Tool(
        "append_file",
        "Append content to a file (creates if not exists). Use for incremental writes or large "
        "files that exceed argument limits.",
        tool_append_file,
        args=[
            arg("path", str, "File path"),
            arg("content", str, "Content to append"),
        ],
    ),
    Tool(
        "read_file",
        "Read file content with line-based pagination. Returns {content, total_lines, total_chars, "
        "read_from, read_to, has_more} and, when there is more, a precise 'note' telling you exactly "
        "what was shown and the next offset. Lines are 1-indexed; default limit is 2000 lines, and a "
        "long chunk is further capped by a char budget — so check has_more/note and continue with "
        "offset=read_to+1.",
        tool_read_file_advanced,
        args=[
            arg("path", str, "File path"),
            arg("offset", int, "Start line number (1-based)", required=False),
            arg("limit", int, "Max lines to read (default: 2000)", required=False),
        ],
    ),

    # ─── File Editing (Atomicity: multi-step ops impossible in sed) ───────
    Tool(
        "replace_string",
        "Find and replace a string in a file. Uses 4-tier matching: exact → trimmed → "
        "indent-agnostic → normalized whitespace. The old_string must match exactly once. "
        "File is unchanged if no match found.",
        tool_replace_string,
        args=[
            arg("path", str, "File path"),
            arg("old_string", str, "Text to find (must appear exactly once)"),
            arg("new_string", str, "Replacement text"),
        ],
    ),
    Tool(
        "multi_replace",
        "Execute multiple find-replace operations on a file atomically. All replacements must "
        "succeed or the file is unchanged (all-or-nothing).",
        tool_multi_replace_string,
        args=[
            arg("path", str, "File path"),
            arg("replacements", list, "List of {old_string, new_string} pairs", items=obj(
                arg("old_string", str), arg("new_string", str),
            )),
        ],
    ),
    # ─── Grep (Structured I/O: no CLI equivalent) ──────────────────
    Tool(
        "grep",
        "Fast text or regex search across workspace files. Returns structured results with file "
        "path, line number, content, and match positions. Auto-skips binary files and noise "
        "directories (.git, node_modules, etc.).",
        tool_grep_search,
        args=[
            arg("query", str, "Search text or ERE regex pattern"),
            arg("is_regexp", bool, "Whether query is a regex", default=False),
            arg("case_sensitive", bool, "Case-sensitive search", default=False),
            arg("include_pattern", str, "Glob to filter files (e.g. '*.py')", required=False),
            arg("max_results", int, "Max results (default: 100)", default=100),
        ],
    ),
)


def build_all_tools() -> dict[str, Tool]:
    """The complete tool set an agent sees: work tools (shell + workspace) + meta tools.

    Single source of truth — use this instead of merging registries by hand. Meta tools
    are imported lazily to avoid an import cycle (meta ← core ← tools).
    """
    from nanoma.tools.meta import META_TOOLS
    return {**WORK_TOOLS, **META_TOOLS}
