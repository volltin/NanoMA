"""nanoma.tools — all tool code in one place.

Layout:
- spec.py        : Tool / arg / obj / registry — structured schema generation
- shell.py       : the shell tool (universal primitive)
- file_ops.py    : create_file / append_file / read_file
- file_edit.py    : replace_string / multi_replace
- grep.py        : grep (ripgrep → grep → python fallback)
- meta.py        : coordination tools (spawn, send, wait, …) + META_TOOLS
- registry.py    : WORK_TOOLS (shell + workspace tools) and build_all_tools()

The full registry an agent sees is `build_all_tools()` = WORK_TOOLS + META_TOOLS.
"""

from __future__ import annotations

from nanoma.tools.spec import Tool, Arg, arg, obj, registry, build_function_schema
from nanoma.tools.shell import SHELL_TOOL, tool_shell
from nanoma.tools.registry import WORK_TOOLS, build_all_tools
from nanoma.tools.meta import META_TOOLS

# Handler re-exports (convenience / back-compat)
from nanoma.tools.file_ops import (
    tool_create_file, tool_append_file, tool_read_file_advanced,
)
from nanoma.tools.file_edit import (
    tool_replace_string, tool_multi_replace_string,
)
from nanoma.tools.grep import tool_grep_search

__all__ = [
    # schema layer
    "Tool", "Arg", "arg", "obj", "registry", "build_function_schema",
    # registries
    "WORK_TOOLS", "META_TOOLS", "build_all_tools",
    # shell
    "SHELL_TOOL", "tool_shell",
    # workspace tool handlers
    "tool_create_file", "tool_append_file", "tool_read_file_advanced",
    "tool_replace_string", "tool_multi_replace_string",
    "tool_grep_search",
]
