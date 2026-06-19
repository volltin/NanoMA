"""Workspace tool — grep: fast text/regex search across files.

Strategy: ripgrep (rg) → GNU grep → pure-Python fallback. Returns structured
results (file, line, content, match offsets), auto-skipping binary files and
noise directories.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nanoma.core import ToolContext


# ─── Helpers ─────────────────────────────────────────────────────────────────

BINARY_EXTENSIONS = frozenset([
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".wav", ".avi",
    ".sqlite", ".db", ".pyc", ".pyo",
])

IGNORE_DIRS = frozenset([
    "node_modules", ".git", "dist", "target", "__pycache__",
    "coverage", ".venv", "venv", ".tox", ".mypy_cache",
])

def escape_regex(s: str) -> str:
    """Escape regex special characters."""
    return re.escape(s)


def is_binary_file(file_path: str) -> bool:
    """Check if a file is likely binary based on extension."""
    return Path(file_path).suffix.lower() in BINARY_EXTENSIONS


def _glob_to_simple_regex(pattern: str) -> re.Pattern:
    """Convert a simple glob to regex for file filtering."""
    escaped = pattern
    escaped = re.sub(r'[.+^${}()|[\]\\]', r'\\\g<0>', escaped)
    escaped = escaped.replace("**", "§§")
    escaped = escaped.replace("*", "[^/]*")
    escaped = escaped.replace("§§", ".*")
    escaped = escaped.replace("?", "[^/]")
    return re.compile(escaped)


# ─── Tool Implementations ────────────────────────────────────────────────────

async def tool_grep_search(args: dict[str, Any], workspace: Path, ctx: "ToolContext") -> dict[str, Any]:
    """Fast text/regex search across workspace files.

    Strategy: try ripgrep (rg) first for performance, fall back to GNU grep,
    then to pure Python as last resort. Subprocess grep is 10-100x faster
    on large workspaces (SIMD, mmap, multi-threading).

    Args:
        query: Search text or regex pattern
        is_regexp: Whether query is a regex
        case_sensitive: If true, search is case-sensitive (default: false)
        include_pattern: Glob to filter files (e.g., "*.py")
        max_results: Maximum results to return (default: 100)
    """
    query = args.get("query", "")
    is_regexp = args.get("is_regexp", False)
    case_sensitive = args.get("case_sensitive", False)
    include_pattern = args.get("include_pattern")
    max_results = min(args.get("max_results", 100), ctx.grep_max_results or 200)

    if not query:
        return {"error": "query is required"}

    # Try subprocess grep for performance
    result = await _try_subprocess_grep(
        query, workspace, is_regexp, case_sensitive, include_pattern, max_results
    )
    if result is not None:
        return result

    # Fallback: pure Python implementation
    return _python_grep(query, workspace, is_regexp, case_sensitive, include_pattern, max_results)


async def _try_subprocess_grep(
    query: str, workspace: Path, is_regexp: bool, case_sensitive: bool,
    include_pattern: str | None, max_results: int,
) -> dict[str, Any] | None:
    """Try to use rg or grep subprocess. Returns None if unavailable."""
    import asyncio
    import shutil

    # Build command: prefer ripgrep, fallback to GNU grep
    rg_path = shutil.which("rg")
    grep_path = shutil.which("grep")

    if rg_path:
        cmd = [rg_path, "--no-heading", "--line-number", "--color=never",
               f"--max-count={max_results}"]
        if not case_sensitive:
            cmd.append("--ignore-case")
        if not is_regexp:
            cmd.append("--fixed-strings")
        if include_pattern:
            cmd.extend(["--glob", include_pattern])
        cmd.append(query)
        cmd.append(str(workspace))
    elif grep_path:
        cmd = [grep_path, "-rn", "--color=never", f"--max-count={max_results}"]
        if not case_sensitive:
            cmd.append("-i")
        if not is_regexp:
            cmd.append("-F")
        if include_pattern:
            cmd.extend(["--include", include_pattern])
        cmd.append(query)
        cmd.append(str(workspace))
    else:
        return None  # No subprocess available, use Python fallback

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace")
        # Exit code 2+ means error (invalid regex, permission denied, etc.)
        # Fall back to Python for proper error reporting
        if proc.returncode and proc.returncode >= 2:
            return None
    except (asyncio.TimeoutError, OSError):
        return None  # Fallback to Python on error

    # Parse output: "filepath:line:content"
    results: list[dict[str, Any]] = []
    for raw_line in output.split("\n"):
        if not raw_line or len(results) >= max_results:
            break
        # Parse "file:line:content" format
        parts = raw_line.split(":", 2)
        if len(parts) < 3:
            continue
        file_path_str, line_no_str, content = parts[0], parts[1], parts[2]
        try:
            line_no = int(line_no_str)
        except ValueError:
            continue
        # Make path relative to workspace
        try:
            rel_path = str(Path(file_path_str).relative_to(workspace))
        except ValueError:
            rel_path = file_path_str
        results.append({
            "file": rel_path,
            "line": line_no,
            "content": content,
        })

    return {"matches": results, "count": len(results), "truncated": len(results) >= max_results}


def _python_grep(
    query: str, workspace: Path, is_regexp: bool, case_sensitive: bool,
    include_pattern: str | None, max_results: int,
) -> dict[str, Any]:
    """Pure Python grep fallback (used when no system grep is available)."""
    # Build search regex
    try:
        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regexp:
            search_regex = re.compile(query, flags)
        else:
            search_regex = re.compile(escape_regex(query), flags)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    # File filter
    file_filter = _glob_to_simple_regex(include_pattern) if include_pattern else None

    results: list[dict[str, Any]] = []

    def search_file(file_path: Path):
        if len(results) >= max_results:
            return

        rel_path = str(file_path.relative_to(workspace))

        # File filter check
        if file_filter and not file_filter.search(rel_path) and not file_filter.search(file_path.name):
            return

        if is_binary_file(str(file_path)):
            return

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(content.split("\n")):
                if len(results) >= max_results:
                    return
                m = search_regex.search(line)
                if m:
                    results.append({
                        "file": rel_path,
                        "line": i + 1,
                        "content": line,
                        "match_start": m.start(),
                        "match_end": m.end(),
                    })
        except (OSError, UnicodeDecodeError):
            pass

    def walk_dir(dir_path: Path):
        if len(results) >= max_results:
            return
        try:
            items = sorted(dir_path.iterdir())
        except PermissionError:
            return
        for item in items:
            if len(results) >= max_results:
                return
            if item.is_dir():
                if item.name not in IGNORE_DIRS and not item.name.startswith("."):
                    walk_dir(item)
            elif item.is_file():
                search_file(item)

    walk_dir(workspace)

    return {"matches": results, "count": len(results), "truncated": len(results) >= max_results}
