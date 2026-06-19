"""Tests for nanoma tools: shell + the 6 workspace tools + their helpers.

Covers:
- nanoma/tools/shell.py        (tool_shell, output truncation)
- nanoma/tools/file_ops.py     (create_file, append_file, read_file)
- nanoma/tools/file_edit.py    (replace_string, multi_replace, 4-tier matcher)
- nanoma/tools/grep.py        (grep + subprocess/python fallback + helpers)
"""

import asyncio
import pytest
from unittest.mock import patch

from nanoma.core import ToolContext
from nanoma.tools import tool_shell, WORK_TOOLS, build_all_tools
from nanoma.tools.file_ops import (
    tool_create_file, tool_append_file, tool_read_file_advanced,
)
from nanoma.tools.file_edit import (
    tool_replace_string, tool_multi_replace_string,
    find_and_replace, count_occurrences,
    _try_indent_agnostic_match, _try_normalized_match,
)
from nanoma.tools.grep import (
    tool_grep_search, escape_regex, is_binary_file,
    _glob_to_simple_regex, _try_subprocess_grep, _python_grep,
    BINARY_EXTENSIONS, IGNORE_DIRS,
)

# Workspace tools = everything in WORK_TOOLS except shell.
WORKSPACE_TOOLS = {k: v for k, v in WORK_TOOLS.items() if k != "shell"}


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "shared").mkdir()
    return tmp_path


@pytest.fixture
def ctx(ws):
    return ToolContext(
        shared_dir=ws / "shared", workspace_root=ws,
        shell_max_output=200, file_read_max_chars=50000,
        file_list_max_entries=10, grep_max_results=50,
    )


# ─── Registry ────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_counts(self):
        assert len(WORK_TOOLS) == 7          # shell + 6 workspace tools
        assert "shell" in WORK_TOOLS
        assert len(WORKSPACE_TOOLS) == 6
        assert len(build_all_tools()) == 15  # + 8 meta tools

    def test_expected_names(self):
        assert set(WORKSPACE_TOOLS) == {
            "create_file", "append_file", "read_file",
            "replace_string", "multi_replace", "grep",
        }

    def test_schema_shape(self):
        for name, tool in build_all_tools().items():
            s = tool["schema"]
            assert s["type"] == "function"
            assert s["function"]["name"] == name
            assert s["function"]["parameters"]["type"] == "object"
            assert callable(tool["handler"])
            assert "handler" in tool


# ─── Shell ───────────────────────────────────────────────────────────────────

class TestShell:
    @pytest.mark.asyncio
    async def test_basic(self, ws, ctx):
        r = await tool_shell({"command": "echo hello"}, ws, ctx)
        assert r["exit_code"] == 0 and "hello" in r["stdout"]

    @pytest.mark.asyncio
    async def test_empty(self, ws, ctx):
        assert "error" in await tool_shell({"command": "  "}, ws, ctx)

    @pytest.mark.asyncio
    async def test_nonzero_exit(self, ws, ctx):
        assert (await tool_shell({"command": "exit 42"}, ws, ctx))["exit_code"] == 42

    @pytest.mark.asyncio
    async def test_stderr(self, ws, ctx):
        assert "err" in (await tool_shell({"command": "echo err >&2"}, ws, ctx))["stderr"]

    @pytest.mark.asyncio
    async def test_timeout(self, ws, ctx):
        r = await tool_shell({"command": "sleep 100", "timeout": 1}, ws, ctx)
        assert r["exit_code"] == -1 and "Timeout" in r["stderr"]

    @pytest.mark.asyncio
    async def test_truncation_reports_precisely(self, ws, ctx):
        r = await tool_shell({"command": "seq 1 1000"}, ws, ctx)
        assert r["stdout_truncated"] is True
        assert r["stdout_total_chars"] > r["stdout_shown_chars"]
        assert r["stdout_total_lines"] >= 1000
        assert "truncated" in r["stdout_note"].lower()
        full = ws / r["stdout_file"]
        assert full.exists() and len(full.read_text()) > 200

    @pytest.mark.asyncio
    async def test_no_truncation_when_short(self, ws, ctx):
        assert "stdout_file" not in await tool_shell({"command": "echo short"}, ws, ctx)

    @pytest.mark.asyncio
    async def test_env_and_cwd(self, ws, ctx):
        assert str(ws) in (await tool_shell({"command": "pwd"}, ws, ctx))["stdout"]
        assert str(ws) in (await tool_shell({"command": "echo $WORKSPACE"}, ws, ctx))["stdout"]


# ─── create_file ─────────────────────────────────────────────────────────────

class TestCreateFile:
    @pytest.mark.asyncio
    async def test_create(self, ws, ctx):
        r = await tool_create_file({"path": "a.txt", "content": "Hello"}, ws, ctx)
        assert r["bytes_written"] == 5 and (ws / "a.txt").read_text() == "Hello"

    @pytest.mark.asyncio
    async def test_auto_parents(self, ws, ctx):
        await tool_create_file({"path": "a/b/c.txt", "content": "x"}, ws, ctx)
        assert (ws / "a/b/c.txt").exists()

    @pytest.mark.asyncio
    async def test_exists_no_overwrite(self, ws, ctx):
        (ws / "e.txt").write_text("orig")
        r = await tool_create_file({"path": "e.txt", "content": "new"}, ws, ctx)
        assert "error" in r and "hint" in r and (ws / "e.txt").read_text() == "orig"

    @pytest.mark.asyncio
    async def test_overwrite(self, ws, ctx):
        (ws / "e.txt").write_text("orig")
        await tool_create_file({"path": "e.txt", "content": "new", "overwrite": True}, ws, ctx)
        assert (ws / "e.txt").read_text() == "new"

    @pytest.mark.asyncio
    async def test_empty_path(self, ws, ctx):
        assert "error" in await tool_create_file({"path": "  ", "content": "x"}, ws, ctx)

    @pytest.mark.asyncio
    async def test_none_content(self, ws, ctx):
        assert "error" in await tool_create_file({"path": "x.txt", "content": None}, ws, ctx)

    @pytest.mark.asyncio
    async def test_sandbox_escape(self, ws, ctx):
        r = await tool_create_file({"path": "../../etc/x", "content": "x"}, ws, ctx)
        assert "error" in r

    @pytest.mark.asyncio
    async def test_utf8(self, ws, ctx):
        content = "中文 🎉 émoji"
        await tool_create_file({"path": "u.txt", "content": content}, ws, ctx)
        assert (ws / "u.txt").read_text() == content


# ─── append_file ─────────────────────────────────────────────────────────────

class TestAppendFile:
    @pytest.mark.asyncio
    async def test_create_on_append(self, ws, ctx):
        await tool_append_file({"path": "n.txt", "content": "first"}, ws, ctx)
        assert (ws / "n.txt").read_text() == "first"

    @pytest.mark.asyncio
    async def test_append(self, ws, ctx):
        (ws / "l.txt").write_text("a\n")
        await tool_append_file({"path": "l.txt", "content": "b\n"}, ws, ctx)
        assert (ws / "l.txt").read_text() == "a\nb\n"

    @pytest.mark.asyncio
    async def test_errors(self, ws, ctx):
        assert "error" in await tool_append_file({"path": "", "content": "x"}, ws, ctx)
        assert "error" in await tool_append_file({"path": "x", "content": None}, ws, ctx)
        assert "error" in await tool_append_file({"path": "/etc/passwd", "content": "x"}, ws, ctx)


# ─── read_file ───────────────────────────────────────────────────────────────

class TestReadFile:
    @pytest.mark.asyncio
    async def test_basic(self, ws, ctx):
        (ws / "t.txt").write_text("l1\nl2\nl3\n")
        r = await tool_read_file_advanced({"path": "t.txt"}, ws, ctx)
        assert r["read_from"] == 1 and "l1" in r["content"]
        assert r["total_lines"] == 4 and "total_chars" in r and r["has_more"] is False

    @pytest.mark.asyncio
    async def test_offset_limit(self, ws, ctx):
        (ws / "t.txt").write_text("\n".join(str(i) for i in range(1, 21)))
        r = await tool_read_file_advanced({"path": "t.txt", "offset": 5, "limit": 3}, ws, ctx)
        assert r["read_from"] == 5 and r["read_to"] == 7
        assert r["content"] == "5\n6\n7" and r["has_more"] is True
        assert "offset=8" in r["note"]

    @pytest.mark.asyncio
    async def test_char_budget(self, ws):
        ws_ctx = ToolContext(shared_dir=ws / "shared", workspace_root=ws, file_read_max_chars=10)
        (ws / "big.txt").write_text("\n".join(f"line{i}" for i in range(50)))
        r = await tool_read_file_advanced({"path": "big.txt"}, ws, ws_ctx)
        assert r["has_more"] is True and r["read_to"] < r["total_lines"]
        assert "note" in r

    @pytest.mark.asyncio
    async def test_errors(self, ws, ctx):
        assert "error" in await tool_read_file_advanced({"path": ""}, ws, ctx)
        assert "error" in await tool_read_file_advanced({"path": "nope.txt"}, ws, ctx)
        assert "error" in await tool_read_file_advanced({"path": "/etc/passwd"}, ws, ctx)
        (ws / "d").mkdir()
        assert "error" in await tool_read_file_advanced({"path": "d"}, ws, ctx)


# ─── replace_string ──────────────────────────────────────────────────────────

class TestReplaceString:
    @pytest.mark.asyncio
    async def test_exact(self, ws, ctx):
        (ws / "f.txt").write_text("hello world")
        await tool_replace_string({"path": "f.txt", "old_string": "world", "new_string": "there"}, ws, ctx)
        assert (ws / "f.txt").read_text() == "hello there"

    @pytest.mark.asyncio
    async def test_ambiguous(self, ws, ctx):
        (ws / "f.txt").write_text("x x")
        r = await tool_replace_string({"path": "f.txt", "old_string": "x", "new_string": "y"}, ws, ctx)
        assert "error" in r

    @pytest.mark.asyncio
    async def test_not_found_and_sandbox(self, ws, ctx):
        (ws / "f.txt").write_text("abc")
        assert "error" in await tool_replace_string({"path": "f.txt", "old_string": "zzz", "new_string": "y"}, ws, ctx)
        assert "error" in await tool_replace_string({"path": "/etc/passwd", "old_string": "a", "new_string": "b"}, ws, ctx)
        assert "error" in await tool_replace_string({"path": "f.txt", "old_string": "a", "new_string": None}, ws, ctx)


# ─── multi_replace ───────────────────────────────────────────────────────────

class TestMultiReplace:
    @pytest.mark.asyncio
    async def test_atomic_success(self, ws, ctx):
        (ws / "f.txt").write_text("a b c")
        r = await tool_multi_replace_string({"path": "f.txt", "replacements": [
            {"old_string": "a", "new_string": "1"},
            {"old_string": "c", "new_string": "3"},
        ]}, ws, ctx)
        assert r["replaced_count"] == 2 and (ws / "f.txt").read_text() == "1 b 3"

    @pytest.mark.asyncio
    async def test_atomicity_on_failure(self, ws, ctx):
        (ws / "f.txt").write_text("a b c")
        r = await tool_multi_replace_string({"path": "f.txt", "replacements": [
            {"old_string": "a", "new_string": "1"},
            {"old_string": "zzz", "new_string": "9"},
        ]}, ws, ctx)
        assert "error" in r and (ws / "f.txt").read_text() == "a b c"  # unchanged

    @pytest.mark.asyncio
    async def test_errors(self, ws, ctx):
        (ws / "f.txt").write_text("x")
        assert "error" in await tool_multi_replace_string({"path": "f.txt", "replacements": []}, ws, ctx)
        assert "error" in await tool_multi_replace_string({"path": "", "replacements": [{"old_string": "x", "new_string": "y"}]}, ws, ctx)


# ─── find_and_replace (matcher tiers) ────────────────────────────────────────

class TestFindAndReplace:
    def test_count_occurrences(self):
        assert count_occurrences("aaa", "a") == 3
        assert count_occurrences("abc", "") == 0

    def test_tier1_exact(self):
        assert find_and_replace("hi there", "there", "world")["tier"] == 1

    def test_tier2_trimmed(self):
        assert find_and_replace("hello world", "  hello  ", "bye")["tier"] == 2

    def test_tier3_indent_agnostic(self):
        content = "    def foo():\n        pass\n"
        r = find_and_replace(content, "def foo():\n    pass", "def foo():\n    return 1")
        assert r["tier"] == 3

    def test_tier4_normalized(self):
        r = find_and_replace("hello   world\nfoo\n", "hello world\nfoo", "x")
        assert r["tier"] == 4

    def test_ambiguous_and_missing(self):
        assert not find_and_replace("a a", "a", "b")["success"]
        assert not find_and_replace("abc", "zzz", "y")["success"]

    def test_helper_no_match(self):
        assert _try_indent_agnostic_match("a\nb\n", "x\ny", "n") is None
        assert _try_normalized_match("a b\n", "x y z", "n") is None


# ─── grep + helpers ──────────────────────────────────────────────────────────

class TestCodeSearchHelpers:
    def test_escape_regex(self):
        assert escape_regex("a.b*c") == r"a\.b\*c"

    def test_is_binary(self):
        assert is_binary_file("x.png") and not is_binary_file("x.py")
        assert ".png" in BINARY_EXTENSIONS and ".git" in IGNORE_DIRS

    def test_glob_to_regex(self):
        assert _glob_to_simple_regex("*.py").search("main.py")
        assert not _glob_to_simple_regex("*.py").search("main.js")


class TestGrep:
    @pytest.mark.asyncio
    async def test_basic(self, ws, ctx):
        (ws / "code.py").write_text("def hello():\n    return 42\n")
        r = await tool_grep_search({"query": "hello"}, ws, ctx)
        assert r["count"] >= 1 and any("hello" in m["content"] for m in r["matches"])

    @pytest.mark.asyncio
    async def test_include_pattern(self, ws, ctx):
        (ws / "a.py").write_text("target\n")
        (ws / "a.js").write_text("target\n")
        r = await tool_grep_search({"query": "target", "include_pattern": "*.py"}, ws, ctx)
        assert all(m["file"].endswith(".py") for m in r["matches"])

    @pytest.mark.asyncio
    async def test_skips_binary_and_ignore_dirs(self, ws, ctx):
        (ws / ".hidden").mkdir()
        (ws / ".hidden" / "s.txt").write_text("findme\n")
        (ws / "v.txt").write_text("findme\n")
        r = _python_grep("findme", ws, False, False, None, 100)
        assert r["count"] == 1 and r["matches"][0]["file"] == "v.txt"

    @pytest.mark.asyncio
    async def test_python_grep_max_results(self, ws):
        (ws / "big.txt").write_text("match\n" * 200)
        r = _python_grep("match", ws, False, False, None, 3)
        assert r["count"] == 3 and r["truncated"]

    @pytest.mark.asyncio
    async def test_subprocess_grep_no_tool(self, ws):
        (ws / "f.txt").write_text("hello\n")
        with patch("shutil.which", return_value=None):
            assert await _try_subprocess_grep("hello", ws, False, False, None, 100) is None

    @pytest.mark.asyncio
    async def test_subprocess_grep_timeout(self, ws):
        (ws / "f.txt").write_text("x\n")
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()), \
             patch("shutil.which", return_value="/usr/bin/grep"):
            assert await _try_subprocess_grep("x", ws, False, False, None, 100) is None
