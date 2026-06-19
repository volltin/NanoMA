"""Extra grep coverage: regex/case/include branches + subprocess success path."""

import pytest

from nanoma.core import ToolContext
from nanoma.tools.grep import tool_grep_search, _python_grep, _try_subprocess_grep


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "shared").mkdir()
    return tmp_path


@pytest.fixture
def ctx(ws):
    return ToolContext(shared_dir=ws / "shared", workspace_root=ws, grep_max_results=50)


class TestPythonGrep:
    def test_regex(self, ws):
        (ws / "a.txt").write_text("foo123\nbar\nfoo456\n")
        r = _python_grep(r"foo\d+", ws, True, False, None, 100)
        assert r["count"] == 2 and all("foo" in m["content"] for m in r["matches"])

    def test_case_sensitive_vs_insensitive(self, ws):
        (ws / "a.txt").write_text("Hello\nhello\nHELLO\n")
        assert _python_grep("hello", ws, False, True, None, 100)["count"] == 1   # only exact case
        assert _python_grep("hello", ws, False, False, None, 100)["count"] == 3  # all

    def test_include_pattern(self, ws):
        (ws / "a.py").write_text("match\n")
        (ws / "a.js").write_text("match\n")
        r = _python_grep("match", ws, False, False, "*.py", 100)
        assert r["count"] == 1 and r["matches"][0]["file"].endswith(".py")

    def test_match_positions(self, ws):
        (ws / "a.txt").write_text("xx target yy\n")
        m = _python_grep("target", ws, False, False, None, 100)["matches"][0]
        assert m["line"] == 1 and m["match_start"] == 3   # 0-based offset of the match


class TestSubprocessGrep:
    @pytest.mark.asyncio
    async def test_subprocess_success_returns_matches(self, ws):
        # rg or grep is present in this environment → real subprocess path exercised.
        (ws / "code.txt").write_text("alpha\nbeta findme\ngamma\n")
        r = await _try_subprocess_grep("findme", ws, False, False, None, 100)
        assert r is not None and r["count"] >= 1
        assert any("findme" in m["content"] for m in r["matches"])

    @pytest.mark.asyncio
    async def test_tool_dispatch_finds_match(self, ws, ctx):
        (ws / "code.txt").write_text("needle in haystack\n")
        r = await tool_grep_search({"query": "needle"}, ws, ctx)
        assert r["count"] >= 1 and any("needle" in m["content"] for m in r["matches"])

    @pytest.mark.asyncio
    async def test_tool_regex_via_dispatch(self, ws, ctx):
        (ws / "code.txt").write_text("v1.2.3\nv9.9.9\nplain\n")
        r = await tool_grep_search({"query": r"v\d+\.\d+\.\d+", "is_regexp": True}, ws, ctx)
        assert r["count"] == 2
