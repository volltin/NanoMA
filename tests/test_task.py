"""Tests for nanoma/task.py — the `nanoma-run` Markdown task runner."""

import pytest

import nanoma.runtime as rtmod
import nanoma.calibrate as calmod
from nanoma.task import parse_task_file, _resolve_task_path, run_task
from nanoma.llm import LLMResponse, ToolCall
from nanoma.cost import UsageRecord


def done_mock(result="task complete"):
    async def f(messages, model, tools=None, **kw):
        return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": result})],
                           usage=UsageRecord(input_tokens=10, output_tokens=5, model=model))
    return f


# ─── parse_task_file ─────────────────────────────────────────────────────────

class TestParse:
    def test_front_matter_and_body(self):
        cfg, body = parse_task_file("---\nmodel: mini\nbudget: 8\n---\n\nDo the thing.\n")
        assert cfg == {"model": "mini", "budget": 8} and body == "Do the thing."

    def test_no_front_matter(self):
        cfg, body = parse_task_file("Just a prompt, no config.")
        assert cfg == {} and body == "Just a prompt, no config."

    def test_unterminated_raises(self):
        with pytest.raises(ValueError, match="Unterminated"):
            parse_task_file("---\nmodel: mini\n\nbody with no closing fence")

    def test_non_mapping_raises(self):
        with pytest.raises(ValueError, match="mapping"):
            parse_task_file("---\n- a\n- b\n---\nbody")

    def test_empty_front_matter_is_empty_dict(self):
        cfg, body = parse_task_file("---\n---\nbody")
        assert cfg == {} and body == "body"


# ─── _resolve_task_path ──────────────────────────────────────────────────────

class TestResolvePath:
    def test_file(self, tmp_path):
        f = tmp_path / "t.md"
        f.write_text("x")
        assert _resolve_task_path(f) == f

    def test_folder_with_task_md(self, tmp_path):
        (tmp_path / "task.md").write_text("x")
        assert _resolve_task_path(tmp_path) == tmp_path / "task.md"

    def test_folder_without_task_md(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_task_path(tmp_path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_task_path(tmp_path / "nope.md")


# ─── run_task (end to end with mock LLM) ─────────────────────────────────────

class TestRunTask:
    @pytest.mark.asyncio
    async def test_runs_and_writes_result(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rtmod, "openai_compatible_call", done_mock("the answer"))
        folder = tmp_path / "mytask"
        folder.mkdir()
        (folder / "task.md").write_text("---\nbudget: 5\nmax_turns: 5\n---\nSolve it.")
        out = await run_task(folder, overrides={"calibrate": False})
        assert out["status"] == "done" and out["result"] == "the answer"
        result_md = folder / "result.md"
        assert result_md.exists()
        text = result_md.read_text()
        assert "## Final result" in text and "the answer" in text
        assert "Status: **done**" in text
        # self-contained folder layout
        assert (folder / "workspace").exists() and (folder / "logs").exists()

    @pytest.mark.asyncio
    async def test_cli_overrides_win_over_front_matter(self, tmp_path, monkeypatch):
        captured = {}
        def spy_mock():
            async def f(messages, model, tools=None, **kw):
                captured["model"] = model
                return await done_mock()(messages, model, tools, **kw)
            return f
        monkeypatch.setattr(rtmod, "openai_compatible_call", spy_mock())
        folder = tmp_path / "t"
        folder.mkdir()
        (folder / "task.md").write_text("---\nmodel: from-file\nbudget: 5\n---\nGo.")
        await run_task(folder, overrides={"calibrate": False, "model": "from-cli"})
        assert captured["model"] == "from-cli"

    @pytest.mark.asyncio
    async def test_empty_body_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rtmod, "openai_compatible_call", done_mock())
        folder = tmp_path / "t"
        folder.mkdir()
        (folder / "task.md").write_text("---\nbudget: 5\n---\n")
        with pytest.raises(ValueError, match="no body"):
            await run_task(folder, overrides={"calibrate": False})

    @pytest.mark.asyncio
    async def test_calibrate_only_probes_and_exits(self, tmp_path, monkeypatch):
        async def probe_ok(messages, model, tools=None, **kw):
            return LLMResponse(content="OK", usage=UsageRecord(model=model))
        monkeypatch.setattr(calmod, "openai_compatible_call", probe_ok)
        folder = tmp_path / "t"
        folder.mkdir()
        (folder / "task.md").write_text("---\nmodel: mini\nbudget: 5\n---\nGo.")
        out = await run_task(folder, overrides={"calibrate_only": True})
        assert out["status"] == "calibrated" and out["calibration"]
        assert all(r["ok"] for r in out["calibration"])
        assert not (folder / "result.md").exists()   # no run happened

    @pytest.mark.asyncio
    async def test_calibration_failure_aborts(self, tmp_path, monkeypatch):
        async def probe_fail(messages, model, tools=None, **kw):
            raise RuntimeError("model unavailable")
        monkeypatch.setattr(calmod, "openai_compatible_call", probe_fail)
        folder = tmp_path / "t"
        folder.mkdir()
        (folder / "task.md").write_text("---\nmodel: mini\nbudget: 5\n---\nGo.")
        with pytest.raises(SystemExit, match="ABORT"):
            await run_task(folder, overrides={"calibrate": True})


# ─── cli() ───────────────────────────────────────────────────────────────────

class TestCli:
    def test_cli_prints_result(self, tmp_path, monkeypatch, capsys):
        import sys
        import nanoma.task as taskmod
        (tmp_path / "task.md").write_text("---\nbudget: 1\n---\nGo.")

        async def fake_run_task(path, overrides):
            assert overrides["calibrate"] is False        # --no-calibrate propagated
            return {"result": "CLI_RESULT"}
        monkeypatch.setattr(taskmod, "run_task", fake_run_task)
        monkeypatch.setattr(sys, "argv", ["nanoma-run", str(tmp_path), "--no-calibrate", "--budget", "2"])
        taskmod.cli()
        assert "CLI_RESULT" in capsys.readouterr().out

    def test_cli_keyboard_interrupt_exits_1(self, tmp_path, monkeypatch):
        import sys
        import nanoma.task as taskmod
        (tmp_path / "task.md").write_text("body")
        monkeypatch.setattr(sys, "argv", ["nanoma-run", str(tmp_path)])

        def boom(coro=None):
            if coro is not None and hasattr(coro, "close"):
                coro.close()
            raise KeyboardInterrupt
        monkeypatch.setattr("asyncio.run", boom)
        with pytest.raises(SystemExit) as ei:
            taskmod.cli()
        assert ei.value.code == 1
