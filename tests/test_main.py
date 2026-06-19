"""Tests for the `nanoma` CLI (nanoma/main.py) — runs offline with a mock LLM."""

import sys
import pytest

import nanoma.runtime as rtmod
from nanoma.main import cli
from nanoma.llm import LLMResponse, ToolCall
from nanoma.cost import UsageRecord


def _done_mock(result="cli-result"):
    async def f(messages, model, tools=None, **kw):
        return LLMResponse(tool_calls=[ToolCall("d", "set_status", {"status": "done", "result": result})],
                           usage=UsageRecord(model=model))
    return f


def test_cli_runs_and_prints_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rtmod, "openai_compatible_call", _done_mock("hello-cli"))
    monkeypatch.setattr(sys, "argv", [
        "nanoma", "do a thing",
        "--budget", "1", "--max-agents", "3",
        "--workspace", str(tmp_path / "ws"), "--log-dir", str(tmp_path / "logs"),
        "--model", "mini",
    ])
    cli()
    assert "hello-cli" in capsys.readouterr().out


def test_cli_loads_custom_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rtmod, "openai_compatible_call", _done_mock())
    reg = tmp_path / "m.yaml"
    reg.write_text("models:\n  custom-x: {context_limit: 4096}\naliases:\n  mini: custom-x\n")
    monkeypatch.setattr(sys, "argv", [
        "nanoma", "task", "--models", str(reg),
        "--workspace", str(tmp_path / "ws"), "--log-dir", str(tmp_path / "logs"),
    ])
    cli()
    from nanoma.models import get_registry
    assert get_registry().resolve("mini") == "custom-x"
    load_default = __import__("pathlib").Path(__file__).parent.parent / "models.yaml"
    from nanoma.models import load_models
    load_models(load_default)


def test_cli_keyboard_interrupt_exits_1(tmp_path, monkeypatch):
    monkeypatch.setattr(rtmod, "openai_compatible_call", _done_mock())
    monkeypatch.setattr(sys, "argv", [
        "nanoma", "task",
        "--workspace", str(tmp_path / "ws"), "--log-dir", str(tmp_path / "logs"),
    ])
    def boom(coro=None):
        if coro is not None and hasattr(coro, "close"):
            coro.close()   # avoid "coroutine was never awaited" warning
        raise KeyboardInterrupt
    monkeypatch.setattr("asyncio.run", boom)
    with pytest.raises(SystemExit) as ei:
        cli()
    assert ei.value.code == 1
