"""Tests for nanoma/calibrate.py — model preflight probing (mock LLM)."""

import pytest

from nanoma.calibrate import (
    models_for_run, probe_model, calibrate, format_results, all_ok,
)
from nanoma.models import ModelRegistry, ModelDef, FusionDef
from nanoma.llm import LLMResponse, ToolCall
from nanoma.cost import UsageRecord


def _reg():
    r = ModelRegistry()
    for n in ("x", "y", "j"):
        r.models[n] = ModelDef(n, "p", 8000, 1, 0.1, 2, "mid")
    r.aliases["ax"] = "x"
    r.fusions["fuse"] = FusionDef("fuse", panel=["ax", "y"], judge="j")
    r.fusions["nested"] = FusionDef("nested", panel=["fuse"], judge="fuse")
    return r


# ─── models_for_run ──────────────────────────────────────────────────────────

class TestModelsForRun:
    def test_plain_model(self):
        assert models_for_run(_reg(), "x") == ["x"]

    def test_alias(self):
        assert models_for_run(_reg(), "ax") == ["x"]

    def test_fusion_expands_to_panel_plus_judge(self):
        assert models_for_run(_reg(), "fuse") == ["x", "y", "j"]

    def test_nested_fusion_collapses(self):
        # nested panel member "fuse" → its judge "j"; judge "fuse" → "j"; deduped
        assert models_for_run(_reg(), "nested") == ["j"]


# ─── probe_model ─────────────────────────────────────────────────────────────

class TestProbeModel:
    @pytest.mark.asyncio
    async def test_ok_with_content(self):
        async def llm(messages, model, tools=None, **kw):
            return LLMResponse(content="OK", usage=UsageRecord(input_tokens=3, output_tokens=1, model=model))
        r = await probe_model("m", llm_call=llm)
        assert r["ok"] is True and r["model"] == "m" and r["tokens"] == 4 and "ms" in r

    @pytest.mark.asyncio
    async def test_ok_with_tool_calls_only(self):
        async def llm(messages, model, tools=None, **kw):
            return LLMResponse(tool_calls=[ToolCall("t", "f", {})], usage=UsageRecord(model=model))
        assert (await probe_model("m", llm_call=llm))["ok"] is True

    @pytest.mark.asyncio
    async def test_empty_response_not_ok(self):
        async def llm(messages, model, tools=None, **kw):
            return LLMResponse(content=None, usage=UsageRecord(model=model))
        r = await probe_model("m", llm_call=llm)
        assert r["ok"] is False and "empty response" in r["note"]

    @pytest.mark.asyncio
    async def test_exception_becomes_error(self):
        async def llm(messages, model, tools=None, **kw):
            raise RuntimeError("model not found")
        r = await probe_model("m", llm_call=llm)
        assert r["ok"] is False and "model not found" in r["error"]

    @pytest.mark.asyncio
    async def test_passes_real_shape(self):
        seen = {}
        async def llm(messages, model, tools=None, **kw):
            seen["messages"] = messages
            seen["has_tools"] = bool(tools)
            return LLMResponse(content="OK", usage=UsageRecord(model=model))
        await probe_model("m", llm_call=llm, with_tools=True)
        assert seen["messages"][0]["role"] == "system" and seen["messages"][1]["role"] == "user"
        assert seen["has_tools"] is True
        # with_tools=False sends no schemas
        await probe_model("m", llm_call=llm, with_tools=False)
        assert seen["has_tools"] is False


# ─── calibrate / helpers ─────────────────────────────────────────────────────

class TestCalibrate:
    @pytest.mark.asyncio
    async def test_parallel_and_dedup(self):
        seen = []
        async def llm(messages, model, tools=None, **kw):
            seen.append(model)
            return LLMResponse(content="OK", usage=UsageRecord(model=model))
        results = await calibrate(["a", "b", "a"], llm_call=llm)
        assert len(results) == 2 and sorted(seen) == ["a", "b"]   # deduped

    @pytest.mark.asyncio
    async def test_mixed_results_reporting(self):
        async def llm(messages, model, tools=None, **kw):
            if model == "bad":
                raise RuntimeError("boom")
            return LLMResponse(content="OK", usage=UsageRecord(model=model))
        results = await calibrate(["good", "bad"], llm_call=llm)
        assert not all_ok(results)
        text = format_results(results)
        assert "OK    good" in text and "FAIL  bad" in text and "boom" in text

    def test_all_ok_true(self):
        assert all_ok([{"ok": True}, {"ok": True}])
        assert not all_ok([{"ok": True}, {"ok": False}])
