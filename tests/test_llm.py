"""Tests for nanoma/llm.py — token utils, router, and the HTTP client.

The client is exercised against an in-process httpx.MockTransport, so the real
request building, retry loop, response parsing, and logging are all covered
without network access.
"""

import httpx
import pytest

import nanoma.llm as llm
from nanoma.llm import (
    estimate_tokens, count_message_tokens, set_log_dir, default_router,
    openai_compatible_call, RetryConfig,
)
from nanoma.models import load_models


def _client_with(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok_body(content="OK", tool_calls=None, cached=0):
    msg = {"content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }


# ─── token utilities ─────────────────────────────────────────────────────────

class TestTokenUtils:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 1            # floor of 1
        assert estimate_tokens("a" * 400) == 100

    def test_count_message_tokens_content(self):
        msgs = [{"role": "system", "content": "x" * 40}, {"role": "user", "content": "y" * 40}]
        assert count_message_tokens(msgs) == 20

    def test_count_message_tokens_includes_tool_calls(self):
        base = [{"role": "user", "content": "hi"}]
        with_tc = base + [{"role": "assistant", "content": "",
                           "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1, "b": 2}'}}]}]
        assert count_message_tokens(with_tc) > count_message_tokens(base)

    def test_none_content_is_safe(self):
        # None content is coerced to "" → floor of 1 token, and must not raise.
        assert count_message_tokens([{"role": "assistant", "content": None}]) == 1


# ─── default_router ──────────────────────────────────────────────────────────

def test_default_router(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(
        "models:\n"
        "  cheap:\n    pricing: {input: 0.1, output: 0.2}\n    tier: cheap\n"
        "  pricey:\n    pricing: {input: 5.0, output: 15.0}\n    tier: strong\n"
    )
    load_models(cfg)
    # Big budget → strongest affordable; tiny budget → cheapest.
    assert default_router("t", 100.0) == "pricey"
    assert default_router("t", 0.001) == "cheap"
    load_models(__import__("pathlib").Path(__file__).parent.parent / "models.yaml")


# ─── openai_compatible_call ──────────────────────────────────────────────────

class TestCall:
    @pytest.mark.asyncio
    async def test_success_parses_content_and_usage(self, monkeypatch):
        def handler(req):
            body = httpx.Response(200, json=_ok_body(content="hello", cached=3))
            return body
        monkeypatch.setattr(llm, "_get_client", lambda timeout=180.0: _client_with(handler))
        r = await openai_compatible_call([{"role": "user", "content": "hi"}], "m", api_key="k")
        assert r.content == "hello"
        assert r.usage.input_tokens == 10 and r.usage.output_tokens == 5
        assert r.usage.cached_input_tokens == 3

    @pytest.mark.asyncio
    async def test_request_shape(self, monkeypatch):
        seen = {}
        def handler(req):
            import json
            seen.update(json.loads(req.content))
            return httpx.Response(200, json=_ok_body())
        monkeypatch.setattr(llm, "_get_client", lambda timeout=180.0: _client_with(handler))
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        await openai_compatible_call([{"role": "user", "content": "hi"}], "mymodel", tools,
                                     api_key="k", temperature=0.4, max_tokens=99)
        assert seen["model"] == "mymodel" and seen["max_tokens"] == 99
        assert seen["temperature"] == 0.4
        assert seen["tools"] == tools and seen["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_temperature_omitted_by_default(self, monkeypatch):
        seen = {}
        def handler(req):
            import json
            seen.update(json.loads(req.content))
            return httpx.Response(200, json=_ok_body())
        monkeypatch.setattr(llm, "_get_client", lambda timeout=180.0: _client_with(handler))
        await openai_compatible_call([{"role": "user", "content": "hi"}], "m", api_key="k")
        assert "temperature" not in seen

    @pytest.mark.asyncio
    async def test_tool_calls_parsed(self, monkeypatch):
        tcs = [{"id": "t1", "function": {"name": "do", "arguments": '{"x": 5}'}}]
        monkeypatch.setattr(llm, "_get_client",
                            lambda timeout=180.0: _client_with(lambda req: httpx.Response(200, json=_ok_body(None, tcs))))
        r = await openai_compatible_call([{"role": "user", "content": "hi"}], "m", api_key="k")
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "do" and r.tool_calls[0].arguments == {"x": 5}

    @pytest.mark.asyncio
    async def test_bad_tool_arguments_fall_back_to_raw(self, monkeypatch):
        tcs = [{"id": "t1", "function": {"name": "do", "arguments": "{not valid json"}}]
        monkeypatch.setattr(llm, "_get_client",
                            lambda timeout=180.0: _client_with(lambda req: httpx.Response(200, json=_ok_body(None, tcs))))
        r = await openai_compatible_call([{"role": "user", "content": "hi"}], "m", api_key="k")
        assert r.tool_calls[0].arguments == {"_raw": "{not valid json"}

    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(self, monkeypatch):
        calls = {"n": 0}
        def handler(req):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(500, json={"error": "server"})
            return httpx.Response(200, json=_ok_body(content="recovered"))
        monkeypatch.setattr(llm, "_get_client", lambda timeout=180.0: _client_with(handler))
        r = await openai_compatible_call([{"role": "user", "content": "hi"}], "m",
                                         api_key="k", retry_config=RetryConfig(max_retries=3, base_delay=0))
        assert r.content == "recovered" and calls["n"] == 2

    @pytest.mark.asyncio
    async def test_non_retryable_400_raises(self, monkeypatch):
        monkeypatch.setattr(llm, "_get_client",
                            lambda timeout=180.0: _client_with(lambda req: httpx.Response(400, json={"error": "bad"})))
        with pytest.raises(httpx.HTTPStatusError):
            await openai_compatible_call([{"role": "user", "content": "hi"}], "m",
                                         api_key="k", retry_config=RetryConfig(max_retries=1, base_delay=0))

    @pytest.mark.asyncio
    async def test_timeout_exhausts_and_raises(self, monkeypatch):
        def handler(req):
            raise httpx.TimeoutException("slow")
        monkeypatch.setattr(llm, "_get_client", lambda timeout=180.0: _client_with(handler))
        with pytest.raises(httpx.TimeoutException):
            await openai_compatible_call([{"role": "user", "content": "hi"}], "m",
                                         api_key="k", retry_config=RetryConfig(max_retries=1, base_delay=0))

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("NANOMA_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="No API key"):
            await openai_compatible_call([{"role": "user", "content": "hi"}], "m")

    @pytest.mark.asyncio
    async def test_logging_writes_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(llm, "_get_client",
                            lambda timeout=180.0: _client_with(lambda req: httpx.Response(200, json=_ok_body())))
        set_log_dir(tmp_path)
        try:
            await openai_compatible_call([{"role": "user", "content": "hi"}], "vendor/model:tag", api_key="k")
            logs = list(tmp_path.glob("*.jsonl"))
            assert len(logs) == 1
            text = logs[0].read_text()
            assert '"request"' in text and '"response"' in text
            assert "/" not in logs[0].name.replace(".jsonl", "")  # model slug sanitized
        finally:
            llm._log_dir = None


# ─── Anthropic-native path ───────────────────────────────────────────────────

from nanoma.llm import (
    is_anthropic_model, _messages_to_anthropic, _tools_to_anthropic, anthropic_messages_call,
)


class TestAnthropicConversion:
    def test_model_detection(self):
        assert is_anthropic_model("claude-opus-4-8") and is_anthropic_model("anthropic/claude-sonnet-4-6")
        assert not is_anthropic_model("gpt-4.1-mini") and not is_anthropic_model("deepseek-v4-pro")

    def test_tools_get_cache_breakpoint(self):
        tools = [{"type": "function", "function": {"name": "a", "description": "d", "parameters": {"type": "object"}}},
                 {"type": "function", "function": {"name": "b", "parameters": {}}}]
        out = _tools_to_anthropic(tools)
        assert out[0]["name"] == "a" and out[0]["input_schema"] == {"type": "object"}
        assert "cache_control" not in out[0]                 # only the last tool
        assert out[-1]["cache_control"] == {"type": "ephemeral"}

    def test_messages_conversion_and_coalescing(self):
        msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
                {"id": "t2", "type": "function", "function": {"name": "g", "arguments": '{"y": 2}'}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "r1"},
            {"role": "tool", "tool_call_id": "t2", "content": "r2"},
            {"role": "user", "content": "continue"},
        ]
        system, amsgs = _messages_to_anthropic(msgs)
        assert system[0]["text"] == "SYS" and system[0]["cache_control"] == {"type": "ephemeral"}
        roles = [m["role"] for m in amsgs]
        assert roles == ["user", "assistant", "user"]        # strict alternation
        # assistant turn has text + two tool_use blocks
        a = amsgs[1]["content"]
        assert a[0] == {"type": "text", "text": "ok"}
        assert [b["type"] for b in a[1:]] == ["tool_use", "tool_use"]
        assert a[1]["input"] == {"x": 1}
        # the two tool results + trailing user are coalesced into ONE user message
        last = amsgs[2]["content"]
        assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"]
        assert last[0]["tool_use_id"] == "t1"
        # incremental cache breakpoint on the final block
        assert last[-1]["cache_control"] == {"type": "ephemeral"}


class TestAnthropicCall:
    @pytest.mark.asyncio
    async def test_parses_content_tools_and_cache_usage(self, monkeypatch):
        def handler(req):
            import json as _j
            body = _j.loads(req.content)
            assert req.url.path.endswith("/messages")
            assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
            return httpx.Response(200, json={
                "content": [{"type": "text", "text": "hello"},
                            {"type": "tool_use", "id": "tu1", "name": "do", "input": {"a": 1}}],
                "usage": {"input_tokens": 12, "cache_creation_input_tokens": 100,
                          "cache_read_input_tokens": 900, "output_tokens": 7},
            })
        monkeypatch.setattr(llm, "_get_client", lambda timeout=180.0: _client_with(handler))
        r = await anthropic_messages_call(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}],
            "claude-opus-4-8", api_key="k")
        assert r.content == "hello"
        assert r.tool_calls[0].name == "do" and r.tool_calls[0].arguments == {"a": 1}
        # total prompt = input + cache_create + cache_read; cached = cache_read
        assert r.usage.input_tokens == 1012 and r.usage.cached_input_tokens == 900
        assert r.usage.output_tokens == 7
