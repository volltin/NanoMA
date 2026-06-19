"""LLM abstraction: OpenAI-compatible client with retry."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from nanoma.cost import UsageRecord

logger = logging.getLogger("nanoma")

Message = dict[str, Any]
ToolDef = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: UsageRecord = field(default_factory=UsageRecord)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    http_timeout: float = 180.0  # seconds; LLM calls can be slow with large contexts


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def count_message_tokens(messages: list[Message]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        if isinstance(content, str):
            total += estimate_tokens(content)
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                total += estimate_tokens(json.dumps(tc.get("function", {}).get("arguments", "")))
    return total


# --- LLM logging ---

_log_dir: Path | None = None
_log_counter: int = 0


def set_log_dir(path: Path | str):
    global _log_dir
    _log_dir = Path(path)
    _log_dir.mkdir(parents=True, exist_ok=True)


# --- Main call ---

_RETRYABLE = {429, 500, 502, 503, 504}
_shared_client: httpx.AsyncClient | None = None


def _get_client(timeout: float = 180.0) -> httpx.AsyncClient:
    """Reuse a single httpx client across all LLM calls (connection pooling)."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=100))
    return _shared_client


async def openai_compatible_call(
    messages: list[Message],
    model: str,
    tools: list[ToolDef] | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 16384,
    retry_config: RetryConfig | None = None,
) -> LLMResponse:
    base_url = base_url or os.environ.get("NANOMA_LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("NANOMA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No API key. Set NANOMA_API_KEY or OPENAI_API_KEY.")
    rc = retry_config or RetryConfig()

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body: dict[str, Any] = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
    }
    # Only send temperature when explicitly set — several modern models (e.g. Claude
    # Opus via Bedrock, OpenAI o-series) reject the parameter with a 400 error.
    if temperature is not None:
        body["temperature"] = temperature
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    t0 = time.time()
    client = _get_client()
    last_err: Exception | None = None
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=body)
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.3))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_err = e
            if attempt == rc.max_retries:
                raise
            await asyncio.sleep(rc.base_delay * (2 ** attempt))
    else:
        raise last_err or RuntimeError("Retry exhausted")

    elapsed_ms = (time.time() - t0) * 1000

    # Log
    if _log_dir:
        global _log_counter
        _log_counter += 1
        safe_model = model.replace("/", "_").replace(":", "_")
        log_path = _log_dir / f"{_log_counter:05d}_{safe_model}.jsonl"
        try:
            log_path.write_text(json.dumps({"request": body, "response": data, "ms": elapsed_ms}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM log {log_path}: {e}")

    choice = data["choices"][0]["message"]
    usage_data = data.get("usage", {})
    cached = 0
    if pd := usage_data.get("prompt_tokens_details"):
        cached = pd.get("cached_tokens", 0)

    usage = UsageRecord(
        input_tokens=usage_data.get("prompt_tokens", 0),
        cached_input_tokens=cached,
        output_tokens=usage_data.get("completion_tokens", 0),
        model=model,
    )

    tool_calls = []
    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            fn = tc["function"]
            try:
                args = json.loads(fn["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": fn["arguments"]}
            tool_calls.append(ToolCall(id=tc["id"], name=fn["name"], arguments=args))

    return LLMResponse(content=choice.get("content"), tool_calls=tool_calls, usage=usage, raw=data)


def default_router(task: str, budget: float, allowed_models: list[str] | None = None) -> str:
    from nanoma.models import get_registry
    return get_registry().route(budget, allowed=allowed_models)


# ─── Anthropic-native path (for prompt caching) ──────────────────────────────
#
# OpenAI-compatible proxies typically strip Anthropic `cache_control`, so Claude
# models get 0% prompt caching through /chat/completions (every turn re-bills the
# full growing prefix). The native /messages endpoint honors cache_control, cutting
# multi-turn cost dramatically. We convert the OpenAI-style request to Anthropic
# format and place cache breakpoints on the stable prefix (system + tools) and the
# latest message (to cache the conversation prefix incrementally).

_ANTHROPIC_HINTS = ("claude", "opus", "sonnet", "haiku")


def is_anthropic_model(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _ANTHROPIC_HINTS)


def _tools_to_anthropic(tools: list[ToolDef]) -> list[dict]:
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    if out:  # cache breakpoint at end of tools → caches the (system + tools) prefix
        out[-1]["cache_control"] = {"type": "ephemeral"}
    return out


def _messages_to_anthropic(messages: list[Message]) -> tuple[list[dict] | None, list[dict]]:
    """Convert OpenAI-style messages to (system_blocks, anthropic_messages).

    Coalesces consecutive same-role turns (Anthropic requires strict alternation and
    wants all tool results for a turn in a single user message).
    """
    system_text = None
    amsgs: list[dict] = []

    def push(role: str, blocks: list[dict]):
        if amsgs and amsgs[-1]["role"] == role:
            amsgs[-1]["content"].extend(blocks)
        else:
            amsgs.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            system_text = content if isinstance(content, str) else (system_text or "")
        elif role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""),
                           "content": content or ""}])
        elif role == "assistant":
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in m.get("tool_calls", []) or []:
                fn = tc["function"]
                try:
                    inp = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except (json.JSONDecodeError, TypeError):
                    inp = {}
                blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": inp})
            push("assistant", blocks or [{"type": "text", "text": ""}])
        else:  # user
            push("user", [{"type": "text", "text": content if isinstance(content, str) else str(content)}])

    system_blocks = None
    if system_text:
        system_blocks = [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
    # Incremental conversation caching: breakpoint on the final block of the last turn.
    if amsgs and isinstance(amsgs[-1]["content"], list) and amsgs[-1]["content"]:
        amsgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return system_blocks, amsgs


async def anthropic_messages_call(
    messages: list[Message],
    model: str,
    tools: list[ToolDef] | None = None,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 16384,
    retry_config: RetryConfig | None = None,
) -> LLMResponse:
    base_url = base_url or os.environ.get("NANOMA_LLM_BASE_URL", "https://api.anthropic.com/v1")
    api_key = api_key or os.environ.get("NANOMA_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        raise RuntimeError("No API key. Set NANOMA_API_KEY or ANTHROPIC_API_KEY.")
    rc = retry_config or RetryConfig()

    headers = {"content-type": "application/json", "x-api-key": api_key,
               "anthropic-version": "2023-06-01"}
    system_blocks, amsgs = _messages_to_anthropic(messages)
    body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": amsgs}
    if system_blocks:
        body["system"] = system_blocks
    if temperature is not None:
        body["temperature"] = temperature
    if tools:
        body["tools"] = _tools_to_anthropic(tools)

    t0 = time.time()
    client = _get_client()
    last_err: Exception | None = None
    for attempt in range(rc.max_retries + 1):
        try:
            resp = await client.post(f"{base_url}/messages", headers=headers, json=body)
            if resp.status_code in _RETRYABLE and attempt < rc.max_retries:
                delay = min(rc.base_delay * (2 ** attempt), rc.max_delay)
                await asyncio.sleep(delay + random.uniform(0, delay * 0.3))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_err = e
            if attempt == rc.max_retries:
                raise
            await asyncio.sleep(rc.base_delay * (2 ** attempt))
    else:
        raise last_err or RuntimeError("Retry exhausted")

    elapsed_ms = (time.time() - t0) * 1000
    if _log_dir:
        global _log_counter
        _log_counter += 1
        safe_model = model.replace("/", "_").replace(":", "_")
        log_path = _log_dir / f"{_log_counter:05d}_{safe_model}.jsonl"
        try:
            log_path.write_text(json.dumps({"request": body, "response": data, "ms": elapsed_ms}, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write LLM log {log_path}: {e}")

    text_parts, tool_calls = [], []
    for block in data.get("content", []) or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(ToolCall(id=block.get("id", ""), name=block.get("name", ""),
                                       arguments=block.get("input", {}) or {}))

    u = data.get("usage", {}) or {}
    cache_read = u.get("cache_read_input_tokens", 0)
    cache_create = u.get("cache_creation_input_tokens", 0)
    # Anthropic's input_tokens excludes cached/created prefix tokens; total prompt is
    # the sum. cached portion (billed ~0.1x) = cache_read.
    usage = UsageRecord(
        input_tokens=u.get("input_tokens", 0) + cache_read + cache_create,
        cached_input_tokens=cache_read,
        output_tokens=u.get("output_tokens", 0),
        model=model,
    )
    content = "".join(text_parts) or None
    return LLMResponse(content=content, tool_calls=tool_calls, usage=usage, raw=data)
