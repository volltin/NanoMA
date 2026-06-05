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
    temperature: float = 0.7,
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
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }
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
