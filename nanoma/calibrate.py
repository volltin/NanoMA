"""Model calibration — a cheap preflight that probes every model a run will use.

The point: before spending a real budget on a multi-agent run, send each model a
tiny request **in the exact shape the runtime uses** (system + user messages, the
real tool schemas, no `temperature` unless configured). This surfaces "this model
is unusable" problems (auth, model-not-found, message-shape, tool-schema rejection,
deprecated params) up front instead of mid-run.

Used automatically by ``nanoma-run`` (disable with ``--no-calibrate`` / front-matter
``calibrate: false``), and importable directly:

    from nanoma.calibrate import calibrate, models_for_run
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from nanoma.llm import RetryConfig, openai_compatible_call


def _all_tool_schemas() -> list[dict]:
    """The exact tool set agents send — so calibration catches tool-schema rejection."""
    from nanoma.tools import build_all_tools
    return [t["schema"] for t in build_all_tools().values()]


def models_for_run(registry, model: str) -> list[str]:
    """Concrete models that a run with ``model`` will actually call.

    - plain model/alias -> [concrete]
    - fusion model      -> [each concrete panel member, concrete judge]
    Order preserved, de-duplicated.
    """
    def concrete(name: str) -> str:
        r = registry.resolve(name)
        fd = registry.fusions.get(r)
        if fd:  # nested fusion -> collapse to its judge/first panel member
            return concrete(fd.judge or (fd.panel[0] if fd.panel else r))
        return r

    resolved = registry.resolve(model)
    fd = registry.fusions.get(resolved)
    out: list[str] = []
    if fd:
        out += [concrete(m) for m in fd.panel]
        out.append(concrete(fd.judge or (fd.panel[0] if fd.panel else resolved)))
    else:
        out.append(resolved)
    return list(dict.fromkeys(out))


async def probe_model(
    model: str,
    *,
    llm_call: Callable[..., Awaitable[Any]] | None = None,
    temperature: float | None = None,
    with_tools: bool = True,
    max_tokens: int = 64,
    retry_config: RetryConfig | None = None,
) -> dict[str, Any]:
    """Send one tiny real request to ``model``; return {model, ok, ms, ...}."""
    llm_call = llm_call or openai_compatible_call
    messages = [
        {"role": "system", "content": "Connectivity probe. Reply tersely."},
        {"role": "user", "content": "Reply with the single word: OK"},
    ]
    tools = _all_tool_schemas() if with_tools else None
    rc = retry_config or RetryConfig(max_retries=1, base_delay=0.5)
    t0 = time.time()
    try:
        resp = await llm_call(messages, model, tools, temperature=temperature,
                              max_tokens=max_tokens, retry_config=rc)
        ms = int((time.time() - t0) * 1000)
        usable = bool(resp.content) or bool(resp.tool_calls)
        return {"model": model, "ok": usable, "ms": ms,
                "tokens": resp.usage.total_tokens,
                "note": "" if usable else "empty response (no content or tool_calls)"}
    except Exception as e:
        return {"model": model, "ok": False, "ms": int((time.time() - t0) * 1000),
                "error": str(e)[:300]}


async def calibrate(
    models: list[str],
    *,
    llm_call: Callable[..., Awaitable[Any]] | None = None,
    temperature: float | None = None,
    with_tools: bool = True,
    retry_config: RetryConfig | None = None,
) -> list[dict[str, Any]]:
    """Probe a list of concrete models in parallel. Returns one result dict each."""
    uniq = list(dict.fromkeys(models))
    return await asyncio.gather(*(
        probe_model(m, llm_call=llm_call, temperature=temperature,
                    with_tools=with_tools, retry_config=retry_config)
        for m in uniq
    ))


def format_results(results: list[dict[str, Any]]) -> str:
    lines = []
    for r in results:
        if r.get("ok"):
            extra = f"tokens={r.get('tokens','?')}" + (f"  ⚠ {r['note']}" if r.get("note") else "")
            lines.append(f"  OK    {r['model']:<34} {r['ms']:>5}ms  {extra}")
        else:
            lines.append(f"  FAIL  {r['model']:<34} {r['ms']:>5}ms  error: {r.get('error','')}")
    return "\n".join(lines)


def all_ok(results: list[dict[str, Any]]) -> bool:
    return all(r.get("ok") for r in results)
