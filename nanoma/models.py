"""Model registry: model definitions, aliases, and fusion specs (loaded from YAML)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger("nanoma")

# Default context limit for unknown models (128K is standard for modern LLMs)
_DEFAULT_CONTEXT_LIMIT = 128_000

# Max alias hops before we declare a cycle and bail out.
_MAX_ALIAS_HOPS = 10

# Fallback pricing — imported from cost.py for consistency
from nanoma.cost import _FALLBACK_PRICE_INPUT, _FALLBACK_PRICE_CACHED, _FALLBACK_PRICE_OUTPUT


@dataclass
class ModelDef:
    name: str
    provider: str
    context_limit: int
    price_input: float       # $/1M tokens (non-cached)
    price_cached: float      # $/1M tokens (cached)
    price_output: float      # $/1M tokens
    tier: str                # cheap, mid, strong


@dataclass
class FusionDef:
    """A fusion model: a panel of models each solve the task, a judge synthesizes.

    This is the Mixture-of-Agents / mix-of-models pattern. In NanoMA a fusion
    model is sugar for an orchestration: an agent runs on the `judge` model and spawns
    one real sub-agent per `panel` member, then synthesizes their answers. Panel/judge
    names may be aliases — they are resolved (to concrete models) at use time.
    """
    name: str
    panel: list[str]                 # models that form the panel (1-8); aliases allowed
    judge: str | None = None         # synthesizer model; None -> first panel member


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, ModelDef] = {}
        self.aliases: dict[str, str] = {}
        self.fusions: dict[str, FusionDef] = {}

    # ─── Loading ─────────────────────────────────────────────────────────

    def load(self, path: Path | str):
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        for name, cfg in (data.get("models") or {}).items():
            cfg = cfg or {}
            p = cfg.get("pricing", {}) or {}
            self.models[name] = ModelDef(
                name=name,
                provider=cfg.get("provider", "unknown"),
                context_limit=cfg.get("context_limit", _DEFAULT_CONTEXT_LIMIT),
                price_input=p.get("input", _FALLBACK_PRICE_INPUT),
                price_cached=p.get("cached_input", _FALLBACK_PRICE_CACHED),
                price_output=p.get("output", _FALLBACK_PRICE_OUTPUT),
                tier=cfg.get("tier", "mid"),
            )
            # Per-model inline aliases: `aliases: [short, other]`
            for a in cfg.get("aliases", []) or []:
                self.aliases[a] = name

        # Top-level alias map: `aliases: {short: target}`
        for alias, target in (data.get("aliases") or {}).items():
            self.aliases[alias] = target

        # Fusion specs
        for name, cfg in (data.get("fusion") or {}).items():
            cfg = cfg or {}
            panel = cfg.get("panel") or cfg.get("analysis_models") or []
            if not panel:
                logger.warning(f"Fusion '{name}' has no panel; skipping.")
                continue
            self.fusions[name] = FusionDef(
                name=name,
                panel=list(panel),
                judge=cfg.get("judge") or cfg.get("model"),
            )
            for a in cfg.get("aliases", []) or []:
                self.aliases[a] = name

    # ─── Resolution ──────────────────────────────────────────────────────

    def resolve(self, name: str) -> str:
        """Follow alias chains to a concrete model or fusion name.

        Aliases may point at models, fusions, or other aliases. Cycles are
        detected and broken (the last seen name is returned).
        """
        seen: set[str] = set()
        current = name
        for _ in range(_MAX_ALIAS_HOPS):
            if current in seen:
                logger.warning(f"Alias cycle detected resolving '{name}'; stopping at '{current}'.")
                break
            seen.add(current)
            nxt = self.aliases.get(current)
            if nxt is None or nxt == current:
                break
            current = nxt
        return current

    def is_fusion(self, name: str) -> bool:
        return self.resolve(name) in self.fusions

    def get_fusion(self, name: str) -> FusionDef | None:
        return self.fusions.get(self.resolve(name))

    # ─── Lookups (alias-aware) ────────────────────────────────────────────

    def get(self, name: str) -> ModelDef | None:
        return self.models.get(self.resolve(name))

    def pricing(self, name: str) -> tuple[float, float, float]:
        m = self.get(name)
        return (m.price_input, m.price_cached, m.price_output) if m else (
            _FALLBACK_PRICE_INPUT, _FALLBACK_PRICE_CACHED, _FALLBACK_PRICE_OUTPUT
        )

    def context_limit(self, name: str) -> int:
        resolved = self.resolve(name)
        # For a fusion model, the usable context is the smallest among its members.
        fusion = self.fusions.get(resolved)
        if fusion:
            limits = [self.context_limit(m) for m in fusion.panel]
            if fusion.judge:
                limits.append(self.context_limit(fusion.judge))
            return min(limits) if limits else _DEFAULT_CONTEXT_LIMIT
        m = self.models.get(resolved)
        return m.context_limit if m else _DEFAULT_CONTEXT_LIMIT

    # ─── Routing ──────────────────────────────────────────────────────────

    def route(self, budget: float, allowed: list[str] | None = None) -> str:
        """Pick strongest model that fits budget.

        Estimates cost for a typical agent run (~10 turns, ~3000 input + 1500 output tokens/turn)
        and picks the most capable model whose estimated total cost stays under 50% of budget
        (reserving the rest for sub-agents or additional turns).

        Routing only considers concrete models (not fusion specs) — fusion is opt-in
        and chosen explicitly via a model string.
        """
        allowed_resolved = [self.resolve(a) for a in allowed] if allowed else None
        candidates = [m for m in self.models.values() if not allowed_resolved or m.name in allowed_resolved]
        if not candidates:
            return next(iter(self.models), "gpt-4o-mini")
        candidates.sort(key=lambda m: m.price_output)

        # Estimate: 10 turns × (3000 input + 1500 output) tokens per turn
        est_turns = 10
        est_input_per_turn = 3000
        est_output_per_turn = 1500
        budget_fraction = 0.5  # only use half the budget (leave room for spawned agents)

        for model in reversed(candidates):  # try strongest (most expensive) first
            est_cost = est_turns * (
                est_input_per_turn * model.price_input +
                est_output_per_turn * model.price_output
            ) / 1_000_000
            if est_cost < budget * budget_fraction:
                return model.name
        # Fall back to cheapest
        return candidates[0].name


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        default = Path(__file__).parent.parent / "models.yaml"
        if default.exists():
            _registry.load(default)
    return _registry


def load_models(path: Path | str) -> ModelRegistry:
    global _registry
    _registry = ModelRegistry()
    _registry.load(path)
    return _registry
