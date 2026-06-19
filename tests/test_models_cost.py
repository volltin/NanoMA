"""Tests for nanoma/models.py and nanoma/cost.py."""

from nanoma.models import ModelRegistry, FusionDef, load_models
from nanoma.cost import UsageRecord, CostLedger, _FALLBACK_PRICE_OUTPUT


def reg_from(text, tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(text)
    r = ModelRegistry()
    r.load(p)
    return r


# ─── ModelRegistry ───────────────────────────────────────────────────────────

class TestRegistry:
    def test_load_models_aliases_fusion(self, tmp_path):
        r = reg_from(
            "models:\n"
            "  big:\n    context_limit: 200000\n    pricing: {input: 3, cached_input: 0.3, output: 15}\n"
            "    tier: strong\n    aliases: [B]\n"
            "  small: {context_limit: 8000}\n"
            "aliases:\n  fast: small\n  chain: fast\n"
            "fusion:\n  combo:\n    panel: [big, small]\n    judge: big\n    aliases: [cb]\n",
            tmp_path,
        )
        assert r.get("big").context_limit == 200000
        assert r.resolve("B") == "big"               # per-model inline alias
        assert r.resolve("chain") == "small"          # alias→alias chain
        assert r.resolve("cb") == "combo"             # fusion inline alias
        assert r.is_fusion("combo") and r.is_fusion("cb")
        assert not r.is_fusion("big")

    def test_resolve_cycle_breaks(self):
        r = ModelRegistry()
        r.aliases = {"a": "b", "b": "a"}
        # Must not infinite-loop; returns last hop.
        assert r.resolve("a") in ("a", "b")

    def test_pricing_and_fallback(self, tmp_path):
        r = reg_from("models:\n  m:\n    pricing: {input: 2, cached_input: 0.5, output: 6}\n", tmp_path)
        assert r.pricing("m") == (2, 0.5, 6)
        assert r.pricing("unknown")[2] == _FALLBACK_PRICE_OUTPUT   # fallback

    def test_context_limit_fusion_is_min(self, tmp_path):
        r = reg_from(
            "models:\n  a: {context_limit: 8000}\n  b: {context_limit: 32000}\n"
            "fusion:\n  f:\n    panel: [a, b]\n    judge: b\n",
            tmp_path,
        )
        assert r.context_limit("f") == 8000          # smallest member wins
        assert r.context_limit("missing") == 128000  # default

    def test_get_fusion(self, tmp_path):
        r = reg_from("models:\n  a: {}\nfusion:\n  f:\n    panel: [a]\n", tmp_path)
        fd = r.get_fusion("f")
        assert isinstance(fd, FusionDef) and fd.panel == ["a"]
        assert r.get_fusion("a") is None

    def test_fusion_without_panel_skipped(self, tmp_path):
        r = reg_from("fusion:\n  bad:\n    judge: x\n", tmp_path)
        assert "bad" not in r.fusions

    def test_route_picks_strongest_affordable(self, tmp_path):
        r = reg_from(
            "models:\n"
            "  cheap:\n    pricing: {input: 0.1, output: 0.2}\n"
            "  mid:\n    pricing: {input: 1, output: 3}\n"
            "  lux:\n    pricing: {input: 30, output: 60}\n",
            tmp_path,
        )
        assert r.route(100.0) == "lux"        # big budget → strongest
        assert r.route(0.0001) == "cheap"     # tiny budget → cheapest fallback

    def test_route_allowed_filter(self, tmp_path):
        r = reg_from(
            "models:\n  cheap:\n    pricing: {input: 0.1, output: 0.2}\n  lux:\n    pricing: {input: 30, output: 60}\n",
            tmp_path,
        )
        assert r.route(100.0, allowed=["cheap"]) == "cheap"

    def test_route_empty_registry(self):
        assert ModelRegistry().route(10.0) == "gpt-4o-mini"

    def test_load_models_sets_global(self, tmp_path):
        p = tmp_path / "m.yaml"
        p.write_text("models:\n  z: {context_limit: 1234}\n")
        r = load_models(p)
        from nanoma.models import get_registry
        assert get_registry() is r and r.context_limit("z") == 1234
        load_models(__import__("pathlib").Path(__file__).parent.parent / "models.yaml")


# ─── cost ────────────────────────────────────────────────────────────────────

class TestCost:
    def test_total_tokens(self):
        assert UsageRecord(input_tokens=100, output_tokens=50).total_tokens == 150

    def test_cost_uses_registry_pricing(self, tmp_path):
        load_models_yaml = tmp_path / "m.yaml"
        load_models_yaml.write_text("models:\n  priced:\n    pricing: {input: 10, cached_input: 1, output: 20}\n")
        load_models(load_models_yaml)
        # 1M non-cached in @ $10, 0 cached, 1M out @ $20 → $30
        u = UsageRecord(input_tokens=1_000_000, output_tokens=1_000_000, model="priced")
        assert round(u.cost_usd(), 6) == 30.0
        # cached portion is discounted
        u2 = UsageRecord(input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=0, model="priced")
        assert round(u2.cost_usd(), 6) == 1.0
        load_models(__import__("pathlib").Path(__file__).parent.parent / "models.yaml")

    def test_cost_fallback_for_unknown_model(self):
        u = UsageRecord(input_tokens=1_000_000, output_tokens=0, model="totally-unknown-xyz")
        assert u.cost_usd() > 0   # fallback pricing applied, no crash

    def test_ledger_record_and_summary(self):
        led = CostLedger(total_budget=5.0)
        assert led.remaining() == 5.0 and led.can_afford(5.0) and not led.can_afford(5.01)
        c = led.record("alpha", UsageRecord(input_tokens=1000, output_tokens=500, model="x"))
        assert c > 0 and led.per_agent["alpha"] == c and led.total_spent == c
        s = led.summary()
        assert s["budget"] == 5.0 and "alpha" in s["per_agent"]

    def test_cache_hit_rate_tracking(self):
        led = CostLedger()
        assert led.cache_hit_rate() == 0.0          # no usage yet
        led.record("a", UsageRecord(input_tokens=1000, cached_input_tokens=800, output_tokens=100, model="x"))
        led.record("a", UsageRecord(input_tokens=1000, cached_input_tokens=600, output_tokens=100, model="x"))
        assert led.total_input_tokens == 2000 and led.total_cached_input_tokens == 1400
        assert led.cache_hit_rate() == 0.7          # 1400/2000
        assert led.summary()["cache_hit_rate"] == 0.7

    def test_estimate_cost(self):
        led = CostLedger()
        assert led.estimate_cost("unknown", 1_000_000, 0) > 0
