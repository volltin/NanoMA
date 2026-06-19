"""Tests for nanoma/prompts.py — prompt construction is pure, so assert on text."""

from pathlib import Path

from nanoma.models import ModelRegistry, FusionDef, ModelDef
from nanoma import prompts


def _reg():
    r = ModelRegistry()
    r.models["concrete-x"] = ModelDef("concrete-x", "p", 8000, 1, 0.1, 2, "mid")
    r.models["concrete-y"] = ModelDef("concrete-y", "p", 8000, 1, 0.1, 2, "mid")
    r.aliases["fast"] = "concrete-x"
    r.fusions["fuse"] = FusionDef("fuse", panel=["fast", "concrete-y"], judge="concrete-x")
    r.fusions["nested"] = FusionDef("nested", panel=["fuse"], judge="fuse")
    return r


class TestConcreteModel:
    def test_alias_resolves(self):
        assert prompts.concrete_model(_reg(), "fast") == "concrete-x"

    def test_plain_model(self):
        assert prompts.concrete_model(_reg(), "concrete-y") == "concrete-y"

    def test_fusion_descends_to_judge(self):
        assert prompts.concrete_model(_reg(), "fuse") == "concrete-x"

    def test_nested_fusion_collapses(self):
        # nested → judge "fuse" (a fusion) → its judge "concrete-x"
        assert prompts.concrete_model(_reg(), "nested") == "concrete-x"


class TestModelsSection:
    def test_empty_registry_returns_blank(self):
        assert prompts.build_models_section(ModelRegistry()) == ""

    def test_lists_aliases_and_fusions(self):
        s = prompts.build_models_section(_reg())
        assert "Aliases:" in s and "fast → concrete-x" in s
        assert "Fusion models" in s and "fuse" in s and "panel:" in s


class TestFusionDirective:
    def test_contains_panel_and_judge(self):
        d = prompts.build_fusion_directive(_reg(), _reg().fusions["fuse"])
        assert "Model Fusion — REQUIRED" in d
        assert 'model="concrete-x"' in d and 'model="concrete-y"' in d
        assert "JUDGE" in d and 'running model `concrete-x`' in d
        assert 'wait(mode="all")' in d


class TestSystemPrompt:
    def test_root_prompt(self):
        s = prompts.build_system_prompt(
            _reg(), agent_id="alpha", task="do X", workspace=Path("/ws/alpha"),
            shared_dir=Path("/ws/shared"), time_limit=0, parent_context=None,
        )
        assert 'You are agent "alpha"' in s
        assert "Task: do X" in s and "/ws/shared" in s
        assert "## Your Context" not in s        # root has no spawner
        assert "Time limit" not in s
        assert "## Models" in s                   # registry has aliases/fusions
        # Finishing protocol must reach every agent (root included).
        assert "## How to act" in s and 'set_status(status="done"' in s

    def test_subagent_prompt_has_context_and_time(self):
        s = prompts.build_system_prompt(
            _reg(), agent_id="bravo", task="sub", workspace=Path("/ws/bravo"),
            shared_dir=Path("/ws/shared"), time_limit=120,
            parent_context={"parent_id": "alpha", "parent_task": "root", "siblings": "charlie(x)", "depth": 1},
        )
        assert "## Your Context" in s and "alpha" in s and "charlie(x)" in s
        assert "Time limit: 120s" in s

    def test_none_registry_no_models_section(self):
        s = prompts.build_system_prompt(
            None, agent_id="a", task="t", workspace=Path("/w"),
            shared_dir=Path("/s"), time_limit=0,
        )
        assert "## Models" not in s
