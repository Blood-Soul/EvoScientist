"""Tests for the A/B harness in scripts/policy_ab.py.

The harness reports the number the design decision rests on, so its scorer needs
to be trustworthy in its own right: a matcher that fires on "PPO" inside
"support" would inflate every condition equally and hide the real effect.

Loaded by path because the harness is deliberately a standalone script, not an
installed module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "policy_ab.py"


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location("policy_ab", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["policy_ab"] = module
    spec.loader.exec_module(module)
    return module


ab = _load_harness()


RECORDS = [
    {
        "id": "l1_x_01",
        "bindings": [
            {"name": "ImageNet", "kind": "dataset"},
            {"name": "ResNet-50", "kind": "model"},
            {"name": "PPO", "kind": "toolchain"},
            {"name": "1e-5", "kind": "hyperparam"},
        ],
    }
]


def _task(**overrides: Any) -> Any:
    fields = {
        "name": "t",
        "description": "Train a ViT-Tiny classifier on CIFAR-10.",
        "setting": "One 4090.",
        "target_values": ["ViT-Tiny", "CIFAR-10"],
    }
    fields.update(overrides)
    return ab.Task(**fields)


class TestMentionMatching:
    """Word-boundary matching, the harness's one source of silent error."""

    def test_substring_does_not_match(self) -> None:
        assert ab._find_mention("our support engineers", "ppo") < 0
        assert ab._find_mention("the loss shifts", "sft") < 0

    def test_whole_token_matches(self) -> None:
        assert ab._find_mention("we run ppo against", "ppo") >= 0

    def test_punctuation_bounded_values_match(self) -> None:
        assert ab._find_mention("anneal to 1e-5 over", "1e-5") >= 0
        assert ab._find_mention("report ndcg@10 on", "ndcg@10") >= 0

    def test_hyphenated_name_matches(self) -> None:
        assert ab._find_mention("we use resnet-50 here", "resnet-50") >= 0


class TestScorer:
    def test_copied_value_is_stale(self) -> None:
        stale, rebound, _ = ab.score_output(
            output="1. Train ResNet-50 on ImageNet for 90 epochs.",
            task=_task(),
            records=RECORDS,
        )
        assert sorted(stale) == ["dataset:imagenet", "model:resnet-50"]
        assert rebound == []

    def test_provenance_mention_is_rebound_not_stale(self) -> None:
        """Naming a source value to explain where it came from is correct reuse."""
        stale, rebound, _ = ab.score_output(
            output=(
                "1. The schedule was validated on ImageNet; for our task use "
                "CIFAR-10 instead."
            ),
            task=_task(),
            records=RECORDS,
        )
        assert stale == []
        assert rebound == ["dataset:imagenet"]

    def test_value_named_by_the_task_is_never_stale(self) -> None:
        stale, _, _ = ab.score_output(
            output="1. Train on ImageNet.",
            task=_task(description="Pretrain on ImageNet.", target_values=[]),
            records=RECORDS,
        )
        assert stale == []

    def test_target_values_counted(self) -> None:
        _, _, hits = ab.score_output(
            output="1. Finetune ViT-Tiny on CIFAR-10.",
            task=_task(),
            records=RECORDS,
        )
        assert sorted(hits) == ["CIFAR-10", "ViT-Tiny"]

    def test_short_binding_names_ignored(self) -> None:
        """Names under three characters cannot be matched without false hits."""
        stale, _, _ = ab.score_output(
            output="we set k to 5 and use a 3B model",
            task=_task(),
            records=[{"bindings": [{"name": "k", "kind": "hyperparam"}]}],
        )
        assert stale == []


class TestFixture:
    """The shipped fixture must be able to measure the effect it claims to."""

    def test_shipped_fixture_loads(self) -> None:
        tasks, papers = ab.load_fixture(
            REPO_ROOT / "scripts" / "policy_ab_fixture.json"
        )
        assert len(tasks) >= 2
        assert len(papers) >= 2

    def test_fixture_without_bindings_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps(
                {
                    "tasks": [{"name": "t", "description": "d", "setting": "s"}],
                    "papers": [{"paper_id": "p", "l1": [{}], "l2": []}],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="no usable bindings"):
            ab.load_fixture(path)


class StubActor:
    """Returns a canned plan, so scoring is deterministic across conditions."""

    def __init__(self, plan: str) -> None:
        self.plan = plan
        self.prompts: list[str] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.prompts.append(str(messages[-1].content))
        return SimpleNamespace(content=self.plan)


class StubPolicyModel:
    """Stands in for the auxiliary model across rerank and synthesis."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.calls += 1
        text = str(messages[0].content)
        if "rerank" in text.casefold():
            return SimpleNamespace(
                content=json.dumps({"selected": [], "reason": "stub"})
            )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "verdict": "adapt",
                    "procedure": ["Anneal the learning rate smoothly to near zero."],
                    "rebind": [
                        {
                            "name": "peak learning rate",
                            "kind": "hyperparam",
                            "why_bound": "tuned for the source model size",
                            "how_to_obtain": "sweep on your own validation split",
                            "source_value": "1e-5",
                        }
                    ],
                    "sources": ["l1_x_01"],
                }
            )
        )


@pytest.mark.asyncio
async def test_end_to_end_with_stub_models(tmp_path: Path) -> None:
    """Seed → retrieve → derive → render → score → aggregate, no API calls."""
    tasks, papers = ab.load_fixture(REPO_ROOT / "scripts" / "policy_ab_fixture.json")
    records = [r for p in papers for lv in ("l1", "l2") for r in p[lv]]
    memory_dir = tmp_path / "memory"
    experience_ids = ab.seed_store(memory_dir, papers)
    assert len(experience_ids) == len(records)

    actor = StubActor("1. Train ResNet-50 on ImageNet with a cosine schedule.")
    policy_model = StubPolicyModel()
    samples = [
        await ab.run_cell(
            task=tasks[0],
            condition=condition,
            model=actor,
            memory_dir=memory_dir,
            records=records,
            raw_limit=4,
            max_selected=4,
            policy_model=policy_model,
        )
        for condition in ab.CONDITIONS
    ]

    assert [s.error for s in samples] == ["", "", ""]
    # The canned plan copies two source values, so every condition scores stale;
    # this asserts the metric fires, not that the layer works.
    for sample in samples:
        assert sorted(sample.stale) == ["dataset:imagenet", "model:resnet-50"]

    none_prompt, raw_prompt, policy_prompt = actor.prompts
    assert "prior experience" not in none_prompt
    assert "prior experience" in raw_prompt
    assert "reuse policy" in policy_prompt
    assert "do not copy the source value" in policy_prompt
    # Retrieval ran for the raw condition too, so B and C differ only in framing.
    assert "Applicable when:" in raw_prompt
    assert policy_model.calls >= 2

    summary = ab.summarize(samples)
    assert summary["policy"]["stale_binding_rate"] == 1.0
    assert summary["policy"]["stale_per_plan"] == 2.0
    assert summary["none"]["n"] == 1


@pytest.mark.asyncio
async def test_failed_cell_is_recorded_not_raised(tmp_path: Path) -> None:
    """One failing condition must not lose the other conditions' samples."""

    class Boom:
        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            raise RuntimeError("model unavailable")

    sample = await ab.run_cell(
        task=_task(),
        condition="none",
        model=Boom(),
        memory_dir=tmp_path,
        records=RECORDS,
        raw_limit=4,
        max_selected=4,
    )
    assert "model unavailable" in sample.error
    assert ab.summarize([sample])["none"] == {"n": 0, "errors": 1}


def test_unset_actor_model_resolves_to_the_project_not_a_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--model` unset means "this project's main model", as the flag documents.

    `get_chat_model(model=None)` does not read the project config -- it falls
    back to its own default, which is an Anthropic model. Passing the flag
    straight through therefore made an unset `--model` mean "Anthropic"
    regardless of configuration, and every cell failed on a missing
    `ANTHROPIC_API_KEY` for a model nobody selected.
    """
    import EvoScientist.config.settings as settings

    monkeypatch.setattr(
        settings,
        "get_effective_config",
        lambda: SimpleNamespace(model="deepseek-v4-pro", provider="custom-openai"),
    )
    monkeypatch.setattr(settings, "apply_config_to_env", lambda cfg: None)

    assert ab._resolve_actor_model(None, None) == ("deepseek-v4-pro", "custom-openai")


def test_an_explicit_actor_flag_overrides_the_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import EvoScientist.config.settings as settings

    monkeypatch.setattr(
        settings,
        "get_effective_config",
        lambda: SimpleNamespace(model="deepseek-v4-pro", provider="custom-openai"),
    )
    monkeypatch.setattr(settings, "apply_config_to_env", lambda cfg: None)

    assert ab._resolve_actor_model("gpt-4o", "openai") == ("gpt-4o", "openai")
    # A flag may set only the model; the provider still comes from config.
    assert ab._resolve_actor_model("gpt-4o", None) == ("gpt-4o", "custom-openai")


def test_resolution_publishes_config_credentials_to_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standalone script bypasses the agent's own config-to-env step.

    Credentials that live in the config file rather than the shell are only
    visible to `get_chat_model` once `apply_config_to_env` has run, so resolving
    the model has to trigger it or the harness fails with a missing key.
    """
    import EvoScientist.config.settings as settings

    applied: list[Any] = []
    monkeypatch.setattr(
        settings,
        "get_effective_config",
        lambda: SimpleNamespace(model="m", provider="p"),
    )
    monkeypatch.setattr(settings, "apply_config_to_env", applied.append)

    ab._resolve_actor_model(None, None)
    assert len(applied) == 1


def test_unloadable_config_falls_back_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config is environment-dependent; failing to read it must not be fatal."""
    import EvoScientist.config.settings as settings

    def _boom() -> Any:
        raise RuntimeError("no config file")

    monkeypatch.setattr(settings, "get_effective_config", _boom)

    assert ab._resolve_actor_model(None, None) == (None, None)
    assert ab._resolve_actor_model("gpt-4o", "openai") == ("gpt-4o", "openai")
