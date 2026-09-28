"""Tests for the push side of experience reuse: gate, suggestion, coach.

The pull path (`apply_experience`) is covered by `test_experience_policy.py`.
What is new here is that the system, not the agent, decides when reuse runs. Two
failure modes matter more than anything else and most of these tests exist for
them: a gate that opens on an unreadable answer spends two extra model calls on a
query nobody wrote, and a coach that persists its injection rewrites the
trajectory the next step would otherwise read from cache.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from EvoScientist.memory.policy.gate import (
    GateOutputError,
    normalize_gate,
    parse_gate_json,
    render_library,
    render_recent,
)
from EvoScientist.memory.policy.suggest import (
    SUGGESTION_CLOSE,
    SUGGESTION_OPEN,
    render_suggestion,
)
from EvoScientist.middleware.coach import ExperienceCoachMiddleware


class TestGateParsing:
    """The gate's output is JSON from a small model; read it defensively."""

    def test_plain_json(self) -> None:
        decision = parse_gate_json(
            '{"reason": "choosing a learning rate", "need": true, '
            '"topic": "transformer fine-tuning", "method": "AdamW warmup", '
            '"task": "pick a learning rate for a 7B model", "state": "no runs yet"}'
        )
        assert decision["need"] is True
        assert decision["topic"] == "transformer fine-tuning"
        assert decision["method"] == "AdamW warmup"
        assert decision["state"] == "no runs yet"

    def test_fenced_json(self) -> None:
        decision = parse_gate_json(
            '```json\n{"reason": "r", "need": true, "topic": "t"}\n```'
        )
        assert decision["need"] is True
        assert decision["topic"] == "t"

    def test_json_with_trailing_prose(self) -> None:
        decision = parse_gate_json(
            'Here is my decision:\n{"reason": "r", "need": false}\nHope that helps.'
        )
        assert decision["need"] is False

    def test_no_json_raises(self) -> None:
        with pytest.raises(GateOutputError):
            parse_gate_json("I think experience would help here.")

    def test_non_object_raises(self) -> None:
        with pytest.raises(GateOutputError):
            parse_gate_json("[1, 2, 3]")


class TestGateNormalization:
    """``need`` is read strictly, because a false positive costs two calls."""

    @pytest.mark.parametrize("raw", [True, "true", "True", " true "])
    def test_truthy_forms_open_the_gate(self, raw: Any) -> None:
        assert normalize_gate({"need": raw, "topic": "t"})["need"] is True

    @pytest.mark.parametrize(
        "raw", [False, "false", "maybe", None, 1, "yes", "", "TRUE-ish"]
    )
    def test_everything_else_closes_it(self, raw: Any) -> None:
        assert normalize_gate({"need": raw, "topic": "t"})["need"] is False

    def test_missing_need_closes_it(self) -> None:
        assert normalize_gate({"topic": "t"})["need"] is False

    def test_need_without_topic_is_downgraded(self) -> None:
        """An empty topic ranks the library in directory order, not by relevance."""
        decision = normalize_gate({"need": True, "topic": "", "reason": "seems useful"})
        assert decision["need"] is False
        assert "downgraded" in decision["reason"]

    def test_downgrade_reason_survives_an_empty_reason(self) -> None:
        decision = normalize_gate({"need": True})
        assert decision["need"] is False
        assert decision["reason"].startswith("downgraded")

    def test_task_falls_back_to_topic(self) -> None:
        """``task`` keys the policy cache, so it must never be empty."""
        decision = normalize_gate({"need": True, "topic": "batch size scaling"})
        assert decision["task"] == "batch size scaling"

    def test_fields_are_flattened_and_capped(self) -> None:
        decision = normalize_gate(
            {"need": True, "topic": "a\n   b\tc", "task": "x" * 2000}
        )
        assert decision["topic"] == "a b c"
        assert len(decision["task"]) < 2000


class TestGateRendering:
    def test_render_recent_respects_limit_and_labels_roles(self) -> None:
        messages = [
            HumanMessage(content="original goal"),
            AIMessage(content="thinking"),
            ToolMessage(content="tool output", tool_call_id="1", name="ls"),
        ]
        rendered = render_recent(messages, limit=2)
        assert "original goal" not in rendered
        assert "[ai] thinking" in rendered
        assert "[tool] tool output" in rendered

    def test_render_recent_skips_empty_bodies(self) -> None:
        rendered = render_recent([AIMessage(content=""), AIMessage(content="kept")])
        assert rendered == "[ai] kept"

    def test_render_library_reports_counts_and_facets(self) -> None:
        rendered = render_library(
            {
                "records": 12,
                "papers": 4,
                "disciplines": [("cs", 9), ("bio", 3)],
                "top_domains": [("nlp", 7)],
            }
        )
        assert "records: 12 from 4 papers" in rendered
        assert "cs(9)" in rendered
        assert "nlp(7)" in rendered

    def test_render_library_without_stats(self) -> None:
        assert "unavailable" in render_library(None)
        assert "unavailable" in render_library({})


def _report(policy: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    return {"status": "ok", "policy": policy, "selected": ["E-1"], **extra}


class TestRenderSuggestion:
    def test_nothing_to_say_renders_nothing(self) -> None:
        assert render_suggestion({"status": "no_candidates", "policy": None}) == ""
        assert render_suggestion(_report(None)) == ""

    def test_decline_without_gaps_renders_nothing(self) -> None:
        """Injecting "memory had no opinion" costs tokens and teaches nothing."""
        assert (
            render_suggestion(_report({"verdict": "decline", "reason": "no fit"})) == ""
        )

    def test_decline_with_gaps_is_still_worth_saying(self) -> None:
        text = render_suggestion(
            _report(
                {
                    "verdict": "decline",
                    "reason": "no fit",
                    "unsupported": ["no records on 100B-scale training"],
                }
            )
        )
        assert "Memory does not cover these" in text
        assert "100B-scale" in text

    def test_full_policy_sections_and_wrapper(self) -> None:
        text = render_suggestion(
            _report(
                {
                    "verdict": "adapt",
                    "procedure": ["warm up then cosine decay"],
                    "rebind": [
                        {
                            "name": "learning rate",
                            "kind": "hyperparam",
                            "how_to_obtain": "sweep 1e-5..5e-5 on your dev split",
                            "source_value": "2e-5",
                        }
                    ],
                    "preconditions": ["decoder-only model"],
                    "declines": ["do not reuse the batch size"],
                    "conflicts": [
                        {"issue": "two schedules", "decide_by": "model size"}
                    ],
                    "checks": ["report dev loss"],
                    "unsupported": ["nothing on MoE"],
                    "sources": [{"id": "E-abc123"}],
                }
            )
        )
        assert text.startswith(SUGGESTION_OPEN)
        assert text.endswith(SUGGESTION_CLOSE)
        assert "`adapt`" in text
        for heading in (
            "What transfers to this step",
            "Values you must re-derive here rather than copy",
            "This only holds if",
            "What does not transfer",
            "Papers disagree here",
            "Verify before treating this as settled",
            "Memory does not cover these",
        ):
            assert heading in text
        assert "read_memory` on: E-abc123" in text

    def test_source_value_is_labelled_as_provenance(self) -> None:
        """The number must never appear without the label that disarms it."""
        text = render_suggestion(
            _report(
                {
                    "verdict": "adapt",
                    "procedure": ["step"],
                    "rebind": [
                        {
                            "name": "lr",
                            "how_to_obtain": "sweep it",
                            "source_value": "2e-5",
                        }
                    ],
                }
            )
        )
        line = next(ln for ln in text.splitlines() if "2e-5" in ln)
        assert "provenance only" in line
        assert line.index("sweep it") < line.index("2e-5")

    def test_non_e_sources_fall_back_to_selected_ids(self) -> None:
        text = render_suggestion(
            _report(
                {"verdict": "adapt", "procedure": ["step"], "sources": ["C-paper-1"]},
                selected=["E-fallback"],
            )
        )
        assert "E-fallback" in text
        assert "C-paper-1" not in text

    def test_malformed_rebind_entries_are_dropped(self) -> None:
        text = render_suggestion(
            _report(
                {
                    "verdict": "adapt",
                    "procedure": ["step"],
                    "rebind": ["a bare string", {"how_to_obtain": "no name"}],
                }
            )
        )
        assert "Values you must re-derive" not in text

    def test_items_are_capped(self) -> None:
        text = render_suggestion(
            _report({"verdict": "adapt", "procedure": [f"step {i}" for i in range(20)]})
        )
        assert "step 5" in text
        assert "step 6" not in text


def _coach(tmp_path, **kwargs: Any) -> ExperienceCoachMiddleware:
    return ExperienceCoachMiddleware(
        memory_dir=tmp_path, project_id="proj", model=MagicMock(), **kwargs
    )


class TestCoachShortCircuits:
    """Every step pays for the gate unless code can answer for free."""

    def test_no_messages(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        assert coach._skip_reason([], {"records": 5}) == "no messages yet"

    def test_empty_library(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        messages = [HumanMessage(content="hi")]
        assert "empty" in coach._skip_reason(messages, {"records": 0})
        assert "empty" in coach._skip_reason(messages, {})
        assert "empty" in coach._skip_reason(messages, None)

    def test_mid_lookup_after_read_only_tools(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[{"id": "1", "name": "grep", "args": {}}]),
            ToolMessage(content="out", tool_call_id="1", name="grep"),
        ]
        assert "mid-lookup" in coach._skip_reason(messages, {"records": 5})

    def test_a_real_action_does_not_short_circuit(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="", tool_calls=[{"id": "1", "name": "write", "args": {}}]
            ),
            ToolMessage(content="done", tool_call_id="1", name="write_file"),
        ]
        assert coach._skip_reason(messages, {"records": 5}) == ""

    def test_a_mixed_batch_counts_as_action(self, tmp_path) -> None:
        """One write in a parallel batch means the step is not pure lookup."""
        coach = _coach(tmp_path)
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(content="a", tool_call_id="1", name="read_memory"),
            ToolMessage(content="b", tool_call_id="2", name="write_file"),
        ]
        assert coach._skip_reason(messages, {"records": 5}) == ""

    def test_tool_scan_stops_at_the_requesting_ai_turn(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        messages = [
            ToolMessage(content="older", tool_call_id="0", name="write_file"),
            AIMessage(content="boundary"),
            ToolMessage(content="new", tool_call_id="1", name="grep"),
        ]
        assert coach._last_tool_names(messages) == {"grep"}

    def test_stats_failure_is_cached_not_retried(self, tmp_path, monkeypatch) -> None:
        coach = _coach(tmp_path)
        calls = {"n": 0}

        def boom(**_: Any) -> dict[str, Any]:
            calls["n"] += 1
            raise OSError("library unreadable")

        monkeypatch.setattr(
            "EvoScientist.memory.experiences.experience_library_stats", boom
        )
        assert coach._cached_stats() == {}
        assert coach._cached_stats() == {}
        assert calls["n"] == 1


class TestCoachRequestText:
    def test_first_real_human_turn_wins(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        messages = [
            HumanMessage(content="the original goal"),
            AIMessage(content="working"),
            HumanMessage(content="a follow-up"),
        ]
        assert coach._request_text(messages) == "the original goal"

    def test_synthetic_human_turns_are_skipped(self, tmp_path) -> None:
        """A summary paraphrases the goal; an injection replaces it."""
        coach = _coach(tmp_path)
        messages = [
            HumanMessage(
                content="<experience_guidance>...", additional_kwargs={"lc_source": "x"}
            ),
            HumanMessage(content="the real goal"),
        ]
        assert coach._request_text(messages) == "the real goal"

    def test_no_human_turn(self, tmp_path) -> None:
        assert _coach(tmp_path)._request_text([AIMessage(content="hi")]) == ""


def _request(messages: list[Any]) -> ModelRequest:
    return ModelRequest(
        messages=messages,
        model=MagicMock(),
        state={},
        runtime=MagicMock(),
        system_message=MagicMock(),
    )


class TestCoachInjection:
    def test_injection_appends_one_marked_human_turn(self, tmp_path) -> None:
        request = _request([HumanMessage(content="goal")])
        injected = ExperienceCoachMiddleware._inject(request, "guidance text")
        assert len(injected.messages) == 2
        assert injected.messages[-1].content == "guidance text"
        assert (
            injected.messages[-1].additional_kwargs["lc_source"] == "experience_coach"
        )

    def test_injection_does_not_mutate_the_original(self, tmp_path) -> None:
        """The trajectory the next step caches must stay byte-identical."""
        original = [HumanMessage(content="goal")]
        request = _request(original)
        ExperienceCoachMiddleware._inject(request, "guidance")
        assert len(original) == 1
        assert len(request.messages) == 1

    def test_system_message_is_untouched(self, tmp_path) -> None:
        """Guidance in the stable prefix would re-cache the whole prompt each step."""
        request = _request([HumanMessage(content="goal")])
        injected = ExperienceCoachMiddleware._inject(request, "guidance")
        assert injected.system_message is request.system_message

    @pytest.mark.asyncio
    async def test_no_suggestion_means_no_injection(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        coach._suggestion = AsyncMock(return_value="")
        seen: list[ModelRequest] = []

        async def handler(request: ModelRequest) -> str:
            seen.append(request)
            return "response"

        request = _request([HumanMessage(content="goal")])
        assert await coach.awrap_model_call(request, handler) == "response"
        assert len(seen[0].messages) == 1

    @pytest.mark.asyncio
    async def test_suggestion_is_injected_for_this_call(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        coach._suggestion = AsyncMock(return_value="guidance text")
        seen: list[ModelRequest] = []

        async def handler(request: ModelRequest) -> str:
            seen.append(request)
            return "response"

        await coach.awrap_model_call(_request([HumanMessage(content="g")]), handler)
        assert seen[0].messages[-1].content == "guidance text"

    @pytest.mark.asyncio
    async def test_a_failing_coach_is_invisible(self, tmp_path) -> None:
        """Reuse is an enhancement; it must never fail the agent's turn."""
        coach = _coach(tmp_path)
        coach._suggestion = AsyncMock(side_effect=RuntimeError("gate exploded"))

        async def handler(request: ModelRequest) -> str:
            return "response"

        request = _request([HumanMessage(content="goal")])
        assert await coach.awrap_model_call(request, handler) == "response"

    def test_sync_path_is_a_pass_through(self, tmp_path) -> None:
        coach = _coach(tmp_path)
        request = _request([HumanMessage(content="goal")])
        assert coach.wrap_model_call(request, lambda r: r) is request


class TestCoachFlow:
    """The `_suggestion` decision path, with the gate and pipeline stubbed."""

    @staticmethod
    def _messages() -> list[Any]:
        return [HumanMessage(content="pick a learning rate"), AIMessage(content="ok")]

    @pytest.fixture(autouse=True)
    def _stub_library(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "EvoScientist.memory.experiences.experience_library_stats",
            lambda **_: {"records": 9, "papers": 3},
        )

    @staticmethod
    def _decision(**over: Any) -> dict[str, Any]:
        return {
            "need": True,
            "reason": "choosing a learning rate",
            "topic": "fine-tuning",
            "method": "AdamW",
            "task": "pick a learning rate",
            "state": "",
        } | over

    @staticmethod
    def _policy_report() -> dict[str, Any]:
        return {
            "status": "ok",
            "cached": False,
            "selected": ["E-1"],
            "policy": {"verdict": "adapt", "procedure": ["warm up"]},
        }

    @pytest.mark.asyncio
    async def test_closed_gate_skips_the_pipeline(self, tmp_path, monkeypatch) -> None:
        derive = AsyncMock()
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision(need=False)),
        )
        monkeypatch.setattr("EvoScientist.memory.policy.derive_policy", derive)
        coach = _coach(tmp_path)
        assert await coach._suggestion(self._messages()) == ""
        derive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_open_gate_derives_and_renders(self, tmp_path, monkeypatch) -> None:
        derive = AsyncMock(return_value=self._policy_report())
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision()),
        )
        monkeypatch.setattr("EvoScientist.memory.policy.derive_policy", derive)
        coach = _coach(tmp_path)
        suggestion = await coach._suggestion(self._messages())
        assert SUGGESTION_OPEN in suggestion
        assert "warm up" in suggestion
        # The gate's facets are what reach retrieval, not the agent's own wording.
        assert derive.await_args.kwargs["method"] == "AdamW"
        assert derive.await_args.kwargs["task"] == "pick a learning rate"
        assert coach.interventions[-1]["selected"] == ["E-1"]

    @pytest.mark.asyncio
    async def test_unchanged_facets_reuse_without_model_calls(
        self, tmp_path, monkeypatch
    ) -> None:
        derive = AsyncMock(return_value=self._policy_report())
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision()),
        )
        monkeypatch.setattr("EvoScientist.memory.policy.derive_policy", derive)
        coach = _coach(tmp_path)
        first = await coach._suggestion(self._messages())
        second = await coach._suggestion(self._messages())
        assert second == first
        assert derive.await_count == 1

    @pytest.mark.asyncio
    async def test_changed_facets_derive_again(self, tmp_path, monkeypatch) -> None:
        derive = AsyncMock(return_value=self._policy_report())
        decisions = [self._decision(), self._decision(topic="evaluation")]
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(side_effect=decisions),
        )
        monkeypatch.setattr("EvoScientist.memory.policy.derive_policy", derive)
        coach = _coach(tmp_path)
        await coach._suggestion(self._messages())
        await coach._suggestion(self._messages())
        assert derive.await_count == 2

    @pytest.mark.asyncio
    async def test_gate_failure_is_swallowed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(side_effect=RuntimeError("aux model down")),
        )
        assert await _coach(tmp_path)._suggestion(self._messages()) == ""

    @pytest.mark.asyncio
    async def test_derivation_failure_is_swallowed(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision()),
        )
        monkeypatch.setattr(
            "EvoScientist.memory.policy.derive_policy",
            AsyncMock(side_effect=RuntimeError("retrieval broke")),
        )
        assert await _coach(tmp_path)._suggestion(self._messages()) == ""

    @pytest.mark.asyncio
    async def test_empty_render_is_not_recorded_as_an_intervention(
        self, tmp_path, monkeypatch
    ) -> None:
        """`no_candidates` is a normal outcome, not something to remember."""
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision()),
        )
        monkeypatch.setattr(
            "EvoScientist.memory.policy.derive_policy",
            AsyncMock(return_value={"status": "no_candidates", "policy": None}),
        )
        coach = _coach(tmp_path)
        assert await coach._suggestion(self._messages()) == ""
        assert coach.interventions == []


class TestRouteExclusivity:
    """Exactly one of the two reuse routes may be described to any one agent.

    The coached agent does not hold `apply_experience`, so the tool tutorial
    would point it at something it cannot call; the tool-holding subagent has no
    coach, so the push-side block would promise guidance that never arrives.
    """

    def _instructions(self, **kwargs: Any) -> str:
        from EvoScientist.middleware.memory import create_memory_middleware

        return create_memory_middleware(
            memory_dir="/tmp/does-not-need-to-exist", **kwargs
        )._observation_memory_instructions()

    def _blocks(self, text: str) -> tuple[bool, bool]:
        from EvoScientist.middleware.memory import (
            EXPERIENCE_COACH_INSTRUCTIONS,
            EXPERIENCE_POLICY_INSTRUCTIONS,
        )

        return (
            EXPERIENCE_COACH_INSTRUCTIONS in text,
            EXPERIENCE_POLICY_INSTRUCTIONS in text,
        )

    def test_coached_agent_gets_only_the_push_block(self) -> None:
        coach, tool = self._blocks(self._instructions(enable_experience_coach=True))
        assert coach
        assert not tool

    def test_tool_holder_gets_only_the_pull_block(self) -> None:
        coach, tool = self._blocks(self._instructions(enable_experience_coach=False))
        assert tool
        assert not coach

    def test_coach_flag_needs_the_reuse_layer(self) -> None:
        coach, tool = self._blocks(
            self._instructions(
                enable_experience_policy=False, enable_experience_coach=True
            )
        )
        assert not coach
        assert not tool

    def test_no_stray_tool_pointer_for_a_coached_agent(self) -> None:
        """Outside its own disclaimer, the withheld tool must not be named.

        The coach block says the tool is absent on purpose; anywhere else -- the
        cross-store routing table, the retrieval guidance -- naming it would send
        the agent after a tool it does not hold.
        """
        from EvoScientist.middleware.memory import EXPERIENCE_COACH_INSTRUCTIONS

        text = self._instructions(enable_experience_coach=True)
        assert "apply_experience" not in text.replace(EXPERIENCE_COACH_INSTRUCTIONS, "")

    def test_store_routing_survives_both_ways(self) -> None:
        for coached in (True, False):
            text = self._instructions(enable_experience_coach=coached)
            assert "search_observations" in text
            assert "search_experience" in text


class TestAgentWiring:
    """The three switches that decide who is coached and who holds the tool."""

    @staticmethod
    def _stack(**over: Any) -> tuple[list[Any], Any]:
        from EvoScientist import EvoScientist as evo

        cfg = evo._ensure_config()
        saved = {key: getattr(cfg, key) for key in over}
        try:
            for key, value in over.items():
                setattr(cfg, key, value)
            middleware = evo._get_default_middleware(
                for_async_subagent=over.pop("_async_subagent", False)
            )
        finally:
            for key, value in saved.items():
                setattr(cfg, key, value)
        memory = next(
            m for m in middleware if type(m).__name__ == "EvoMemoryMiddleware"
        )
        return middleware, memory

    def test_main_agent_is_coached(self) -> None:
        middleware, memory = self._stack()
        assert any(getattr(m, "name", "") == "experience_coach" for m in middleware)
        assert memory._enable_experience_coach is True

    def test_subagents_are_not_coached(self) -> None:
        """A per-step gate inside every subagent multiplies aux calls by the fan-out."""
        from EvoScientist import EvoScientist as evo

        middleware = evo._get_default_middleware(for_async_subagent=True)
        memory = next(
            m for m in middleware if type(m).__name__ == "EvoMemoryMiddleware"
        )
        assert not any(getattr(m, "name", "") == "experience_coach" for m in middleware)
        assert memory._enable_experience_coach is False

    def test_coach_off_removes_middleware_and_restores_the_tutorial(self) -> None:
        middleware, memory = self._stack(memory_experience_coach_enabled=False)
        assert not any(getattr(m, "name", "") == "experience_coach" for m in middleware)
        assert memory._enable_experience_coach is False

    def test_tool_is_registered_but_withheld_from_the_main_agent(self) -> None:
        """Subagent YAML resolves against the registry, so the entry always stays."""
        from EvoScientist import EvoScientist as evo

        cfg = evo._ensure_config()
        saved = cfg.memory_experience_coach_enabled
        try:
            cfg.memory_experience_coach_enabled = True
            registry, base = evo._build_paper_tools(cfg=cfg, workspace_dir=None)
            assert "apply_experience" in registry
            assert not any(t.name == "apply_experience" for t in base)

            cfg.memory_experience_coach_enabled = False
            registry, base = evo._build_paper_tools(cfg=cfg, workspace_dir=None)
            assert "apply_experience" in registry
            assert any(t.name == "apply_experience" for t in base)
        finally:
            cfg.memory_experience_coach_enabled = saved
