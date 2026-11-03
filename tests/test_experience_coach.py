"""Tests for the push side of experience reuse: gate, suggestion, coach.

The pull path (`apply_experience`) is covered by `test_experience_policy.py`.
What is new here is that the system, not the agent, decides when reuse runs. Two
failure modes matter more than anything else and most of these tests exist for
them: a gate that opens on an unreadable answer spends two extra model calls on a
query nobody wrote, and a coach that persists its injection rewrites the
trajectory the next step would otherwise read from cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from EvoScientist.memory.policy.gate import (
    GateOutputError,
    normalize_gate,
    parse_gate_json,
    render_activity,
    render_library,
    render_recent,
)
from EvoScientist.memory.policy.suggest import (
    SUGGESTION_CLOSE,
    SUGGESTION_OPEN,
    render_suggestion,
)
from EvoScientist.middleware.coach import ExperienceCoachMiddleware
from EvoScientist.paths import Workspace


class TestGateParsing:
    """The gate's output is JSON from a small model; read it defensively."""

    IDS = ("E-1", "E-2", "E-3")

    def test_plain_json(self) -> None:
        decision = parse_gate_json(
            '{"reason": "choosing a threshold", "need": true, '
            '"situation": "tuning the no-answer threshold", '
            '"pick": ["E-1", "E-2"], "purpose": "set the no-answer threshold"}',
            allowed_ids=self.IDS,
        )
        assert decision["need"] is True
        assert decision["pick"] == ["E-1", "E-2"]
        assert decision["purpose"] == "set the no-answer threshold"
        assert decision["situation"] == "tuning the no-answer threshold"

    def test_fenced_json(self) -> None:
        decision = parse_gate_json(
            '```json\n{"reason": "r", "need": true, "pick": ["E-1"]}\n```',
            allowed_ids=self.IDS,
        )
        assert decision["need"] is True
        assert decision["pick"] == ["E-1"]

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
    """``need`` is read strictly, and the gate may only name records that exist."""

    @pytest.mark.parametrize("raw", [True, "true", "True", " true "])
    def test_truthy_forms_open_the_gate(self, raw: Any) -> None:
        assert normalize_gate({"need": raw, "pick": ["E-1"]})["need"] is True

    @pytest.mark.parametrize(
        "raw", [False, "false", "maybe", None, 1, "yes", "", "TRUE-ish"]
    )
    def test_everything_else_closes_it(self, raw: Any) -> None:
        assert normalize_gate({"need": raw, "pick": ["E-1"]})["need"] is False

    def test_missing_need_closes_it(self) -> None:
        assert normalize_gate({"pick": ["E-1"]})["need"] is False

    def test_need_without_a_pick_is_downgraded(self) -> None:
        decision = normalize_gate({"need": True, "reason": "seems useful"})
        assert decision["need"] is False
        assert "downgraded" in decision["reason"]

    def test_downgrade_reason_survives_an_empty_reason(self) -> None:
        decision = normalize_gate({"need": True})
        assert decision["need"] is False
        assert decision["reason"].startswith("downgraded")

    def test_invented_ids_are_dropped(self) -> None:
        """The gate chooses from the candidates; an id it made up retrieves nothing."""
        decision = normalize_gate(
            {"need": True, "pick": ["E-1", "E-made-up", "E-1"]},
            allowed_ids=["E-1", "E-2"],
        )
        assert decision["pick"] == ["E-1"]

    def test_all_invented_ids_close_the_gate(self) -> None:
        decision = normalize_gate({"need": True, "pick": ["E-x"]}, allowed_ids=["E-1"])
        assert decision["need"] is False

    def test_pick_is_capped(self) -> None:
        ids = [f"E-{i}" for i in range(20)]
        assert len(normalize_gate({"need": True, "pick": ids})["pick"]) == 8

    def test_purpose_falls_back_to_the_situation(self) -> None:
        decision = normalize_gate(
            {"need": True, "pick": ["E-1"], "situation": "tuning a threshold"}
        )
        assert decision["purpose"] == "tuning a threshold"

    def test_fields_are_flattened_and_capped(self) -> None:
        decision = normalize_gate(
            {
                "need": True,
                "pick": ["E-1"],
                "situation": "a\n   b\tc",
                "purpose": "x" * 2000,
            }
        )
        assert decision["situation"] == "a b c"
        assert len(decision["purpose"]) < 2000


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

    def test_render_library_without_stats(self) -> None:
        assert "unavailable" in render_library(None)
        assert "unavailable" in render_library({})


class TestActivityRendering:
    """The gate reads actions and results, with the coach's own echo removed."""

    @staticmethod
    def _step(call_id: str, narration: str, name: str, command: str, result: str):
        return [
            AIMessage(
                content=narration,
                tool_calls=[
                    {"id": call_id, "name": name, "args": {"command": command}}
                ],
            ),
            ToolMessage(content=result, tool_call_id=call_id, name=name),
        ]

    def test_shows_narration_action_and_result(self) -> None:
        text = render_activity(
            self._step("1", "Now training", "execute", "python train.py", "loss 0.4")
        )
        assert "agent says: Now training" in text
        assert "did: execute: python train.py" in text
        assert "result: loss 0.4" in text

    @pytest.mark.parametrize(
        "narration",
        [
            "Guidance received. It asks for a balanced objective. Writing the sweep.",
            "Guidance noted. I'll adopt it: freeze the balanced objective.",
            "The guidance's key requirement is a pre-declared objective.",
            "**Guidance received** - it matches my plan.",
        ],
    )
    def test_the_agents_restatement_of_guidance_is_dropped_whole(
        self, narration: str
    ) -> None:
        """Dropping only the acknowledgement would leave the coach's advice in."""
        text = render_activity(self._step("1", narration, "write_file", "x", "ok"))
        assert "balanced" not in text
        assert "objective" not in text
        assert "agent says" not in text
        # What the agent then did is still shown.
        assert "did: write_file: x" in text

    def test_ordinary_narration_that_mentions_guidance_later_is_kept(self) -> None:
        text = render_activity(
            self._step(
                "1",
                "Training finished. The guidance was useful.",
                "execute",
                "python p.py",
                "ok",
            )
        )
        assert "Training finished" in text

    def test_an_injected_guidance_block_is_removed(self) -> None:
        text = render_activity(
            [
                AIMessage(
                    content="<experience_guidance>do X</experience_guidance> and I will go on"
                )
            ]
        )
        assert "do X" not in text
        assert "I will go on" in text

    def test_only_the_last_steps_are_shown(self) -> None:
        messages = []
        for i in range(30):
            messages += self._step(str(i), "", "execute", f"cmd-{i}", f"out-{i}")
        text = render_activity(messages, actions=3)
        assert "cmd-29" in text
        assert "cmd-27" in text
        assert "cmd-26" not in text

    def test_no_steps(self) -> None:
        assert render_activity([HumanMessage(content="goal")]) == "[no steps yet]"


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
    """The `_suggestion` decision path, with the gate, catalog and pipeline stubbed."""

    @staticmethod
    def _messages() -> list[Any]:
        return [HumanMessage(content="pick a threshold"), AIMessage(content="ok")]

    @pytest.fixture(autouse=True)
    def _stub(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "EvoScientist.memory.experiences.experience_library_stats",
            lambda **_: {"records": 9, "papers": 3},
        )
        monkeypatch.setattr(
            "EvoScientist.memory.policy.render_catalog",
            lambda **_: {
                "records": 9,
                "map": "- topic (9)",
                "candidate_ids": ["E-1", "E-2"],
                "candidates": "E-1 [L1] a\nE-2 [L2] b",
            },
        )
        brief = MagicMock()
        brief.render.return_value = "- (metric) accuracy"
        brief.to_dict.return_value = {"source": "model", "facts": [], "dropped": 0}
        monkeypatch.setattr(
            "EvoScientist.memory.policy.build_brief", AsyncMock(return_value=brief)
        )

    @staticmethod
    def _decision(**over: Any) -> dict[str, Any]:
        return {
            "need": True,
            "reason": "choosing a threshold",
            "situation": "tuning the no-answer threshold",
            "pick": ["E-1"],
            "purpose": "set the no-answer threshold",
        } | over

    @staticmethod
    def _policy_report() -> dict[str, Any]:
        return {
            "status": "ok",
            "cached": False,
            "selected": ["E-1"],
            "policy": {"verdict": "adapt", "procedure": ["calibrate on held-out rows"]},
        }

    @pytest.mark.asyncio
    async def test_closed_gate_skips_the_pipeline(self, tmp_path, monkeypatch) -> None:
        derive = AsyncMock()
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision(need=False, pick=[])),
        )
        monkeypatch.setattr("EvoScientist.memory.policy.derive_policy", derive)
        assert await _coach(tmp_path)._suggestion(self._messages()) == ""
        derive.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_open_gate_derives_from_the_picked_records(
        self, tmp_path, monkeypatch
    ) -> None:
        derive = AsyncMock(return_value=self._policy_report())
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision()),
        )
        monkeypatch.setattr("EvoScientist.memory.policy.derive_policy", derive)
        coach = _coach(tmp_path)
        suggestion = await coach._suggestion(self._messages())
        assert SUGGESTION_OPEN in suggestion
        assert "calibrate on held-out rows" in suggestion
        kwargs = derive.await_args.kwargs
        # The records the gate chose are what reach the pipeline, and the writer
        # receives the brief, not a paraphrase of the task.
        assert kwargs["picked_ids"] == ["E-1"]
        assert kwargs["task"] == "set the no-answer threshold"
        assert kwargs["brief"] == "- (metric) accuracy"
        assert kwargs["situation"] == "tuning the no-answer threshold"
        assert coach.interventions[-1]["selected"] == ["E-1"]

    @pytest.mark.asyncio
    async def test_the_gate_receives_the_brief_activity_and_catalog(
        self, tmp_path, monkeypatch
    ) -> None:
        gate = AsyncMock(return_value=self._decision(need=False, pick=[]))
        monkeypatch.setattr("EvoScientist.memory.policy.decide_experience_need", gate)
        await _coach(tmp_path)._suggestion(self._messages())
        kwargs = gate.await_args.kwargs
        assert kwargs["brief_text"] == "- (metric) accuracy"
        assert kwargs["library_map"] == "- topic (9)"
        assert "E-2 [L2] b" in kwargs["candidates_text"]
        assert kwargs["allowed_ids"] == ["E-1", "E-2"]

    @pytest.mark.asyncio
    async def test_the_same_pick_reuses_without_model_calls(
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
    async def test_a_different_pick_derives_again(self, tmp_path, monkeypatch) -> None:
        derive = AsyncMock(return_value=self._policy_report())
        decisions = [self._decision(), self._decision(pick=["E-2"])]
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

    @pytest.mark.asyncio
    async def test_the_brief_is_built_once(self, tmp_path, monkeypatch) -> None:
        build = AsyncMock()
        brief = MagicMock()
        brief.render.return_value = "b"
        brief.to_dict.return_value = {}
        build.return_value = brief
        monkeypatch.setattr("EvoScientist.memory.policy.build_brief", build)
        monkeypatch.setattr(
            "EvoScientist.memory.policy.decide_experience_need",
            AsyncMock(return_value=self._decision(need=False, pick=[])),
        )
        coach = _coach(tmp_path)
        for _ in range(3):
            await coach._suggestion(self._messages())
        assert build.await_count == 1


class TestRouteExclusivity:
    """Exactly one of the two reuse routes may be described to any one agent.

    The coached agent does not hold `apply_experience`, so the tool tutorial
    would point it at something it cannot call; the tool-holding subagent has no
    coach, so the push-side block would promise guidance that never arrives.
    """

    def _instructions(self, **kwargs: Any) -> str:
        from EvoScientist.middleware.memory import create_memory_middleware

        return create_memory_middleware(
            memory_dir="/tmp/does-not-need-to-exist",
            workspace_dir="/tmp/does-not-need-to-exist",
            **kwargs,
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
                workspace=Workspace(Path.cwd()),
                for_async_subagent=over.pop("_async_subagent", False),
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

        middleware = evo._get_default_middleware(
            workspace=Workspace(Path.cwd()), for_async_subagent=True
        )
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
            registry, base = evo._build_paper_tools(
                cfg=cfg, workspace=Workspace(Path.cwd()), work_dir=Path.cwd()
            )
            assert "apply_experience" in registry
            assert not any(t.name == "apply_experience" for t in base)

            cfg.memory_experience_coach_enabled = False
            registry, base = evo._build_paper_tools(
                cfg=cfg, workspace=Workspace(Path.cwd()), work_dir=Path.cwd()
            )
            assert "apply_experience" in registry
            assert any(t.name == "apply_experience" for t in base)
        finally:
            cfg.memory_experience_coach_enabled = saved


class TestCoachModelIsQuiet:
    """The coach's aux calls run inside the agent's model node; none may stream out.

    Regression: without the ``nostream`` tag, langgraph's messages handler emitted
    the gate/rerank/policy JSON as the agent's own ``text`` events, so it landed
    in the transcript and in the run's final answer.
    """

    @staticmethod
    async def _streamed_text(model: Any) -> str:
        from typing import TypedDict

        from langgraph.graph import END, START, StateGraph

        class State(TypedDict):
            out: str

        async def node(state: State) -> State:
            reply = await model.ainvoke([HumanMessage(content="Decide now.")])
            return {"out": reply.content}

        graph = StateGraph(State)
        graph.add_node("model", node)
        graph.add_edge(START, "model")
        graph.add_edge("model", END)
        app = graph.compile()
        chunks = [
            chunk.content
            async for chunk, _ in app.astream({"out": ""}, stream_mode="messages")
        ]
        return "".join(str(c) for c in chunks)

    @staticmethod
    def _fake() -> Any:
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )

        return FakeListChatModel(responses=['{"need": false, "reason": "x"}'])

    @pytest.mark.asyncio
    async def test_untagged_model_would_leak(self) -> None:
        """Control: proves the harness actually sees a leak when there is one."""
        assert '"need"' in await self._streamed_text(self._fake())

    @pytest.mark.asyncio
    async def test_resolved_model_emits_nothing(self, tmp_path) -> None:
        coach = ExperienceCoachMiddleware(
            memory_dir=tmp_path, project_id="proj", model=self._fake()
        )
        assert await self._streamed_text(coach._resolve_model()) == ""

    def test_tagging_is_applied_once_and_keeps_existing_tags(self, tmp_path) -> None:
        from langgraph.constants import TAG_NOSTREAM

        base = self._fake().model_copy(update={"tags": ["keep"]})
        coach = ExperienceCoachMiddleware(
            memory_dir=tmp_path, project_id="proj", model=base
        )
        model = coach._resolve_model()
        assert coach._resolve_model() is model
        assert model.tags == ["keep", TAG_NOSTREAM]
        assert model.disable_streaming is True
        assert base.tags == ["keep"]
