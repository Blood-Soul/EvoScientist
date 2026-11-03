"""Regression tests for the grounded coach: no invented task rules, no waiting.

The sentences below are the ones the coach actually produced in two recorded runs
(DuoRC, AIRS-Bench) and the agent then adopted as hard constraints. They are kept
verbatim so the check keeps catching the real failure, not a toy version of it.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage

from EvoScientist.memory.policy.grounding import (
    REWRITE,
    find_ungrounded_claims,
    ground_text,
)
from EvoScientist.middleware.coach import (
    ExperienceCoachMiddleware,
    create_experience_coach_middleware,
)

# The parts of the real task statement that matter here.
SPEC = """
# Overview
## Task Description
Answer questions about movie plots. The evaluation will be performed on the
`submission.csv` file you have submitted using the Accuracy metric. For each
question, if the submitted `has_answer` is `false`, the prediction is only
considered correct if the `no_answer` label is `true`. If the submitted
`has_answer` is `true`, the prediction is only considered correct if the
`no_answer` label is `false` and the submitted answer matches one of the answers.
> Note: The `answers` and `no_answer` columns are not available in the test set.
You have one RTX 4090 and 4 hours. Use only ./models/roberta-base.
"""


class TestInventedRulesAreCaught:
    """Verbatim from the recorded injections (exp 06:31:51, 07:01; exp2 12:59, 14:44)."""

    @pytest.mark.parametrize(
        "sentence",
        [
            "it accepts trading has-answer accuracy for no-answer accuracy, "
            "which this task excludes.",
            "this task imposes an answerable-accuracy non-inferiority constraint "
            "and a balanced-accuracy objective.",
            "Do not report only aggregate EM/F1; this task requires separated "
            "answerable and unanswerable accuracy.",
            "the source improvements rely on learned modules and this task forbids "
            "extra GPU compute.",
            "the current task fixes max_len 512, stride 256, n_best 12, and "
            "max_ans_tokens 16.",
        ],
    )
    def test_flagged_and_rewritten(self, sentence: str) -> None:
        assert find_ungrounded_claims(sentence, SPEC)
        fixed, changed = ground_text(sentence, SPEC)
        assert changed
        assert REWRITE in fixed or "your plan" in fixed
        assert "this task" not in fixed.lower().replace("not a rule in the task", "")

    def test_a_value_credited_to_the_task_statement_is_caught(self) -> None:
        sentence = (
            "Take the pre-declared feasibility tolerance from the current task "
            "statement (TOL = 0.005 on answerable accuracy) as the margin."
        )
        assert find_ungrounded_claims(sentence, SPEC)

    def test_derive_from_the_task_statement_is_an_instruction_not_a_claim(self) -> None:
        text = "Derive it from the task statement."
        assert find_ungrounded_claims(text, SPEC) == []

    def test_rewriting_is_idempotent(self) -> None:
        once, first = ground_text("this task forbids extra GPU compute.", SPEC)
        assert first
        twice, second = ground_text(once, SPEC)
        assert second == []
        assert twice == once

    def test_number_words_match_digits(self) -> None:
        # "4 hours" is in the statement; "four-hour" must count as supported.
        assert (
            find_ungrounded_claims(
                "the task fixes roberta-base and a four-hour run.", SPEC
            )
            == []
        )

    @pytest.mark.xfail(
        reason="Known gap: 'required by the task' (no 'statement') is not covered. "
        "Matching it needs a way to tell it from 'from the task data'; neither "
        "recorded run contains it, so it is left out rather than risk rewriting "
        "ordinary prepositional phrases.",
        strict=True,
    )
    def test_required_by_the_task_is_a_known_gap(self) -> None:
        assert find_ungrounded_claims(
            "Define the anti-tradeoff guard as required by the task.", SPEC
        )

    def test_the_advice_survives_only_the_authority_goes(self) -> None:
        fixed, _ = ground_text(
            "Do not train a new verifier; this task forbids extra GPU compute.", SPEC
        )
        assert fixed.startswith("Do not train a new verifier;")
        assert "forbids extra GPU compute" in fixed
        assert "could not be matched to the task statement" in fixed

    def test_possessive_keeps_grammar(self) -> None:
        fixed, _ = ground_text(
            "Use the task's fixed 512-token windows and 256-token stride.", SPEC
        )
        assert "your plan's fixed 512-token windows" in fixed


class TestRealRulesAreLeftAlone:
    @pytest.mark.parametrize(
        "sentence",
        [
            "The task states that the answers and no_answer columns are not "
            "available in the test set.",
            "This task requires a submission.csv scored with the Accuracy metric.",
            "The task asks for has_answer and an answer for every question.",
        ],
    )
    def test_grounded_claims_pass(self, sentence: str) -> None:
        assert find_ungrounded_claims(sentence, SPEC) == []
        assert ground_text(sentence, SPEC) == (sentence, [])

    def test_no_task_attribution_is_never_touched(self) -> None:
        text = "Report answerable and unanswerable accuracy separately."
        assert ground_text(text, SPEC) == (text, [])

    @pytest.mark.parametrize(
        "text",
        [
            "Identify the answerable/unanswerable labels from the task data.",
            "The source does not show a no-answer flag in this task has the same semantics.",
            "From this task's gold-answer length distribution, set the overlap.",
            "The tolerance is pre-declared in the project but not provided in the task text.",
        ],
    )
    def test_plain_prepositions_are_not_attributions(self, text: str) -> None:
        assert find_ungrounded_claims(text, SPEC) == []

    def test_imperative_and_noun_uses_are_not_claims(self) -> None:
        for text in (
            "Define EM/F1 over token index pairs for this task, state the normalization.",
            "The task state is a cached validation split with normalized scores.",
        ):
            assert find_ungrounded_claims(text, SPEC) == []

    def test_short_claims_cannot_be_verified_so_are_left_alone(self) -> None:
        assert find_ungrounded_claims("this task requires it.", SPEC) == []

    def test_empty_inputs(self) -> None:
        assert ground_text("", SPEC) == ("", [])
        # No task statement at all: every task-level claim is unsupported.
        assert find_ungrounded_claims("this task forbids extra GPU compute.", "")


# -- background mode ---------------------------------------------------------


def _request(messages: list[Any]) -> ModelRequest:
    return ModelRequest(
        model=MagicMock(),
        messages=messages,
        system_message=None,
        tools=[],
        state={},
        runtime=MagicMock(),
    )


def _coach(tmp_path, **kwargs: Any) -> ExperienceCoachMiddleware:
    return ExperienceCoachMiddleware(
        memory_dir=tmp_path, project_id="proj", model=MagicMock(), **kwargs
    )


def _acted(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"id": call_id, "name": name, "args": args}]
    )


async def _handler(request: ModelRequest) -> ModelRequest:
    return request


class TestBackgroundCoach:
    """Guidance is derived beside the agent, started by actions, bounded by a cap."""

    async def test_the_model_call_never_waits_for_the_coach(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True)
        release = asyncio.Event()

        async def slow(_messages: list[Any]) -> str:
            await release.wait()
            return "late guidance"

        coach._suggestion = slow  # type: ignore[method-assign]
        messages = [
            HumanMessage(content="goal"),
            _acted("write_file", "w1", file_path="a.py"),
        ]
        # Would hang forever if the call awaited the coach.
        out = await asyncio.wait_for(
            coach.awrap_model_call(_request(messages), _handler), 1
        )
        assert len(out.messages) == len(messages)
        release.set()
        await asyncio.sleep(0)

    async def test_lookups_do_not_start_a_derivation(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True, idle_trigger_steps=100)
        calls = 0

        async def counted(_messages: list[Any]) -> str:
            nonlocal calls
            calls += 1
            return ""

        coach._suggestion = counted  # type: ignore[method-assign]
        for i in range(6):
            messages = [
                HumanMessage(content="goal"),
                _acted("read_file", f"r{i}", file_path="a"),
            ]
            await coach.awrap_model_call(_request(messages), _handler)
            await asyncio.sleep(0)
        assert calls == 0

    async def test_a_substantive_action_starts_one(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True)
        coach._suggestion = AsyncMock(return_value="")  # type: ignore[method-assign]
        messages = [
            HumanMessage(content="goal"),
            _acted("write_file", "w1", file_path="a.py"),
        ]
        await coach.awrap_model_call(_request(messages), _handler)
        await asyncio.sleep(0.01)
        assert coach._suggestion.await_count == 1

    async def test_a_long_idle_stretch_also_starts_one(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True, idle_trigger_steps=3)
        coach._suggestion = AsyncMock(return_value="")  # type: ignore[method-assign]
        for i in range(4):
            messages = [
                HumanMessage(content="goal"),
                _acted("read_file", f"r{i}", file_path="a"),
            ]
            await coach.awrap_model_call(_request(messages), _handler)
            await asyncio.sleep(0.01)
        assert coach._suggestion.await_count == 1

    async def test_finished_guidance_arrives_on_a_later_call(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True)
        coach._suggestion = AsyncMock(return_value="guidance text")  # type: ignore[method-assign]
        start = [
            HumanMessage(content="goal"),
            _acted("write_file", "w1", file_path="a.py"),
        ]
        await coach.awrap_model_call(_request(start), _handler)  # starts it
        await asyncio.sleep(0.01)  # it finishes beside the agent
        later = [*start, _acted("read_file", "r1", file_path="a")]
        out = await coach.awrap_model_call(_request(later), _handler)
        assert out.messages[-1].content == "guidance text"
        assert len(out.messages) == len(later) + 1

    async def test_only_one_derivation_is_in_flight(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True)
        release = asyncio.Event()
        calls = 0

        async def slow(_messages: list[Any]) -> str:
            nonlocal calls
            calls += 1
            await release.wait()
            return ""

        coach._suggestion = slow  # type: ignore[method-assign]
        messages = [HumanMessage(content="goal")]
        for i in range(4):
            messages = [*messages, _acted("write_file", f"w{i}", file_path="a")]
            await coach.awrap_model_call(_request(messages), _handler)
            await asyncio.sleep(0)
        assert calls == 1
        release.set()
        await asyncio.sleep(0.01)

    async def test_the_agent_is_made_to_wait_after_the_step_cap(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True, max_wait_steps=2)
        release = asyncio.Event()

        async def slow(_messages: list[Any]) -> str:
            await release.wait()
            return "waited-for guidance"

        coach._suggestion = slow  # type: ignore[method-assign]
        messages = [
            HumanMessage(content="goal"),
            _acted("write_file", "w0", file_path="a"),
        ]
        await coach.awrap_model_call(_request(messages), _handler)  # step 1, starts
        messages = [*messages, _acted("read_file", "r1", file_path="a")]
        await coach.awrap_model_call(
            _request(messages), _handler
        )  # step 2, not yet overdue
        release_later = asyncio.get_running_loop().call_later(0.05, release.set)
        messages = [*messages, _acted("read_file", "r2", file_path="a")]
        out = await asyncio.wait_for(
            coach.awrap_model_call(_request(messages), _handler), 2
        )
        assert (
            "waited-for guidance" in out.messages[-1].content
        )  # step 3: cap reached, blocked
        release_later.cancel()

    async def test_a_hung_derivation_cannot_hang_the_agent(self, tmp_path) -> None:
        coach = _coach(
            tmp_path, background=True, max_wait_steps=1, max_wait_seconds=0.2
        )
        never = asyncio.Event()

        async def hang(_messages: list[Any]) -> str:
            await never.wait()
            return "never delivered"

        coach._suggestion = hang  # type: ignore[method-assign]
        m1 = [HumanMessage(content="goal"), _acted("write_file", "w0", file_path="a")]
        await coach.awrap_model_call(_request(m1), _handler)  # starts the derivation
        m2 = [*m1, _acted("read_file", "r1", file_path="a")]
        # step cap reached -> forced wait -> must return within the timeout, without guidance
        out = await asyncio.wait_for(coach.awrap_model_call(_request(m2), _handler), 2)
        assert len(out.messages) == len(m2)
        assert (
            coach._pending is None
            or coach._pending.cancelled()
            or coach._pending.done()
        )

    async def test_guidance_after_commits_says_what_the_agent_did(
        self, tmp_path
    ) -> None:
        coach = _coach(tmp_path, background=True)
        release = asyncio.Event()

        async def slow(_messages: list[Any]) -> str:
            await release.wait()
            return "the guidance"

        coach._suggestion = slow  # type: ignore[method-assign]
        messages = [
            HumanMessage(content="goal"),
            _acted("read_file", "r0", file_path="a"),
        ]
        coach._step = (
            coach._idle_trigger_steps
        )  # an idle stretch triggers the derivation
        await coach.awrap_model_call(_request(messages), _handler)
        messages = [*messages, _acted("edit_file", "e1", file_path="train.py")]
        await coach.awrap_model_call(_request(messages), _handler)
        release.set()
        await asyncio.sleep(0.01)
        messages = [*messages, _acted("read_file", "r2", file_path="a")]
        out = await coach.awrap_model_call(_request(messages), _handler)
        text = out.messages[-1].content
        assert "edit_file train.py" in text
        assert "the guidance" in text
        assert "may no longer apply" in text

    async def test_a_failing_derivation_is_silent(self, tmp_path) -> None:
        coach = _coach(tmp_path, background=True)
        coach._suggestion = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        messages = [
            HumanMessage(content="goal"),
            _acted("write_file", "w1", file_path="a"),
        ]
        await coach.awrap_model_call(_request(messages), _handler)
        await asyncio.sleep(0.01)
        messages = [*messages, _acted("read_file", "r1", file_path="a")]
        out = await coach.awrap_model_call(_request(messages), _handler)
        assert len(out.messages) == len(messages)

    async def test_blocking_mode_is_unchanged(self, tmp_path) -> None:
        coach = _coach(tmp_path)  # default: background off
        coach._suggestion = AsyncMock(return_value="now")  # type: ignore[method-assign]
        out = await coach.awrap_model_call(
            _request([HumanMessage(content="goal")]), _handler
        )
        assert out.messages[-1].content == "now"


class TestFactoryDefaults:
    def test_background_is_on_by_default(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("EVOSCIENTIST_COACH_BLOCKING", raising=False)
        coach = create_experience_coach_middleware(memory_dir=tmp_path, project_id="p")
        assert coach._background is True

    def test_env_switch_restores_blocking(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("EVOSCIENTIST_COACH_BLOCKING", "1")
        coach = create_experience_coach_middleware(memory_dir=tmp_path, project_id="p")
        assert coach._background is False


# -- what the models are given ------------------------------------------------


class TestWriterAndGateInputs:
    async def test_writer_prompt_gets_the_brief_and_the_situation(self) -> None:
        from EvoScientist.memory.policy.synthesize import synthesize_policy

        model = MagicMock()
        model.ainvoke = AsyncMock(
            return_value=MagicMock(
                content='{"verdict":"decline","reason":"r","declines":["d"]}'
            )
        )
        await synthesize_policy(
            task="the decision",
            selected=[],
            state="what was done",
            model=model,
            prompt="BRIEF={brief}|SIT={situation}|TASK={task}|STATE={state}|REC={records}",
            brief="THE-BRIEF",
            situation="WHERE-IT-IS",
        )
        sent = model.ainvoke.call_args.args[0][0].content
        assert "BRIEF=THE-BRIEF" in sent
        assert "SIT=WHERE-IT-IS" in sent

    async def test_gate_prompt_gets_brief_activity_map_and_candidates(self) -> None:
        from EvoScientist.memory.policy import gate

        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=MagicMock(content='{"need": false}'))
        await gate.decide_experience_need(
            brief_text="B-TEXT",
            activity_text="A-TEXT",
            library_map="M-TEXT",
            candidates_text="C-TEXT",
            allowed_ids=[],
            model=model,
            prompt="{brief}|{activity}|{library_map}|{candidates}",
        )
        sent = model.ainvoke.call_args.args[0][0].content
        assert sent == "B-TEXT|A-TEXT|M-TEXT|C-TEXT"

    def test_shipped_prompts_have_the_new_slots(self) -> None:
        from EvoScientist.memory.policy.prompts import (
            GATE_PROMPT_FILENAME,
            WRITER_PROMPT_FILENAME,
            load_policy_prompt,
        )

        writer = load_policy_prompt(WRITER_PROMPT_FILENAME)
        for slot in ("{brief}", "{situation}", "{task}", "{state}", "{records}"):
            assert slot in writer
        assert "ONLY authority" in writer
        gate_prompt = load_policy_prompt(GATE_PROMPT_FILENAME)
        for slot in ("{brief}", "{activity}", "{library_map}", "{candidates}"):
            assert slot in gate_prompt
        assert "not a rule of the task" in gate_prompt
        # Both must format without leftover braces.
        gate_prompt.format(brief="b", activity="a", library_map="m", candidates="c")
        writer.format(brief="b", situation="s", task="t", state="x", records="r")
