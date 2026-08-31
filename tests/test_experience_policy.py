"""Tests for the experience→policy reuse layer.

The reuse layer addresses the stale-binding problem: agents copy dataset names,
model names, and hyperparameters from experience records because those records
are source-bound (they state what the source paper did). The solution transforms
long source-bound prose into compact target-bound policies before injection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from EvoScientist.memory.policy.schema import (
    BINDING_KINDS,
    VERDICTS,
    PolicyOutputError,
    normalize_policy,
)
from EvoScientist.memory.policy.select import transferable_core
from EvoScientist.memory.policy.store import _policy_cache_key
from EvoScientist.memory.policy.synthesize import parse_policy_json


class TestSchema:
    """Test policy shape validation and normalization."""

    def test_empty_policy_valid(self) -> None:
        policy = normalize_policy(
            {
                "verdict": "decline",
                "reason": "no transferable procedure",
                "declines": ["task not covered"],
            }
        )
        assert policy["verdict"] == "decline"
        assert policy["reason"]
        assert policy["declines"]

    def test_adopt_requires_procedure(self) -> None:
        with pytest.raises(PolicyOutputError):
            normalize_policy({"verdict": "adopt", "procedure": []})

    def test_decline_requires_reason_or_declines(self) -> None:
        with pytest.raises(PolicyOutputError):
            normalize_policy({"verdict": "decline"})
        # Either field is sufficient
        p1 = normalize_policy({"verdict": "decline", "reason": "unsupported"})
        assert p1["reason"]
        p2 = normalize_policy({"verdict": "decline", "declines": ["out of scope"]})
        assert p2["declines"]

    def test_rebind_lenient_on_malformed(self) -> None:
        """Rebind normalization drops entries missing name or how_to_obtain."""
        policy = normalize_policy(
            {
                "verdict": "adapt",
                "procedure": ["step 1"],
                "rebind": [
                    {"name": "dataset", "how_to_obtain": "check the task"},
                    {"name": "broken"},  # missing how_to_obtain
                    {"how_to_obtain": "derive"},  # missing name
                    {"name": "scale", "how_to_obtain": "measure", "kind": "unknown"},
                ],
            }
        )
        assert len(policy["rebind"]) == 2
        assert policy["rebind"][0]["name"] == "dataset"
        # unknown kind → "other"
        assert policy["rebind"][1]["kind"] == "other"

    def test_conflict_normalization(self) -> None:
        policy = normalize_policy(
            {
                "verdict": "adapt",
                "procedure": ["read the conflict"],
                "conflicts": [
                    {"between": ["E-a_01", "E-b_02"], "disagreement": "X vs Y"},
                    {},  # missing disagreement, dropped
                ],
            }
        )
        assert len(policy["conflicts"]) == 1
        assert policy["conflicts"][0]["disagreement"] == "X vs Y"

    def test_all_required_fields_present(self) -> None:
        """Every field from the prompt's schema survives normalization."""
        payload = {
            "verdict": "adapt",
            "procedure": ["step 1", "step 2"],
            "rebind": [{"name": "ds", "how_to_obtain": "check task"}],
            "preconditions": ["must have X"],
            "declines": [],
            "checks": ["verify Y"],
            "conflicts": [],
            "unsupported": ["topic Z"],
            "sources": ["E-foo_01"],
            "reason": "partial coverage",
        }
        result = normalize_policy(payload)
        for field in (
            "verdict",
            "procedure",
            "rebind",
            "preconditions",
            "declines",
            "checks",
            "conflicts",
            "unsupported",
            "sources",
            "reason",
        ):
            assert field in result


class TestTransferableCore:
    """Test extraction of the paper-agnostic core statement."""

    def test_uses_structured_field_when_present(self) -> None:
        record = {
            "transferable_core": "Using reflection improves reasoning.",
            "statement": "On HumanEval with GPT-4, reflection improved...",
        }
        core = transferable_core(record)
        assert core == "Using reflection improves reasoning."

    def test_falls_back_to_statement_head(self) -> None:
        record = {"statement": "A" * 300 + " B" * 100}
        core = transferable_core(record)
        assert len(core) == 200
        assert "A" in core
        assert "B" not in core

    def test_handles_missing_fields(self) -> None:
        assert transferable_core({}) == ""


class TestCacheKey:
    """Test policy cache stability."""

    def test_same_inputs_same_key(self) -> None:
        k1 = _policy_cache_key(task="t1", selected_ids=["E-a_01", "E-b_02"])
        k2 = _policy_cache_key(task="t1", selected_ids=["E-a_01", "E-b_02"])
        assert k1 == k2

    def test_id_order_irrelevant(self) -> None:
        """Rerank order varies; the cache should still hit."""
        k1 = _policy_cache_key(task="t", selected_ids=["E-a_01", "E-b_02"])
        k2 = _policy_cache_key(task="t", selected_ids=["E-b_02", "E-a_01"])
        assert k1 == k2

    def test_different_task_different_key(self) -> None:
        k1 = _policy_cache_key(task="task one", selected_ids=["E-a_01"])
        k2 = _policy_cache_key(task="task two", selected_ids=["E-a_01"])
        assert k1 != k2

    def test_whitespace_normalized(self) -> None:
        k1 = _policy_cache_key(task="  task  ", selected_ids=["E-a_01"])
        k2 = _policy_cache_key(task="task", selected_ids=["E-a_01"])
        assert k1 == k2


class TestParsePolicyJSON:
    """Test tolerance of the writer model's output formatting."""

    def test_plain_json(self) -> None:
        payload = parse_policy_json('{"verdict": "adopt", "procedure": ["step"]}')
        assert payload["verdict"] == "adopt"
        assert payload["procedure"] == ["step"]

    def test_fenced_json(self) -> None:
        payload = parse_policy_json(
            '```json\n{"verdict": "adopt", "procedure": ["step"]}\n```'
        )
        assert payload["verdict"] == "adopt"

    def test_trailing_prose_tolerated(self) -> None:
        """A model that appends commentary after the object still parses."""
        payload = parse_policy_json(
            '{"verdict": "adapt", "procedure": ["x"]}\n\nHope that helps!'
        )
        assert payload["verdict"] == "adapt"

    def test_unparseable_raises(self) -> None:
        with pytest.raises(PolicyOutputError):
            parse_policy_json("not json at all")


class TestBindingKinds:
    """The binding taxonomy is shared between extraction and the policy layer."""

    def test_extraction_kinds_cover_policy_kinds(self) -> None:
        from EvoScientist.memory.experiences.extraction import _BINDING_KINDS

        assert set(BINDING_KINDS) == _BINDING_KINDS

    def test_verdicts_stable(self) -> None:
        assert VERDICTS == ("adopt", "adapt", "decline")


class _StubModel:
    """Serves rerank then writer output; counts calls to prove caching works."""

    def __init__(self, *, rerank: str | None = None, writer: str | None = None) -> None:
        self.rerank = rerank or json.dumps({"selected": [], "reason": "stub"})
        self.writer = writer or json.dumps(
            {"verdict": "adapt", "procedure": ["do the transferable part"]}
        )
        self.calls: list[str] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        from types import SimpleNamespace

        text = str(messages[0].content)
        kind = "rerank" if "rerank" in text.casefold() else "writer"
        self.calls.append(kind)
        return SimpleNamespace(content=self.rerank if kind == "rerank" else self.writer)


def _seed_one_experience(memory_dir: Path) -> None:
    from EvoScientist.memory.experiences.store import store_paper_experiences

    record = {
        "id": "l1_p_01",
        "layer": "L1",
        "domain": "information_retrieval",
        "task": "chunking for dense retrieval",
        "statement": "Boundary-aware chunking beat fixed splits on BEIR.",
        "transferable_core": "Chunk on structural boundaries, not fixed sizes.",
        "bindings": [{"name": "BEIR", "kind": "dataset"}],
        "scope": "dense retrieval",
        "confidence": 0.6,
        "utility": None,
        "domain_arxiv": None,
        "evidence": [{"source_id": "p", "section": "results", "quote": "q"}],
    }
    store_paper_experiences(
        memory_dir=memory_dir,
        project_id="proj",
        paper_id="p",
        url="",
        title="Chunking",
        paper_text="t",
        prompts={"l1": "x", "l2": "x"},
        payloads={
            "l1": {"paper_id": "p", "experiences": [record]},
            "l2": {"paper_id": "p", "experiences": []},
        },
    )


class TestPipeline:
    """derive_policy orchestration, including its degraded paths."""

    @pytest.mark.asyncio
    async def test_no_candidates_is_a_normal_outcome(self, tmp_path: Path) -> None:
        from EvoScientist.memory.policy import derive_policy

        report = await derive_policy(
            memory_dir=tmp_path,
            project_id="empty",
            task="anything at all",
            model=_StubModel(),
        )
        assert report["status"] == "no_candidates"
        assert report["policy"] is None
        assert report["hint"]

    @pytest.mark.asyncio
    async def test_synthesizes_then_serves_from_cache(self, tmp_path: Path) -> None:
        from EvoScientist.memory.policy import derive_policy

        _seed_one_experience(tmp_path)
        model = _StubModel()
        first = await derive_policy(
            memory_dir=tmp_path,
            project_id="proj",
            task="chunking strategy for retrieval over internal documents",
            model=model,
        )
        assert first["status"] == "ok"
        assert first["cached"] is False
        assert first["policy"]["verdict"] == "adapt"
        assert first["selected"]
        assert model.calls.count("writer") == 1

        second = await derive_policy(
            memory_dir=tmp_path,
            project_id="proj",
            task="chunking strategy for retrieval over internal documents",
            model=model,
        )
        assert second["cached"] is True
        # Rerank still runs (it picks the cache key), synthesis does not.
        assert model.calls.count("writer") == 1

    @pytest.mark.asyncio
    async def test_refresh_bypasses_cache(self, tmp_path: Path) -> None:
        from EvoScientist.memory.policy import derive_policy

        _seed_one_experience(tmp_path)
        model = _StubModel()
        kwargs: dict[str, Any] = {
            "memory_dir": tmp_path,
            "project_id": "proj",
            "task": "chunking strategy for retrieval",
            "model": model,
        }
        await derive_policy(**kwargs)
        report = await derive_policy(**kwargs, refresh=True)
        assert report["cached"] is False
        assert model.calls.count("writer") == 2

    @pytest.mark.asyncio
    async def test_unparseable_rerank_falls_back_to_scored_order(
        self, tmp_path: Path
    ) -> None:
        """One bad rerank response must not sink the whole call."""
        from EvoScientist.memory.policy import derive_policy

        _seed_one_experience(tmp_path)
        report = await derive_policy(
            memory_dir=tmp_path,
            project_id="proj",
            task="chunking strategy for retrieval",
            model=_StubModel(rerank="not json"),
        )
        assert report["status"] == "ok"
        assert report["selected"]
        assert "parse failed" in report["selection_reason"]

    @pytest.mark.asyncio
    async def test_unparseable_writer_output_propagates(self, tmp_path: Path) -> None:
        """Synthesis failure is not silently swallowed into an empty policy."""
        from EvoScientist.memory.policy import derive_policy

        _seed_one_experience(tmp_path)
        with pytest.raises(PolicyOutputError):
            await derive_policy(
                memory_dir=tmp_path,
                project_id="proj",
                task="chunking strategy for retrieval",
                model=_StubModel(writer="the model rambled instead"),
            )


class TestTrace:
    """Opt-in debug trace of the retrieve → rerank → synthesize chain.

    Off by default and gated purely by an env var (not the config schema) --
    this is a developer tool for inspecting what one `apply_experience` call
    actually did, not a feature. See `memory/policy/trace.py`.
    """

    def test_disabled_by_default_writes_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from EvoScientist.memory.policy.trace import emit_trace

        monkeypatch.delenv("EVOSCIENTIST_POLICY_TRACE", raising=False)
        emit_trace(tmp_path, "request", task="x")
        assert not (tmp_path / "policies" / "trace.jsonl").exists()

    @pytest.mark.asyncio
    async def test_enabled_traces_full_chain_under_one_call_id(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from EvoScientist.memory.policy import derive_policy

        monkeypatch.setenv("EVOSCIENTIST_POLICY_TRACE", "1")
        _seed_one_experience(tmp_path)
        await derive_policy(
            memory_dir=tmp_path,
            project_id="proj",
            task="chunking strategy for retrieval over internal documents",
            model=_StubModel(),
        )
        lines = (
            (tmp_path / "policies" / "trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        events = [json.loads(line) for line in lines]
        by_event = {e["event"]: e for e in events}
        assert set(by_event) == {
            "request",
            "retrieve",
            "rerank",
            "synthesize",
            "report",
        }
        call_ids = {e["call_id"] for e in events}
        assert len(call_ids) == 1  # one call, one id, joins the whole chain
        assert by_event["retrieve"]["candidate_ids"]
        assert by_event["synthesize"]["raw_output"]  # the pre-parse rewrite
        assert by_event["report"]["policy"]["verdict"] == "adapt"

    def test_disabled_after_being_enabled_writes_nothing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The switch is read live, not cached from process start."""
        from EvoScientist.memory.policy.trace import emit_trace

        monkeypatch.setenv("EVOSCIENTIST_POLICY_TRACE", "1")
        emit_trace(tmp_path, "request", task="x")
        monkeypatch.setenv("EVOSCIENTIST_POLICY_TRACE", "0")
        emit_trace(tmp_path, "request", task="y")
        lines = (
            (tmp_path / "policies" / "trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 1

    def test_write_failure_does_not_raise(self, tmp_path: Path, monkeypatch) -> None:
        """A trace write is best-effort: it must never break the pipeline."""
        from EvoScientist.memory.policy.trace import emit_trace

        monkeypatch.setenv("EVOSCIENTIST_POLICY_TRACE", "1")
        # Point the trace path at a location that cannot be created (a file's
        # child), which is the failure mode `mkdir` would raise OSError on.
        blocked = tmp_path / "not_a_dir"
        blocked.write_text("x", encoding="utf-8")
        monkeypatch.setenv(
            "EVOSCIENTIST_POLICY_TRACE_PATH", str(blocked / "trace.jsonl")
        )
        emit_trace(tmp_path, "request", task="x")  # must not raise

    def test_custom_trace_path_overrides_memory_dir_default(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from EvoScientist.memory.policy.trace import emit_trace

        monkeypatch.setenv("EVOSCIENTIST_POLICY_TRACE", "1")
        custom = tmp_path / "elsewhere" / "trace.jsonl"
        monkeypatch.setenv("EVOSCIENTIST_POLICY_TRACE_PATH", str(custom))
        emit_trace(tmp_path, "request", task="x")
        assert custom.exists()
        assert not (tmp_path / "policies" / "trace.jsonl").exists()


class TestTool:
    """apply_experience returns JSON the model can act on, in every outcome."""

    @pytest.mark.asyncio
    async def test_returns_policy_json(self, tmp_path: Path, monkeypatch) -> None:
        from EvoScientist.tools import create_apply_experience_tool

        _seed_one_experience(tmp_path)
        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_auxiliary_chat_model",
            lambda: _StubModel(),
        )
        tool = create_apply_experience_tool(memory_dir=tmp_path, project_id="proj")
        payload = json.loads(
            await tool.ainvoke({"task": "chunking strategy for retrieval"})
        )
        assert payload["status"] == "ok"
        assert payload["policy"]["verdict"] == "adapt"

    @pytest.mark.asyncio
    async def test_configured_max_selected_is_used(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The config value must reach the rerank prompt, not just the schema."""
        from EvoScientist.tools import create_apply_experience_tool

        _seed_one_experience(tmp_path)
        seen: list[str] = []

        class Recording(_StubModel):
            async def ainvoke(self, messages: list[Any]) -> Any:
                seen.append(str(messages[0].content))
                return await super().ainvoke(messages)

        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_auxiliary_chat_model",
            lambda: Recording(),
        )
        tool = create_apply_experience_tool(
            memory_dir=tmp_path, project_id="proj", max_selected=6
        )
        await tool.ainvoke({"task": "chunking strategy for retrieval"})
        rerank_prompt = next(p for p in seen if "candidates" in p.casefold())
        assert "Select 6 experiences" in rerank_prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("requested", "effective"), [(0, 1), (-1, 1), (99, 6)])
    async def test_configured_max_selected_is_clamped(
        self, tmp_path: Path, monkeypatch, requested: int, effective: int
    ) -> None:
        """Out-of-range config is clamped, mirroring settings.py validation for
        callers that construct the tool directly."""
        from EvoScientist.tools.experience_policy import create_apply_experience_tool

        _seed_one_experience(tmp_path)
        seen: list[str] = []

        class Recording(_StubModel):
            async def ainvoke(self, messages: list[Any]) -> Any:
                seen.append(str(messages[0].content))
                return await super().ainvoke(messages)

        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_auxiliary_chat_model",
            lambda: Recording(),
        )
        tool = create_apply_experience_tool(
            memory_dir=tmp_path, project_id="proj", max_selected=requested
        )
        await tool.ainvoke({"task": "chunking strategy for retrieval"})
        rerank_prompt = next(p for p in seen if "candidates" in p.casefold())
        assert f"Select {effective} experiences" in rerank_prompt

    @pytest.mark.asyncio
    async def test_failure_returns_hint_instead_of_raising(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A broken reuse layer must not end the caller's turn."""
        from EvoScientist.tools import create_apply_experience_tool

        _seed_one_experience(tmp_path)

        class Broken:
            async def ainvoke(self, messages: list[Any]) -> Any:
                raise RuntimeError("aux model down")

        monkeypatch.setattr(
            "EvoScientist.EvoScientist._ensure_auxiliary_chat_model", lambda: Broken()
        )
        tool = create_apply_experience_tool(memory_dir=tmp_path, project_id="proj")
        payload = json.loads(await tool.ainvoke({"task": "chunking strategy"}))
        assert payload["status"] == "error"
        assert "aux model down" in payload["error"]
        assert payload["hint"]


class TestInstructionGating:
    """Instructions and tool availability must be switched by the same flag.

    Injecting the reuse guidance to an agent that lacks `apply_experience` tells
    it to route decisions through a tool it cannot call.

    The assertions target the reuse *block* rather than the bare tool name,
    because the cross-store routing table names every memory tool in one place
    so the agent can tell the three stores apart. That table is what stops
    subject-matter queries from being aimed at the observation store, so it is
    not gated on the reuse layer.
    """

    def _instructions(self, **kwargs: Any) -> str:
        from EvoScientist.middleware.memory import create_memory_middleware

        middleware = create_memory_middleware(
            memory_dir="/tmp/does-not-need-to-exist", **kwargs
        )
        return middleware._observation_memory_instructions()

    def _has_policy_block(self, text: str) -> bool:
        from EvoScientist.middleware.memory import EXPERIENCE_POLICY_INSTRUCTIONS

        return EXPERIENCE_POLICY_INSTRUCTIONS in text

    def test_injected_when_enabled(self) -> None:
        text = self._instructions()
        assert self._has_policy_block(text)
        assert "apply_experience" in text

    def test_absent_when_disabled(self) -> None:
        text = self._instructions(enable_experience_policy=False)
        assert not self._has_policy_block(text)
        # The rest of the observation guidance must survive the flag.
        assert "search_observations" in text

    def test_absent_when_observations_off(self) -> None:
        """Reuse rides on observation memory; without it there is nothing to reuse."""
        text = self._instructions(enable_observation_memory=False)
        assert not self._has_policy_block(text)
        assert "apply_experience" not in text

    def test_experience_search_gated_independently(self) -> None:
        """Retrieval and reuse are separate switches.

        Disabling the two aux-model calls of `apply_experience` must not also
        remove the cheap read-only path to the library, or the inlined
        experience block would advertise a store with no way to reach it.
        """
        from EvoScientist.middleware.memory import EXPERIENCE_SEARCH_INSTRUCTIONS

        policy_off = self._instructions(enable_experience_policy=False)
        assert EXPERIENCE_SEARCH_INSTRUCTIONS in policy_off

        search_off = self._instructions(enable_experience_search=False)
        assert EXPERIENCE_SEARCH_INSTRUCTIONS not in search_off
        assert self._has_policy_block(search_off)


class TestExtractionBackwardCompatibility:
    """The ~100 records extracted before `bindings` existed must stay valid."""

    def _l1_record(self, **overrides: Any) -> dict[str, Any]:
        record = {
            "domain": "agent_learning",
            "task": "planning",
            "statement": "A" * 400,
            "applicable_when": ["setting"],
            "not_applicable_when": ["boundary"],
            "scope": "vision, 7B, finetune",
            "action": "did a thing",
            "effect": "improved by 3%",
            "practice_trace": [{"action": "a", "feedback": "b"}],
            "evidence": [{"section": "experiment", "quote": "Q" * 160}],
        }
        record.update(overrides)
        return record

    def test_record_without_new_fields_validates(self) -> None:
        from EvoScientist.memory.experiences.extraction import (
            _validate_llm_experience,
        )

        _validate_llm_experience(self._l1_record(), level="l1")

    def test_record_with_new_fields_validates(self) -> None:
        from EvoScientist.memory.experiences.extraction import (
            _validate_llm_experience,
        )

        _validate_llm_experience(
            self._l1_record(
                transferable_core="Doing X helps Y.",
                bindings=[{"name": "ImageNet", "kind": "dataset"}],
            ),
            level="l1",
        )

    def test_unknown_binding_kind_rejected(self) -> None:
        from EvoScientist.memory.experiences.extraction import (
            ExperienceOutputError,
            _validate_llm_experience,
        )

        with pytest.raises(ExperienceOutputError):
            _validate_llm_experience(
                self._l1_record(bindings=[{"name": "X", "kind": "nonsense"}]),
                level="l1",
            )

    def test_binding_without_name_rejected(self) -> None:
        from EvoScientist.memory.experiences.extraction import (
            ExperienceOutputError,
            _validate_llm_experience,
        )

        with pytest.raises(ExperienceOutputError):
            _validate_llm_experience(
                self._l1_record(bindings=[{"kind": "dataset"}]), level="l1"
            )

    def test_genuinely_unknown_field_still_rejected(self) -> None:
        """Loosening for two optional fields must not accept arbitrary keys."""
        from EvoScientist.memory.experiences.extraction import (
            ExperienceOutputError,
            _validate_llm_experience,
        )

        with pytest.raises(ExperienceOutputError):
            _validate_llm_experience(
                self._l1_record(hallucinated_field="oops"), level="l1"
            )

    def test_new_fields_survive_normalization(self) -> None:
        """Optional fields must reach the stored record, not be filtered out."""
        from EvoScientist.memory.experiences.extraction import (
            _normalize_current_payload,
        )

        payload = {
            "experiences": [
                self._l1_record(
                    transferable_core="core text",
                    bindings=[{"name": "ImageNet", "kind": "dataset"}],
                )
            ]
        }
        result = _normalize_current_payload(payload, level="l1", paper_id="2401.00001")
        stored = result["experiences"][0]
        assert stored["transferable_core"] == "core text"
        assert stored["bindings"][0]["name"] == "ImageNet"
        assert stored["id"].startswith("l1_")
