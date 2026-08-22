"""Tests for paper-navigator experience extraction and session memory."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import EvoScientist.EvoScientist as agent_module
from EvoScientist import paths
from EvoScientist.tools import paper_experience as P


def _l1_payload(paper_id: str = "1706.03762") -> dict:
    return {
        "paper_id": paper_id,
        "experiences": [
            {
                "granularity": "fine",
                "domain": "agent_learning",
                "keywords": ["training", "ablation"],
                "t": {
                    "summary": "Trains a model and measures a concrete gain.",
                    "modality": "text",
                    "scale": "1,000 examples",
                    "constraint": "Only tested in English.",
                },
                "e": "Dataset A; baseline B; accuracy metric.",
                "practice_trace": [
                    {
                        "action": "Changed the training schedule.",
                        "feedback": "Accuracy improved by 2 points.",
                    }
                ],
                "narrative": "Reusable procedural lesson.",
                "source_section": "results",
                "source_quote": "A supporting quote.",
            }
        ],
    }


def _l2_payload(paper_id: str = "1706.03762") -> dict:
    return {
        "paper_id": paper_id,
        "experiences": [
            {
                "claim_type": "relation",
                "domain": "agent_learning",
                "keywords": "training, scaling",
                "keywords_summary": "training scale",
                "declaration": "More targeted data improves the result.",
                "context": {
                    "summary": "Selects targeted data for training.",
                    "modality": "text",
                    "scale": "three datasets",
                    "constraint": "Requires matched task distributions.",
                },
                "μ": "high",
                "r": "The data better matches the target task.",
                "μ_r": "medium",
                "r_depth": "shallow",
                "narrative": "Reusable empirical and causal lesson.",
                "source_section": "discussion",
                "source_quote": "Another supporting quote.",
            }
        ],
    }


def _as_model_json(payload: dict, *, fenced: bool = False) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    return f"```json\n{text}\n```" if fenced else text


class PromptModel:
    def __init__(self, responses: dict[str, str]):
        self.responses = responses
        self.calls: list[str] = []

    async def ainvoke(self, messages):
        prompt = messages[0].content
        self.calls.append(prompt)
        return SimpleNamespace(content=self.responses[prompt])


def test_paper_identifier_normalizes_arxiv_and_doi():
    assert (
        P.paper_identifier("https://arxiv.org/abs/1706.03762v5?ref=search")
        == "1706.03762"
    )
    assert P.paper_identifier("ArXiv:1706.03762v2") == "1706.03762"
    assert P.paper_identifier("https://doi.org/10.1000/ABC") == "10.1000/ABC"


def test_parse_fenced_json_and_validate_shape():
    parsed = P.parse_experience_json(
        _as_model_json(_l1_payload(), fenced=True), level="l1"
    )
    assert parsed["experiences"][0]["granularity"] == "fine"

    with pytest.raises(P.ExperienceOutputError, match="experiences"):
        P.parse_experience_json('{"paper_id": "x"}', level="l2")


@pytest.mark.asyncio
async def test_extraction_uses_both_prompts_with_same_paper():
    model = PromptModel(
        {"L1": _as_model_json(_l1_payload()), "L2": _as_model_json(_l2_payload())}
    )

    results = await P.run_experience_extraction(
        "https://arxiv.org/abs/1706.03762",
        "# Paper body",
        prompts={"l1": "L1", "l2": "L2"},
        model=model,
    )

    assert set(results) == {"l1", "l2"}
    assert set(model.calls) == {"L1", "L2"}


@pytest.mark.asyncio
async def test_extraction_messages_are_system_plus_full_paper():
    model = SimpleNamespace(ainvoke=AsyncMock())
    model.ainvoke.return_value = SimpleNamespace(content=_as_model_json(_l1_payload()))

    await P.run_experience_extraction(
        "ArXiv:1706.03762",
        "# Paper body",
        levels=("l1",),
        prompts={"l1": "L1", "l2": "L2"},
        model=model,
    )

    messages = model.ainvoke.await_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "[paper_id] 1706.03762\n\n# Paper body"


def test_format_output_renders_fields_instead_of_raw_json():
    result = P.format_experience_output(
        "ArXiv:1706.03762", _l1_payload(), _l2_payload()
    )

    assert "**Paper ID:** 1706.03762" in result
    assert "## L1 Practical Experiences" in result
    assert "1. Action: Changed the training schedule." in result
    assert "Feedback: Accuracy improved by 2 points." in result
    assert "## L2 Inductive Experiences" in result
    assert "Core declaration: More targeted data improves the result." in result
    assert '"experiences":' not in result


def test_main_agent_registers_runtime_extraction_tool(monkeypatch):
    import EvoScientist.utils as utils

    cfg = SimpleNamespace(
        model="main-model",
        provider="provider",
        auxiliary_model="aux-model",
        auxiliary_provider="provider",
    )
    monkeypatch.setattr(utils, "load_subagents", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        agent_module, "_ensure_general_purpose_subagent", lambda subs: None
    )
    monkeypatch.setattr(
        agent_module, "_inject_subagent_middleware", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        agent_module,
        "_maybe_swap_async_subagents",
        lambda subs, middleware=None, *, tool_registry=None, cfg=None: subs,
    )
    monkeypatch.setattr(agent_module, "_configured_system_prompt", lambda cfg: "prompt")

    kwargs = agent_module._build_base_kwargs(
        base_backend=object(),
        base_middleware=[],
        cfg=cfg,
        chat_model=object(),
        workspace_dir="/workspace",
    )

    assert [tool.name for tool in kwargs["tools"]] == [
        "think_tool",
        "skill_manager",
        "extract_paper_experiences",
        "extract_paper_experiences_batch",
    ]


@pytest.mark.asyncio
async def test_batch_extraction_honors_concurrency_limit_and_reports_elapsed(
    tmp_path, monkeypatch
):
    files = []
    for index in range(3):
        paper_file = tmp_path / f"paper-{index}.md"
        paper_file.write_text(f"PAPER {index}", encoding="utf-8")
        files.append(paper_file)

    monkeypatch.setattr(P, "_resolve_workspace_paper_file", lambda value: Path(value))
    active = 0
    peak_active = 0

    async def extract(**kwargs):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return f"EXPERIENCES {kwargs['paper_id']}"

    monkeypatch.setattr(P, "extract_and_store_paper_experiences", extract)
    result = await P.extract_and_store_paper_experiences_batch(
        papers=[
            {"paper_id": f"paper-{index}", "paper_file": str(paper_file)}
            for index, paper_file in enumerate(files)
        ],
        session_id="thread-batch",
        memory_dir=tmp_path / "memories",
        max_concurrency=2,
    )

    assert peak_active == 2
    assert "- Papers: 3" in result
    assert "- Concurrency limit: 2" in result
    assert "experience_extraction_elapsed_seconds:" in result
    assert "EXPERIENCES paper-2" in result


@pytest.mark.asyncio
async def test_same_session_cache_hit_avoids_llm_calls(tmp_path):
    model = PromptModel(
        {"L1": _as_model_json(_l1_payload()), "L2": _as_model_json(_l2_payload())}
    )
    kwargs = {
        "paper_id": "ArXiv:1706.03762v3",
        "paper_markdown": "FULL PAPER",
        "session_id": "thread-one",
        "memory_dir": tmp_path,
        "model": model,
        "prompts": {"l1": "L1", "l2": "L2"},
    }

    first = await P.extract_and_store_paper_experiences(**kwargs)
    second = await P.extract_and_store_paper_experiences(**kwargs)

    assert len(model.calls) == 2
    assert "Session cache:** updated" in first
    assert "Session cache:** hit (no LLM calls)" in second
    paper_dirs = list((tmp_path / "paper_experiences" / "sessions").glob("*/*"))
    assert len(paper_dirs) == 1
    assert json.loads((paper_dirs[0] / "l1.json").read_text())["experiences"]
    assert json.loads((paper_dirs[0] / "l2.json").read_text())["experiences"]
    assert "### L1-001" in (paper_dirs[0] / "rendered.md").read_text()


@pytest.mark.asyncio
async def test_different_sessions_do_not_share_experiences(tmp_path):
    model = PromptModel(
        {"L1": _as_model_json(_l1_payload()), "L2": _as_model_json(_l2_payload())}
    )
    common = {
        "paper_id": "1706.03762",
        "paper_markdown": "FULL PAPER",
        "memory_dir": tmp_path,
        "model": model,
        "prompts": {"l1": "L1", "l2": "L2"},
    }

    await P.extract_and_store_paper_experiences(session_id="thread-a", **common)
    await P.extract_and_store_paper_experiences(session_id="thread-b", **common)

    assert len(model.calls) == 4
    session_dirs = list((tmp_path / "paper_experiences" / "sessions").iterdir())
    assert len(session_dirs) == 2


@pytest.mark.asyncio
async def test_partial_success_is_persisted_and_only_missing_level_retried(tmp_path):
    model = PromptModel({"L1": _as_model_json(_l1_payload()), "L2": "not-json"})
    kwargs = {
        "paper_id": "1706.03762",
        "paper_markdown": "FULL PAPER",
        "session_id": "thread-one",
        "memory_dir": tmp_path,
        "model": model,
        "prompts": {"l1": "L1", "l2": "L2"},
    }

    with pytest.raises(P.ExperienceOutputError, match="L2"):
        await P.extract_and_store_paper_experiences(**kwargs)

    paper_dir = next((tmp_path / "paper_experiences" / "sessions").glob("*/*"))
    assert (paper_dir / "l1.json").is_file()
    assert not (paper_dir / "l2.json").exists()

    model.responses["L2"] = _as_model_json(_l2_payload())
    result = await P.extract_and_store_paper_experiences(**kwargs)

    assert model.calls.count("L1") == 1
    assert model.calls.count("L2") == 2
    assert "## L2 Inductive Experiences" in result


@pytest.mark.asyncio
async def test_prompt_and_paper_hashes_invalidate_only_affected_cache(tmp_path):
    model = PromptModel(
        {
            "L1-v1": _as_model_json(_l1_payload()),
            "L1-v2": _as_model_json(_l1_payload()),
            "L2": _as_model_json(_l2_payload()),
        }
    )
    common = {
        "paper_id": "1706.03762",
        "session_id": "thread-one",
        "memory_dir": tmp_path,
        "model": model,
    }

    await P.extract_and_store_paper_experiences(
        paper_markdown="PAPER-v1",
        prompts={"l1": "L1-v1", "l2": "L2"},
        **common,
    )
    await P.extract_and_store_paper_experiences(
        paper_markdown="PAPER-v1",
        prompts={"l1": "L1-v2", "l2": "L2"},
        **common,
    )
    assert model.calls.count("L1-v1") == 1
    assert model.calls.count("L1-v2") == 1
    assert model.calls.count("L2") == 1

    await P.extract_and_store_paper_experiences(
        paper_markdown="PAPER-v2",
        prompts={"l1": "L1-v2", "l2": "L2"},
        **common,
    )
    assert model.calls.count("L1-v2") == 2
    assert model.calls.count("L2") == 2


@pytest.mark.asyncio
async def test_default_extraction_model_uses_auxiliary_getter(monkeypatch):
    model = PromptModel({"L1": _as_model_json(_l1_payload())})
    calls = 0

    def _getter():
        nonlocal calls
        calls += 1
        return model

    monkeypatch.setattr(P, "_get_experience_model", _getter)

    results = await P.run_experience_extraction(
        "1706.03762",
        "PAPER",
        levels=("l1",),
        prompts={"l1": "L1", "l2": "L2"},
    )

    assert calls == 1
    assert isinstance(results["l1"], dict)


@pytest.mark.asyncio
async def test_runtime_tool_uses_thread_id_and_workspace_file(tmp_path, monkeypatch):
    paper_file = tmp_path / "artifacts" / "paper.md"
    paper_file.parent.mkdir()
    paper_file.write_text("FULL PAPER", encoding="utf-8")
    monkeypatch.setattr(paths, "_active_workspace", tmp_path)
    model = PromptModel(
        {"L1": _as_model_json(_l1_payload()), "L2": _as_model_json(_l2_payload())}
    )
    getter_calls = 0

    def _model_getter():
        nonlocal getter_calls
        getter_calls += 1
        return model

    monkeypatch.setattr(P, "load_experience_prompts", lambda: {"l1": "L1", "l2": "L2"})
    tool = P.create_paper_experience_tool(
        memory_dir=tmp_path / "memories", model_getter=_model_getter
    )
    runtime = SimpleNamespace(
        execution_info=None,
        config={"configurable": {"thread_id": "thread-tool"}},
    )

    result = await tool.coroutine(
        paper_file="artifacts/paper.md",
        paper_id="1706.03762",
        runtime=runtime,
    )
    cached = await tool.coroutine(
        paper_file="artifacts/paper.md",
        paper_id="1706.03762",
        runtime=runtime,
    )

    assert "## L1 Practical Experiences" in result
    assert "hit (no LLM calls)" in cached
    assert len(model.calls) == 2
    assert getter_calls == 1
    assert "runtime" not in tool.tool_call_schema.model_fields


@pytest.mark.asyncio
async def test_cli_runner_reads_paper_file(tmp_path, monkeypatch):
    paper_file = tmp_path / "paper.md"
    paper_file.write_text("FULL PAPER", encoding="utf-8")
    extract = AsyncMock(return_value="RENDERED")
    monkeypatch.setattr(P, "extract_and_store_paper_experiences", extract)

    result = await P._run_cli(
        paper_file, "CorpusId:123", "thread-cli", tmp_path / "memories"
    )

    assert result == "RENDERED"
    extract.assert_awaited_once_with(
        paper_id="CorpusId:123",
        paper_markdown="FULL PAPER",
        session_id="thread-cli",
        memory_dir=tmp_path / "memories",
    )
