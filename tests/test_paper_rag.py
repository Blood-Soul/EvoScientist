"""Tests for paper full-text storage, chunking, retrieval, and tools.

The load-bearing property under test is separation: paper chunks must be
reachable through the dedicated paper tools and provably absent from
``search_observations``, whose ranking exists to surface ``E-*`` and ``O-*``
records rather than raw text.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from langchain.tools import ToolRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import ExecutionInfo

from EvoScientist.memory.agents import paper_experience_worker
from EvoScientist.memory.experiences import (
    enqueue_paper,
    list_experience_documents,
    list_tasks,
    search_memory_files,
    store_paper_experiences,
)
from EvoScientist.memory.experiences.store import paper_storage_key
from EvoScientist.memory.observations.index import build_observation_index_context
from EvoScientist.memory.papers import (
    chunk_paper_text,
    describe_chunks,
    has_paper_text,
    list_paper_chunk_documents,
    list_papers,
    load_paper_text,
    persist_paper_fulltext,
    read_paper_chunk,
    read_paper_full,
    reset_paper_fulltext_settings_cache,
    search_paper_chunks,
    store_paper_text,
)
from EvoScientist.memory.papers.chunking import (
    DEFAULT_OVERLAP_CHARS,
    chunk_id_for,
)
from EvoScientist.tools.paper_rag import (
    create_read_paper_tool,
    create_search_paper_text_tool,
)

PAPER_URL = "https://arxiv.org/abs/2401.09999"
PAPER_ID = "2401.09999"
PAPER_TITLE = "Contrastive Pretraining for Catalysts"


def _paper_text() -> str:
    """A Jina-Reader-shaped Markdown paper with nested headings."""
    method_body = (
        "We regularise the encoder with a temperature-scaled InfoNCE loss.\n\n"
        + "The projection head is a two-layer MLP with hidden width 512. " * 40
    )
    return (
        f"# {PAPER_TITLE}\n\n"
        "Abstract paragraph naming the contrastive objective.\n\n"
        "## Method\n\n"
        "We describe the pretraining pipeline.\n\n"
        "### Regularisation\n\n"
        f"{method_body}\n\n"
        "## Results\n\n"
        "Top-1 accuracy reaches 84.6 percent on the held-out catalyst split, "
        "against a 79.1 percent supervised baseline.\n\n"
        "## Limitations\n\n"
        "Short.\n"
    )


def _store_text(
    memory_dir: Path,
    *,
    project_id: str = "P-alpha",
    paper_id: str = PAPER_ID,
    url: str = PAPER_URL,
    title: str = PAPER_TITLE,
    text: str | None = None,
    **kwargs: object,
) -> Path:
    return store_paper_text(
        memory_dir=memory_dir,
        project_id=project_id,
        paper_id=paper_id,
        url=url,
        title=title,
        paper_text=text if text is not None else _paper_text(),
        **kwargs,
    )


def _tool_runtime(tool: Any, *, config: RunnableConfig | None = None) -> ToolRuntime:
    """Build a minimal runtime so a tool can see an injected project id."""
    return ToolRuntime(
        state={},
        context=None,
        config=config if config is not None else {},
        stream_writer=lambda _chunk: None,
        tool_call_id=None,
        store=None,
        tools=[tool],
        execution_info=ExecutionInfo(
            checkpoint_id="checkpoint-1",
            checkpoint_ns="",
            task_id="task-1",
            thread_id=None,
        ),
        server_info=None,
    )


def _experience_payload(paper_id: str, level: str) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "experiences": [
            {
                "id": f"{level}_{paper_id}_01",
                "layer": level.upper(),
                "statement": "Warm up the temperature before contrastive training.",
                "evidence": [],
            }
        ],
    }


def _store_experiences(
    memory_dir: Path,
    *,
    project_id: str = "P-alpha",
    paper_id: str = PAPER_ID,
    url: str = PAPER_URL,
    title: str = PAPER_TITLE,
) -> None:
    store_paper_experiences(
        memory_dir=memory_dir,
        project_id=project_id,
        paper_id=paper_id,
        url=url,
        title=title,
        paper_text="full paper",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
        payloads={
            level: _experience_payload(paper_id, level) for level in ("l1", "l2")
        },
    )


@pytest.fixture(autouse=True)
def _fresh_fulltext_settings() -> None:
    """Keep the module-level settings cache from leaking between tests."""
    reset_paper_fulltext_settings_cache()
    yield
    reset_paper_fulltext_settings_cache()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def test_chunking_follows_headings_and_tracks_section_paths() -> None:
    chunks = chunk_paper_text(
        _paper_text(), project_id="P-alpha", paper_key="key", max_chunk_chars=2000
    )

    assert chunks
    paths = [chunk.section_path for chunk in chunks]
    # The document title is an h1, so it heads every path below it.
    assert any(path.endswith("Method > Regularisation") for path in paths)
    assert any(path.endswith("Results") for path in paths)
    # Every chunk stays inside exactly one section.
    for chunk in chunks:
        assert chunk.section_path
        assert chunk.section in chunk.section_path


def test_chunking_labels_leading_text_as_preamble() -> None:
    text = "Title block with no heading.\n\n## Method\n\n" + ("body. " * 100)
    chunks = chunk_paper_text(text, project_id="P-alpha", paper_key="key")

    assert chunks[0].section_path == "Preamble"
    assert chunks[0].char_start == 0


def test_chunking_handles_heading_free_text() -> None:
    text = "One dense paragraph with no headings at all. " * 40
    chunks = chunk_paper_text(text, project_id="P-alpha", paper_key="key")

    assert chunks
    assert all(chunk.section_path == "" for chunk in chunks)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)


def test_chunking_returns_nothing_for_blank_text() -> None:
    assert chunk_paper_text("   \n\n  ", project_id="P-alpha", paper_key="key") == []


def test_oversized_section_splits_with_overlap_and_progresses() -> None:
    long_section = "## Method\n\n" + "\n\n".join(
        f"Paragraph {index} describing the loss in detail. " * 6 for index in range(40)
    )
    chunks = chunk_paper_text(
        long_section,
        project_id="P-alpha",
        paper_key="key",
        max_chunk_chars=800,
        overlap_chars=150,
    )

    method = [chunk for chunk in chunks if chunk.section == "Method"]
    assert len(method) > 1
    for previous, current in pairwise(method):
        # Windows advance, and consecutive windows overlap rather than abut.
        assert current.char_start > previous.char_start
        assert current.char_start < previous.char_end
        assert len(current.text) <= 800


def test_chunk_offsets_recover_text_from_stored_paper(tmp_path: Path) -> None:
    text = _paper_text()
    _store_text(tmp_path, text=text)

    loaded = load_paper_text(
        memory_dir=tmp_path, project_id="P-alpha", paper_id=PAPER_ID, url=PAPER_URL
    )
    assert loaded is not None
    stored_text = loaded["text"]
    assert stored_text == text
    for chunk in loaded["chunks"]:
        assert stored_text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


def test_overlap_is_clamped_below_the_window() -> None:
    # A pathological config (overlap >= window) must still terminate.
    chunks = chunk_paper_text(
        "sentence. " * 400,
        project_id="P-alpha",
        paper_key="key",
        max_chunk_chars=100,
        overlap_chars=100_000,
    )
    assert len(chunks) > 1
    assert chunks[-1].char_end == len("sentence. " * 400)


def test_chunk_ids_are_stable_across_rechunking() -> None:
    text = _paper_text()
    first = chunk_paper_text(text, project_id="P-alpha", paper_key="key")
    second = chunk_paper_text(text, project_id="P-alpha", paper_key="key")

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == chunk_id_for(
        project_id="P-alpha", paper_key="key", chunk_index=0
    )
    # Ids are project-scoped, so the same paper in another project differs.
    other = chunk_paper_text(text, project_id="P-beta", paper_key="key")
    assert other[0].chunk_id != first[0].chunk_id


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_store_writes_three_files_and_no_temporaries(tmp_path: Path) -> None:
    directory = _store_text(tmp_path)

    names = sorted(path.name for path in directory.iterdir())
    assert names == ["chunks.jsonl", "metadata.json", "paper.md"]
    assert not list(directory.glob("*.tmp*"))

    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["project_id"] == "P-alpha"
    assert metadata["paper_id"] == PAPER_ID
    assert metadata["chunk_count"] > 0
    assert metadata["section_count"] > 1
    assert metadata["char_count"] == len(_paper_text())
    assert metadata["overlap_chars"] == DEFAULT_OVERLAP_CHARS


def test_store_shares_its_directory_name_with_the_experience_store(
    tmp_path: Path,
) -> None:
    directory = _store_text(tmp_path)
    assert directory.name == paper_storage_key(PAPER_ID, PAPER_URL)
    assert has_paper_text(tmp_path, project_id="P-alpha", paper_key=directory.name)


def test_restoring_the_same_paper_replaces_rather_than_duplicates(
    tmp_path: Path,
) -> None:
    _store_text(tmp_path)
    first = list_paper_chunk_documents(memory_dir=tmp_path, project_id="P-alpha")

    directory = _store_text(tmp_path, text="# Paper\n\nA much shorter body.\n")
    second = list_paper_chunk_documents(memory_dir=tmp_path, project_id="P-alpha")

    assert len(second) < len(first)
    assert (directory / "paper.md").read_text(encoding="utf-8").endswith("body.\n")
    assert len(list_papers(memory_dir=tmp_path, project_id="P-alpha")) == 1


def test_paper_text_is_project_isolated(tmp_path: Path) -> None:
    _store_text(tmp_path, project_id="P-alpha")

    assert list_papers(memory_dir=tmp_path, project_id="P-beta") == []
    assert list_paper_chunk_documents(memory_dir=tmp_path, project_id="P-beta") == []
    assert (
        load_paper_text(
            memory_dir=tmp_path, project_id="P-beta", paper_id=PAPER_ID, url=PAPER_URL
        )
        is None
    )
    assert (
        read_paper_full(memory_dir=tmp_path, project_id="P-beta", paper_id=PAPER_ID)
        is None
    )


def test_malformed_chunk_lines_are_skipped_not_raised(tmp_path: Path) -> None:
    directory = _store_text(tmp_path)
    with (directory / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write("[1, 2, 3]\n")

    documents = list_paper_chunk_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert documents
    assert all(document.observation_id.startswith("C-") for document in documents)


def test_describe_chunks_resolves_paper_and_section(tmp_path: Path) -> None:
    _store_text(tmp_path)
    documents = list_paper_chunk_documents(memory_dir=tmp_path, project_id="P-alpha")
    ids = [document.observation_id for document in documents[:3]]

    described = describe_chunks(
        memory_dir=tmp_path, project_id="P-alpha", chunk_ids=ids
    )
    assert set(described) == set(ids)
    for detail in described.values():
        assert detail["paper"]["paper_id"] == PAPER_ID
        assert detail["paper"]["title"] == PAPER_TITLE
        assert isinstance(detail["section_path"], str)

    assert (
        describe_chunks(memory_dir=tmp_path, project_id="P-alpha", chunk_ids=[]) == {}
    )


# --------------------------------------------------------------------------
# Retrieval isolation
# --------------------------------------------------------------------------


def test_chunks_are_searchable_through_the_paper_entry_point(tmp_path: Path) -> None:
    _store_text(tmp_path)

    hits = search_paper_chunks(
        memory_dir=tmp_path, project_id="P-alpha", query="top-1 accuracy catalyst split"
    )
    assert hits
    assert all(hit["observation_id"].startswith("C-") for hit in hits)
    assert all(hit.get("record_kind") == "paper_chunk" for hit in hits)


def test_chunks_never_appear_in_observation_search(tmp_path: Path) -> None:
    """The central constraint: raw text must not crowd the E-*/O-* ranking."""
    _store_text(tmp_path)
    _store_experiences(tmp_path)

    observation_hits = search_memory_files(
        memory_dir=tmp_path,
        project_id="P-alpha",
        query="top-1 accuracy catalyst split contrastive",
        limit=20,
    )
    assert observation_hits, "the experience record should still be findable"
    assert all(not hit["observation_id"].startswith("C-") for hit in observation_hits)
    assert all(hit.get("record_kind") != "paper_chunk" for hit in observation_hits)

    # ...while the same query does reach the text through the paper tool.
    assert search_paper_chunks(
        memory_dir=tmp_path,
        project_id="P-alpha",
        query="top-1 accuracy catalyst split",
    )


def test_paper_id_filter_restricts_candidates_before_ranking(tmp_path: Path) -> None:
    _store_text(tmp_path)
    _store_text(
        tmp_path,
        paper_id="2402.00001",
        url="https://arxiv.org/abs/2402.00001",
        title="Other Paper",
        text="# Other Paper\n\n## Results\n\nTop-1 accuracy reaches 91.2 percent.\n",
    )

    filtered = search_paper_chunks(
        memory_dir=tmp_path,
        project_id="P-alpha",
        query="top-1 accuracy",
        paper_id="2402.00001",
    )
    assert filtered
    described = describe_chunks(
        memory_dir=tmp_path,
        project_id="P-alpha",
        chunk_ids=[hit["observation_id"] for hit in filtered],
    )
    assert {detail["paper"]["paper_id"] for detail in described.values()} == {
        "2402.00001"
    }


def test_regex_mode_searches_paper_text(tmp_path: Path) -> None:
    from EvoScientist.memory.types import ObservationSearchMode

    _store_text(tmp_path)
    hits = search_paper_chunks(
        memory_dir=tmp_path,
        project_id="P-alpha",
        query=r"8[0-9]\.\d percent",
        mode=ObservationSearchMode.REGEX,
    )
    assert hits


def test_blank_query_returns_no_hits(tmp_path: Path) -> None:
    _store_text(tmp_path)
    assert (
        search_paper_chunks(memory_dir=tmp_path, project_id="P-alpha", query="   ")
        == []
    )


# --------------------------------------------------------------------------
# read_paper granularities
# --------------------------------------------------------------------------


def _results_chunk_id(tmp_path: Path) -> str:
    hits = search_paper_chunks(
        memory_dir=tmp_path, project_id="P-alpha", query="top-1 accuracy catalyst"
    )
    return hits[0]["observation_id"]


def test_read_chunk_returns_only_that_passage(tmp_path: Path) -> None:
    _store_text(tmp_path)
    chunk_id = _results_chunk_id(tmp_path)

    result = read_paper_chunk(
        memory_dir=tmp_path, project_id="P-alpha", chunk_id=chunk_id, expand="chunk"
    )
    assert result is not None
    assert result["expand"] == "chunk"
    assert result["truncated"] is False
    assert "84.6" in result["text"]
    assert result["paper"]["paper_id"] == PAPER_ID


def test_read_section_spans_all_sibling_chunks_without_duplicating_overlap(
    tmp_path: Path,
) -> None:
    text = _paper_text()
    _store_text(tmp_path, text=text, max_chunk_chars=600, overlap_chars=120)

    documents = list_paper_chunk_documents(memory_dir=tmp_path, project_id="P-alpha")
    described = describe_chunks(
        memory_dir=tmp_path,
        project_id="P-alpha",
        chunk_ids=[document.observation_id for document in documents],
    )
    split_sections = {
        detail["section_path"]
        for detail in described.values()
        if sum(
            1
            for other in described.values()
            if other["section_path"] == detail["section_path"]
        )
        > 1
    }
    assert split_sections, "expected at least one section split into several chunks"
    target_path = next(iter(split_sections))
    chunk_id = next(
        chunk_id
        for chunk_id, detail in described.items()
        if detail["section_path"] == target_path
    )

    section = read_paper_chunk(
        memory_dir=tmp_path,
        project_id="P-alpha",
        chunk_id=chunk_id,
        expand="section",
    )
    assert section is not None
    assert section["expand"] == "section"
    siblings = [
        detail for detail in described.values() if detail["section_path"] == target_path
    ]
    assert section["char_start"] == min(detail["char_start"] for detail in siblings)
    assert section["char_end"] == max(detail["char_end"] for detail in siblings)
    # Reconstructed by slicing paper.md once, so overlap appears exactly once.
    assert section["text"] == text[section["char_start"] : section["char_end"]]


def test_read_section_truncates_and_reports_it(tmp_path: Path) -> None:
    _store_text(tmp_path)
    chunk_id = _results_chunk_id(tmp_path)

    result = read_paper_chunk(
        memory_dir=tmp_path,
        project_id="P-alpha",
        chunk_id=chunk_id,
        expand="section",
        max_chars=20,
    )
    assert result is not None
    assert result["truncated"] is True
    assert len(result["text"]) == 20
    assert result["char_end"] == result["char_start"] + 20


def test_read_full_returns_whole_paper_and_marks_truncation(tmp_path: Path) -> None:
    text = _paper_text()
    _store_text(tmp_path, text=text)

    whole = read_paper_full(
        memory_dir=tmp_path, project_id="P-alpha", paper_id=PAPER_ID
    )
    assert whole is not None
    assert whole["text"] == text
    assert whole["truncated"] is False
    assert whole["char_count"] == len(text)

    clipped = read_paper_full(
        memory_dir=tmp_path, project_id="P-alpha", paper_id=PAPER_ID, max_chars=100
    )
    assert clipped is not None
    assert clipped["truncated"] is True
    assert clipped["returned_chars"] == 100
    assert clipped["char_count"] == len(text)


def test_read_full_resolves_a_paper_by_url_or_key(tmp_path: Path) -> None:
    directory = _store_text(tmp_path)

    for identifier in (PAPER_URL, directory.name, "https://arxiv.org/abs/2401.09999v2"):
        found = read_paper_full(
            memory_dir=tmp_path, project_id="P-alpha", paper_id=identifier
        )
        assert found is not None, identifier
        assert found["paper"]["paper_id"] == PAPER_ID


def test_read_unknown_chunk_returns_none(tmp_path: Path) -> None:
    _store_text(tmp_path)
    assert (
        read_paper_chunk(
            memory_dir=tmp_path, project_id="P-alpha", chunk_id="C-doesnotexist"
        )
        is None
    )


# --------------------------------------------------------------------------
# Agent tools
# --------------------------------------------------------------------------


def test_search_tool_returns_locators_without_passage_text(tmp_path: Path) -> None:
    _store_text(tmp_path)
    tool = create_search_paper_text_tool(memory_dir=tmp_path, project_id="P-alpha")

    payload = json.loads(tool.run({"query": "top-1 accuracy catalyst split"}))
    assert payload["results"]
    first = payload["results"][0]
    assert set(first) == {
        "chunk_id",
        "paper_id",
        "title",
        "url",
        "section_path",
        "matches",
        "score",
    }
    assert first["chunk_id"].startswith("C-")
    assert first["paper_id"] == PAPER_ID
    assert first["title"] == PAPER_TITLE


def test_search_tool_differentiates_empty_store_from_no_match(tmp_path: Path) -> None:
    tool = create_search_paper_text_tool(memory_dir=tmp_path, project_id="P-alpha")

    empty = json.loads(tool.run({"query": "anything"}))
    assert empty["results"] == []
    assert empty["papers_stored"] == 0
    assert "No paper full text is stored" in empty["hint"]

    _store_text(tmp_path)
    missed = json.loads(tool.run({"query": "zzzzz nonexistent terminology qqqqq"}))
    assert missed["results"] == []
    assert missed["papers_stored"] == 1
    assert "lexical" in missed["hint"]


def test_read_tool_covers_all_three_granularities(tmp_path: Path) -> None:
    _store_text(tmp_path)
    tool = create_read_paper_tool(memory_dir=tmp_path, project_id="P-alpha")
    chunk_id = _results_chunk_id(tmp_path)

    chunk = json.loads(tool.run({"chunk_id": chunk_id, "expand": "chunk"}))
    assert chunk["expand"] == "chunk"

    section = json.loads(tool.run({"chunk_id": chunk_id}))
    assert section["expand"] == "section"

    whole = json.loads(tool.run({"paper_id": PAPER_ID, "expand": "full"}))
    assert whole["expand"] == "full"
    assert whole["text"] == _paper_text()


def test_read_tool_marks_full_truncation_inline(tmp_path: Path) -> None:
    _store_text(tmp_path)
    tool = create_read_paper_tool(memory_dir=tmp_path, project_id="P-alpha")

    payload = json.loads(
        tool.run({"paper_id": PAPER_ID, "expand": "full", "max_chars": 120})
    )
    assert payload["truncated"] is True
    assert "[truncated at 120 of" in payload["text"]


def test_read_tool_reports_misuse_as_errors(tmp_path: Path) -> None:
    _store_text(tmp_path)
    tool = create_read_paper_tool(memory_dir=tmp_path, project_id="P-alpha")

    assert (
        "expand must be one of"
        in json.loads(tool.run({"chunk_id": "C-x", "expand": "everything"}))["error"]
    )
    assert "requires paper_id" in json.loads(tool.run({"expand": "full"}))["error"]
    assert "chunk_id is required" in json.loads(tool.run({}))["error"]
    assert "No stored passage" in json.loads(tool.run({"chunk_id": "C-nope"}))["error"]
    assert (
        "No stored full text"
        in json.loads(tool.run({"paper_id": "missing", "expand": "full"}))["error"]
    )


def test_tools_follow_the_runtime_project_id(tmp_path: Path) -> None:
    _store_text(tmp_path, project_id="P-beta")
    # Constructed for P-alpha, but the runtime says P-beta.
    search = create_search_paper_text_tool(memory_dir=tmp_path, project_id="P-alpha")

    assert json.loads(search.run({"query": "top-1 accuracy"}))["results"] == []
    scoped = json.loads(
        search.run(
            {
                "query": "top-1 accuracy",
                "runtime": _tool_runtime(
                    search,
                    config={"configurable": {"evomemory_project_id": "P-beta"}},
                ),
            }
        )
    )
    assert scoped["results"]


# --------------------------------------------------------------------------
# Persistence wiring
# --------------------------------------------------------------------------


def test_persist_respects_the_feature_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from EvoScientist.memory.papers import persist as persist_module

    monkeypatch.setattr(
        persist_module,
        "paper_fulltext_settings",
        lambda: persist_module.PaperFulltextSettings(
            enabled=False, max_chunk_chars=2000, overlap_chars=200
        ),
    )
    assert (
        persist_paper_fulltext(
            memory_dir=tmp_path,
            project_id="P-alpha",
            paper_id=PAPER_ID,
            url=PAPER_URL,
            title=PAPER_TITLE,
            paper_text=_paper_text(),
        )
        is None
    )
    assert list_papers(memory_dir=tmp_path, project_id="P-alpha") == []


def test_persist_swallows_storage_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from EvoScientist.memory.papers import persist as persist_module

    def explode(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(persist_module, "store_paper_text", explode)
    assert (
        persist_paper_fulltext(
            memory_dir=tmp_path,
            project_id="P-alpha",
            paper_id=PAPER_ID,
            url=PAPER_URL,
            title=PAPER_TITLE,
            paper_text=_paper_text(),
        )
        is None
    )


@pytest.mark.asyncio
async def test_worker_persists_full_text_before_extracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id=PAPER_ID,
        url=PAPER_URL,
        title=PAPER_TITLE,
    )

    async def download(_url: str) -> str:
        return _paper_text()

    async def extract(**_kwargs):
        return {level: _experience_payload(PAPER_ID, level) for level in ("l1", "l2")}

    monkeypatch.setattr(paper_experience_worker, "download_paper_text", download)
    monkeypatch.setattr(
        paper_experience_worker,
        "load_experience_prompts",
        lambda: {"l1": "l1 prompt", "l2": "l2 prompt"},
    )
    monkeypatch.setattr(paper_experience_worker, "run_experience_extraction", extract)

    result = await paper_experience_worker.drain_paper_experience_queue(
        memory_dir=tmp_path, project_id="P-alpha", model=object()
    )

    assert result == {"processed": 1, "failed": 0}
    papers = list_papers(memory_dir=tmp_path, project_id="P-alpha")
    assert len(papers) == 1
    assert papers[0]["paper_id"] == PAPER_ID
    assert search_paper_chunks(
        memory_dir=tmp_path, project_id="P-alpha", query="top-1 accuracy"
    )


@pytest.mark.asyncio
async def test_full_text_survives_a_failed_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Text lands before extraction, so a retry reuses it instead of re-downloading."""
    enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id=PAPER_ID,
        url=PAPER_URL,
        title=PAPER_TITLE,
    )

    async def download(_url: str) -> str:
        return _paper_text()

    async def extract(**_kwargs):
        raise RuntimeError("model refused")

    monkeypatch.setattr(paper_experience_worker, "download_paper_text", download)
    monkeypatch.setattr(
        paper_experience_worker,
        "load_experience_prompts",
        lambda: {"l1": "l1 prompt", "l2": "l2 prompt"},
    )
    monkeypatch.setattr(paper_experience_worker, "run_experience_extraction", extract)

    result = await paper_experience_worker.drain_paper_experience_queue(
        memory_dir=tmp_path, project_id="P-alpha", model=object()
    )

    assert result == {"processed": 0, "failed": 1}
    assert list_tasks(memory_dir=tmp_path, project_id="P-alpha", status="failed")
    assert list_experience_documents(memory_dir=tmp_path, project_id="P-alpha") == []
    assert len(list_papers(memory_dir=tmp_path, project_id="P-alpha")) == 1


@pytest.mark.asyncio
async def test_active_tool_backfills_full_text_on_a_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Experiences cached before full-text storage existed get their text filled.

    A cache hit skips extraction, so it never reaches that path's download. The
    gap would never close on its own, and closing it costs one Jina fetch rather
    than any metered call -- so the backfill runs, once per paper.
    """
    from EvoScientist.tools import paper_experience_active

    _store_experiences(tmp_path)
    downloads: list[str] = []

    async def run_sync(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def download(url: str) -> str:
        downloads.append(url)
        return _paper_text()

    monkeypatch.setattr(paper_experience_active, "download_paper_text", download)
    monkeypatch.setattr(paper_experience_active.asyncio, "to_thread", run_sync)
    monkeypatch.setattr(
        paper_experience_active,
        "load_experience_prompts",
        lambda: {"l1": "l1 prompt", "l2": "l2 prompt"},
    )
    tool = paper_experience_active.create_extract_paper_experiences_tool(
        memory_dir=tmp_path, project_id="P-alpha"
    )
    request = {
        "papers": [{"url": PAPER_URL, "paper_id": PAPER_ID, "title": PAPER_TITLE}]
    }

    payload = json.loads(await tool.arun(request))
    assert payload["failed"] == []
    # Still a cache hit: the experiences were reused, not re-extracted.
    assert payload["completed"][0]["cached"] is True
    assert payload["completed"][0]["full_text_available"] is True
    assert downloads == [PAPER_URL]
    stored = list_papers(memory_dir=tmp_path, project_id="P-alpha")
    assert len(stored) == 1
    assert stored[0]["chunk_count"] >= 1

    # The text now exists, so a second hit reports it without fetching again.
    second = json.loads(await tool.arun(request))
    assert second["completed"][0]["full_text_available"] is True
    assert downloads == [PAPER_URL]


@pytest.mark.asyncio
async def test_a_failed_backfill_still_returns_the_cached_experiences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full text is a complement, so failing to fetch it is not a task failure."""
    from EvoScientist.tools import paper_experience_active

    _store_experiences(tmp_path)

    async def run_sync(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def download(_url: str) -> str:
        raise RuntimeError("Jina is unreachable")

    monkeypatch.setattr(paper_experience_active, "download_paper_text", download)
    monkeypatch.setattr(paper_experience_active.asyncio, "to_thread", run_sync)
    monkeypatch.setattr(
        paper_experience_active,
        "load_experience_prompts",
        lambda: {"l1": "l1 prompt", "l2": "l2 prompt"},
    )
    tool = paper_experience_active.create_extract_paper_experiences_tool(
        memory_dir=tmp_path, project_id="P-alpha"
    )

    payload = json.loads(
        await tool.arun(
            {"papers": [{"url": PAPER_URL, "paper_id": PAPER_ID, "title": PAPER_TITLE}]}
        )
    )
    assert payload["failed"] == []
    assert payload["completed"][0]["cached"] is True
    assert payload["completed"][0]["full_text_available"] is False
    assert list_papers(memory_dir=tmp_path, project_id="P-alpha") == []


@pytest.mark.asyncio
async def test_active_tool_stores_full_text_when_it_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from EvoScientist.tools import paper_experience_active

    async def run_sync(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def download(_url: str) -> str:
        return _paper_text()

    async def extract(**_kwargs):
        return {level: _experience_payload(PAPER_ID, level) for level in ("l1", "l2")}

    monkeypatch.setattr(paper_experience_active, "download_paper_text", download)
    monkeypatch.setattr(paper_experience_active.asyncio, "to_thread", run_sync)
    monkeypatch.setattr(
        paper_experience_active,
        "load_experience_prompts",
        lambda: {"l1": "l1 prompt", "l2": "l2 prompt"},
    )
    monkeypatch.setattr(paper_experience_active, "run_experience_extraction", extract)
    tool = paper_experience_active.create_extract_paper_experiences_tool(
        memory_dir=tmp_path, project_id="P-alpha"
    )

    payload = json.loads(
        await tool.arun(
            {"papers": [{"url": PAPER_URL, "paper_id": PAPER_ID, "title": PAPER_TITLE}]}
        )
    )
    assert payload["completed"][0]["cached"] is False
    assert payload["completed"][0]["full_text_available"] is True
    assert len(list_papers(memory_dir=tmp_path, project_id="P-alpha")) == 1
    assert search_paper_chunks(
        memory_dir=tmp_path, project_id="P-alpha", query="top-1 accuracy"
    )


# --------------------------------------------------------------------------
# Prompt surfaces
# --------------------------------------------------------------------------


def test_experience_records_point_at_their_own_full_text(tmp_path: Path) -> None:
    _store_experiences(tmp_path)
    _store_experiences(
        tmp_path,
        paper_id="2402.00002",
        url="https://arxiv.org/abs/2402.00002",
        title="No Text Paper",
    )
    _store_text(tmp_path)

    pointers = {}
    for document in list_experience_documents(
        memory_dir=tmp_path, project_id="P-alpha"
    ):
        paper = json.loads(document.text)["paper"]
        pointers[paper["paper_id"]] = paper

    assert pointers[PAPER_ID]["full_text_available"] is True
    assert pointers[PAPER_ID]["paper_key"] == paper_storage_key(PAPER_ID, PAPER_URL)
    assert pointers["2402.00002"]["full_text_available"] is False


def test_index_renders_one_line_per_paper(tmp_path: Path) -> None:
    _store_text(tmp_path)
    _store_text(
        tmp_path,
        paper_id="2402.00003",
        url="https://arxiv.org/abs/2402.00003",
        title="Second Paper",
    )
    _store_experiences(tmp_path)

    context = build_observation_index_context(memory_dir=tmp_path, project_id="P-alpha")
    block = context[context.index("<paper_fulltext_memory>") :]

    assert "Papers with stored full text: total=2." in block
    assert block.count("\n- ") == 2
    assert f"- {PAPER_ID}: {PAPER_TITLE}" in block
    assert "passages" in block
    assert "sections" in block
    assert "search_paper_text" in block


def test_index_offers_paper_text_even_without_experiences(tmp_path: Path) -> None:
    _store_text(tmp_path)

    context = build_observation_index_context(memory_dir=tmp_path, project_id="P-alpha")
    assert "<paper_experience_memory>" not in context
    assert "<paper_fulltext_memory>" in context


def test_index_omits_the_paper_block_when_nothing_is_stored(tmp_path: Path) -> None:
    context = build_observation_index_context(memory_dir=tmp_path, project_id="P-alpha")
    assert "<paper_fulltext_memory>" not in context


def test_index_truncates_the_paper_block_to_the_shared_budget(tmp_path: Path) -> None:
    for index in range(30):
        _store_text(
            tmp_path,
            paper_id=f"2403.{index:05d}",
            url=f"https://arxiv.org/abs/2403.{index:05d}",
            title=f"Paper number {index} with a deliberately long title " * 3,
        )

    context = build_observation_index_context(
        memory_dir=tmp_path, project_id="P-alpha", max_inline_chars=1200
    )
    assert len(context) <= 1200
    if "<paper_fulltext_memory>" in context:
        block = context[context.index("<paper_fulltext_memory>") :]
        assert block.count("\n- ") < 30


def test_middleware_instructions_are_gated_on_the_switch(tmp_path: Path) -> None:
    from EvoScientist.memory.observations import MemorySourceType
    from EvoScientist.middleware.memory import (
        PAPER_FULLTEXT_INSTRUCTIONS,
        create_memory_middleware,
    )

    def instructions(**kwargs: object) -> str:
        middleware = create_memory_middleware(
            str(tmp_path),
            workspace_dir=tmp_path,
            source_type=MemorySourceType.TURN,
            source_agent="EvoScientist",
            **kwargs,
        )
        return middleware._observation_memory_instructions()

    assert "search_paper_text" in PAPER_FULLTEXT_INSTRUCTIONS
    assert PAPER_FULLTEXT_INSTRUCTIONS in instructions()
    assert PAPER_FULLTEXT_INSTRUCTIONS not in instructions(enable_paper_fulltext=False)
    # Paper guidance describes the second half of one preflight, so it is
    # suppressed whenever observation memory itself is off.
    assert PAPER_FULLTEXT_INSTRUCTIONS not in instructions(
        enable_observation_memory=False
    )


def test_paper_tools_are_always_included_by_the_tool_selector() -> None:
    from EvoScientist.middleware.tool_selector import DEFAULT_ALWAYS_INCLUDE_TOOLS

    assert "search_paper_text" in DEFAULT_ALWAYS_INCLUDE_TOOLS
    assert "read_paper" in DEFAULT_ALWAYS_INCLUDE_TOOLS


def test_only_research_and_planner_subagents_receive_the_paper_tools() -> None:
    from EvoScientist.EvoScientist import SUBAGENTS_CONFIG
    from EvoScientist.utils import load_subagents

    subagents = load_subagents(SUBAGENTS_CONFIG)
    # `tools` stays empty until an execution mode resolves names against a
    # registry; the YAML grant lives in `_tool_names`.
    granted = {
        agent["name"]
        for agent in subagents
        if "search_paper_text" in (agent.get("_tool_names") or [])
    }
    assert granted == {"research-agent", "planner-agent"}
    for agent in subagents:
        if agent["name"] in granted:
            assert "read_paper" in agent["_tool_names"]
