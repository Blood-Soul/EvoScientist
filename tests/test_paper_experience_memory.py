from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import EvoScientist.middleware.memory as memory_middleware
from EvoScientist.memory.agents import paper_experience_worker
from EvoScientist.memory.experiences import (
    EXPERIENCE_CATALOG_FILENAME,
    browse_experience_facets,
    claim_next_task,
    enqueue_paper,
    experience_catalog_path,
    list_experience_documents,
    list_tasks,
    read_memory_file,
    refresh_all_experience_catalogs,
    run_experience_extraction,
    search_experience_records,
    store_paper_experiences,
)
from EvoScientist.memory.experiences.extraction import (
    ExperienceOutputError,
    _strip_references_section,
    parse_experience_json,
)
from EvoScientist.memory.observations import (
    MemoryScope,
    MemorySourceType,
    MemoryType,
    record_observation_file,
    search_observation_files,
)
from EvoScientist.tools import paper_experience_active
from EvoScientist.tools.paper_experience_queue import (
    create_paper_experience_queue_tool,
)


def _payload(paper_id: str, text: str) -> dict[str, object]:
    return {"paper_id": paper_id, "experiences": [{"narrative": text}]}


def _new_payload(paper_id: str, text: str, *, level: str) -> dict[str, object]:
    if level == "l1":
        item = {
            "id": f"l1_{paper_id}_01",
            "layer": "L1",
            "domain": "agent_learning",
            "domain_arxiv": "cs.AI",
            "task": "test task",
            "statement": text,
            "applicable_when": ["test setting"],
            "not_applicable_when": ["different setting"],
            "scope": "test scope",
            "action": "test action",
            "effect": "test effect",
            "utility": None,
            "confidence": 0.7,
            "practice_trace": [{"action": "test action", "feedback": "test feedback"}],
            "evidence": [
                {
                    "source_id": paper_id,
                    "section": "experiment",
                    "quote": "test evidence",
                }
            ],
        }
    else:
        item = {
            "id": f"l2_{paper_id}_01",
            "layer": "L2",
            "domain": "agent_learning",
            "domain_arxiv": "cs.AI",
            "task": "test task",
            "statement": text,
            "claim_type": "conditional",
            "applicable_when": ["test setting"],
            "not_applicable_when": ["different setting"],
            "scope": "test scope",
            "action": "test action",
            "effect": "test effect",
            "utility": None,
            "confidence": 0.7,
            "rationale": "test rationale",
            "rationale_depth": "shallow",
            "evidence": [
                {
                    "source_id": paper_id,
                    "section": "experiment",
                    "quote": "test evidence",
                }
            ],
        }
    return {"paper_id": paper_id, "experiences": [item]}


def _llm_payload(paper_id: str, text: str, *, level: str) -> dict[str, object]:
    item = {
        "domain": "agent_learning",
        "task": "test task",
        "statement": text,
        "applicable_when": ["test setting"],
        "not_applicable_when": ["different setting"],
        "scope": "test scope",
        "action": "test action",
        "effect": "test effect",
        "evidence": [{"section": "experiment", "quote": "test evidence"}],
    }
    if level == "l1":
        item["practice_trace"] = [
            {"action": "step 1", "feedback": "feedback 1"},
            {"action": "step 2", "feedback": "feedback 2"},
            {"action": "step 3", "feedback": "feedback 3"},
        ]
    else:
        item.update(
            {
                "claim_type": "conditional",
                "rationale": "test rationale",
                "rationale_depth": "shallow",
            }
        )
    return {"experiences": [item]}


def _store(
    memory_dir: Path, *, project_id: str, marker: str = "contrastive catalyst"
) -> None:
    store_paper_experiences(
        memory_dir=memory_dir,
        project_id=project_id,
        paper_id="2401.01234",
        url="https://arxiv.org/abs/2401.01234",
        title="Catalyst Study",
        paper_text="full paper",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
        payloads={
            "l1": _payload("2401.01234", f"practical {marker}"),
            "l2": _payload("2401.01234", f"inductive {marker}"),
        },
    )


def test_experience_storage_is_project_scoped_and_searchable(tmp_path: Path) -> None:
    _store(tmp_path, project_id="P-alpha")

    alpha = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    beta = list_experience_documents(memory_dir=tmp_path, project_id="P-beta")

    assert len(alpha) == 2
    assert beta == []
    assert {document.record_kind for document in alpha} == {"experience"}
    assert {document.experience_level for document in alpha} == {"l1", "l2"}
    assert all("experiences/projects/P-alpha" in document.path for document in alpha)

    hits = search_experience_records(
        memory_dir=tmp_path,
        project_id="P-alpha",
        topic="contrastive catalyst",
    )
    assert {hit["record_kind"] for hit in hits} == {"experience"}
    read = read_memory_file(
        memory_dir=tmp_path,
        project_id="P-alpha",
        record_id=hits[0]["observation_id"],
    )
    assert read is not None
    assert read["record_kind"] == "experience"
    assert "contrastive catalyst" in read["text"]


def test_observation_and_experience_retrieval_are_separate_entry_points(
    tmp_path: Path,
) -> None:
    """Each store answers only its own queries, even on a shared token.

    This previously asserted the opposite -- that one query returned both kinds
    -- which is the behaviour that broke experience retrieval. A single entry
    point meant a single query vocabulary, and since the observation store came
    first, process-shaped queries got aimed at records that only ever hold
    subject-matter findings.
    """
    _store(tmp_path, project_id="P-alpha", marker="shared retrieval token")
    record_observation_file(
        memory_dir=tmp_path,
        project_id="P-alpha",
        memory_type=MemoryType.SEMANTIC,
        summary="Shared retrieval token observation",
        observation="shared retrieval token appears in an observation",
        why_it_matters="It verifies split retrieval over shared storage roots.",
        scope=MemoryScope.PROJECT,
        source_type=MemorySourceType.TURN,
        source_session_id="thread-1",
        source_agent="EvoScientist",
    )

    experience_hits = search_experience_records(
        memory_dir=tmp_path,
        project_id="P-alpha",
        topic="shared retrieval token",
        limit=10,
    )
    observation_hits = search_observation_files(
        memory_dir=tmp_path,
        project_id="P-alpha",
        query="shared retrieval token",
        limit=10,
    )

    assert experience_hits
    assert observation_hits
    assert {hit["record_kind"] for hit in experience_hits} == {"experience"}
    assert all(
        hit.get("record_kind", "observation") == "observation"
        for hit in observation_hits
    )
    assert list((tmp_path / "observations").rglob("*.md"))
    assert list((tmp_path / "experiences").rglob("l1.json"))


def test_experience_retrieval_takes_no_observation_filters() -> None:
    """`memory_type`/`scope` must not be reachable on the experience channel.

    Every `E-*` record is stored semantic/project. The merged retriever accepted
    both arguments anyway, so `memory_type=procedural` -- the natural pairing for
    a "how do I do X" query, and one the old tool schema actively suggested --
    silently returned zero of the library's records with no signal that a filter
    had emptied the result set.
    """
    parameters = inspect.signature(search_experience_records).parameters
    assert "memory_type" not in parameters
    assert "scope" not in parameters
    assert {"topic", "method", "task", "discipline", "domain", "level"} <= set(
        parameters
    )


def test_experience_search_survives_process_shaped_facets(tmp_path: Path) -> None:
    """A process-shaped query must not be able to empty the library silently.

    The failure being guarded is not "returns nothing" -- a lexical miss is
    legitimate -- but returning nothing *because a filter was applied*, which is
    indistinguishable to the caller from an empty library.
    """
    _store(tmp_path, project_id="P-alpha", marker="contrastive catalyst screening")

    assert search_experience_records(
        memory_dir=tmp_path, project_id="P-alpha", topic="contrastive catalyst"
    )
    # Facet-only browse through the search entry point still lists records.
    assert search_experience_records(memory_dir=tmp_path, project_id="P-alpha")
    # An unknown discipline filters honestly to empty rather than by accident.
    assert not search_experience_records(
        memory_dir=tmp_path,
        project_id="P-alpha",
        topic="contrastive catalyst",
        discipline="bio",
    )


def test_experience_facets_browse_and_page(tmp_path: Path) -> None:
    _store(tmp_path, project_id="P-alpha", marker="contrastive catalyst screening")

    disciplines = browse_experience_facets(
        memory_dir=tmp_path, project_id="P-alpha", facet="discipline"
    )
    assert disciplines["facet"] == "discipline"
    assert disciplines["total_records"] == 2
    assert sum(value["records"] for value in disciplines["values"]) == 2

    records = browse_experience_facets(
        memory_dir=tmp_path, project_id="P-alpha", facet="records", limit=1
    )
    assert records["facet"] == "records"
    assert records["total_records"] == 2
    assert len(records["records"]) == 1

    second = browse_experience_facets(
        memory_dir=tmp_path,
        project_id="P-alpha",
        facet="records",
        limit=1,
        offset=1,
    )
    assert second["offset"] == 1
    assert second["records"][0]["id"] != records["records"][0]["id"]

    l1_only = browse_experience_facets(
        memory_dir=tmp_path, project_id="P-alpha", facet="records", level="l1"
    )
    assert [record["level"] for record in l1_only["records"]] == ["l1"]


def test_experience_summary_leads_with_subject_facets(tmp_path: Path) -> None:
    """`summary` carries 3x the ranking weight of `body`, so facets belong in it.

    `domain` and `task` are the record's own subject-matter vocabulary and used
    to live only in the JSON body at weight 1.0, while the summary held the
    opening characters of `statement` -- prose framing a problem rather than
    naming the subject it is about.
    """
    store_paper_experiences(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id="2401.01234",
        url="https://arxiv.org/abs/2401.01234",
        title="Catalyst Study",
        paper_text="full paper",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
        payloads={
            "l1": _new_payload("2401.01234", "L1 statement", level="l1"),
            "l2": _new_payload("2401.01234", "L2 statement", level="l2"),
        },
        domain_arxiv="cs.AI",
    )

    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert documents
    for document in documents:
        assert document.domain == "agent_learning"
        assert document.task == "test task"
        # Facets first, then the descriptive tail: what the record is ABOUT is
        # what the high-weight field should carry.
        assert document.summary.startswith("agent_learning · test task · ")
        assert document.discipline == "cs"
        assert document.paper_key


def test_legacy_records_without_facets_stay_retrievable(tmp_path: Path) -> None:
    """Records predating the facet fields must not drop out of the library.

    The facets are optional on purpose. A record written before they existed has
    no `domain`/`task` and no `discipline`, and the fallback puts it in `other`
    rather than excluding it -- an unreachable record is worse than a coarsely
    filed one.
    """
    _store(tmp_path, project_id="P-alpha", marker="contrastive catalyst screening")

    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert documents
    assert {document.discipline for document in documents} == {"other"}
    assert all(document.domain is None for document in documents)
    assert all(document.summary for document in documents)
    assert search_experience_records(
        memory_dir=tmp_path, project_id="P-alpha", topic="contrastive catalyst"
    )
    browsed = browse_experience_facets(
        memory_dir=tmp_path, project_id="P-alpha", facet="discipline"
    )
    assert browsed["values"] == [{"value": "other", "records": 2}]


def test_experience_ids_follow_content_not_position(tmp_path: Path) -> None:
    """An `E-*` ID must name one claim, even when re-extraction reorders records.

    The ID used to hash the record's index in its level's array. Re-extracting a
    paper reorders or re-counts records, so the same finding took a new ID and a
    different finding inherited the old one. Two things read an ID as meaning one
    specific claim: the policy cache keyed on `(task, sorted(selected_ids))`, and
    the audit trail from a policy line back to its supporting record. Under drift
    the cache serves a policy synthesized from other records.
    """
    first = _new_payload("2401.01234", "first finding", level="l1")
    second = _new_payload("2401.01234", "second finding", level="l1")
    both = dict(first)
    both["experiences"] = [
        first["experiences"][0],  # type: ignore[index]
        second["experiences"][0],  # type: ignore[index]
    ]
    reversed_order = dict(first)
    reversed_order["experiences"] = list(reversed(both["experiences"]))  # type: ignore[arg-type]

    def store_and_map(payload: dict[str, object]) -> dict[str, str]:
        store_paper_experiences(
            memory_dir=tmp_path,
            project_id="P-alpha",
            paper_id="2401.01234",
            url="https://arxiv.org/abs/2401.01234",
            title="Catalyst Study",
            paper_text="full paper",
            prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
            payloads={
                "l1": payload,
                "l2": _new_payload("2401.01234", "L2 statement", level="l2"),
            },
            domain_arxiv="cs.AI",
        )
        documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
        return {
            document.observation_id: document.body
            for document in documents
            if document.experience_level == "l1"
        }

    before = store_and_map(both)
    after = store_and_map(reversed_order)

    assert len(before) == 2
    # Same two claims, so the same two IDs -- order in the payload is not part
    # of a record's identity.
    assert set(before) == set(after)
    for record_id, body in before.items():
        statement_before = "first finding" if "first finding" in body else "second"
        assert statement_before in after[record_id]


def test_experience_parse_cache_reuses_documents_until_the_file_changes(
    tmp_path: Path,
) -> None:
    """Re-reading and re-parsing every level file per query does not scale.

    `experience_library_stats` runs on every model request, so the parse cost is
    paid on the hot path. The cache is validated on mtime and size rather than
    on a timer, so a rewritten file is still picked up -- correctness does not
    depend on the library being small.
    """
    from EvoScientist.memory.experiences.store import reset_experience_parse_cache

    _store(tmp_path, project_id="P-alpha", marker="cached catalyst lesson")
    reset_experience_parse_cache()

    first = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    second = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert first
    assert len(first) == len(second)
    # Same objects, not merely equal ones: the parse was skipped entirely.
    assert all(a is b for a, b in zip(first, second, strict=True))

    store_paper_experiences(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id="2401.01234",
        url="https://arxiv.org/abs/2401.01234",
        title="Catalyst Study",
        paper_text="full paper",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
        payloads={
            "l1": _payload("2401.01234", "rewritten catalyst lesson"),
            "l2": _payload("2401.01234", "rewritten catalyst claim"),
        },
    )
    third = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert any("rewritten catalyst lesson" in document.body for document in third)


def test_ranked_search_memoizes_document_tokens_without_changing_scores(
    tmp_path: Path,
) -> None:
    """Tokenizing is corpus-wide work repeated per facet pass, so it is memoized.

    Every document is tokenized to compute IDF whether or not it can score, and
    facet fusion multiplied that by the number of facets supplied. The memo must
    be invisible in the results: it is a cache of the document's own fields, so
    the tokens it hands back have to equal a fresh tokenization.
    """
    from EvoScientist.memory.search import _tokens, document_token_index

    _store(tmp_path, project_id="P-alpha", marker="memoized catalyst lesson")
    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert documents

    for document in documents:
        index = document_token_index(document)
        assert index.id_tokens == frozenset(_tokens(document.observation_id))
        assert index.summary_tokens == frozenset(_tokens(document.summary))
        assert index.body_tokens == frozenset(_tokens(document.body))
        assert index.all_tokens >= index.summary_tokens | index.body_tokens
        # Second call returns the memo rather than rebuilding it.
        assert document_token_index(document) is index

    # The memo is not part of the record's identity, so it must not leak into
    # equality -- a document is the same document before and after being scored.
    assert documents[0] == documents[0]
    assert search_experience_records(
        memory_dir=tmp_path, project_id="P-alpha", topic="memoized catalyst"
    )


def test_experience_catalog_is_webui_visible_and_derived(tmp_path: Path) -> None:
    _store(tmp_path, project_id="P-alpha", marker="visible catalyst lesson")

    catalog = experience_catalog_path(memory_dir=tmp_path, project_id="P-alpha")
    content = catalog.read_text(encoding="utf-8")
    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")

    assert catalog == (
        tmp_path / "profile" / "projects" / "P-alpha" / EXPERIENCE_CATALOG_FILENAME
    )
    assert "# Paper experiences" in content
    assert "## Catalyst Study" in content
    assert "https://arxiv.org/abs/2401.01234" in content
    assert "visible catalyst lesson" in content
    catalog_ids = {
        part.strip("`:") for part in content.split() if part.startswith("`E-")
    }
    assert {document.observation_id for document in documents} <= catalog_ids


def test_experience_catalog_uses_new_statement_field(tmp_path: Path) -> None:
    store_paper_experiences(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id="paper-new",
        url="https://example.test/paper-new",
        title="New Schema Paper",
        paper_text="full paper",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
        payloads={
            "l1": _new_payload("paper-new", "L1 statement summary", level="l1"),
            "l2": _new_payload("paper-new", "L2 statement summary", level="l2"),
        },
    )
    content = experience_catalog_path(
        memory_dir=tmp_path, project_id="P-alpha"
    ).read_text(encoding="utf-8")
    assert "L1 statement summary" in content
    assert "L2 statement summary" in content


def test_parser_rejects_malformed_current_schema() -> None:
    payload = _llm_payload("paper-new", "statement", level="l1")
    del payload["experiences"][0]["evidence"]
    with pytest.raises(ExperienceOutputError, match="keys mismatch"):
        parse_experience_json(json.dumps(payload), level="l1", paper_id="paper-new")


def test_parser_injects_runtime_fields_into_current_schema() -> None:
    parsed = parse_experience_json(
        json.dumps(_llm_payload("paper-new", "statement", level="l1")),
        level="l1",
        paper_id="paper-new",
        domain_arxiv="cs.AI",
    )
    item = parsed["experiences"][0]
    assert item["id"] == "l1_paper_new_01"
    assert item["layer"] == "L1"
    assert item["domain_arxiv"] == "cs.AI"
    assert item["utility"] is None
    assert 0 <= item["confidence"] <= 1
    assert item["evidence"][0]["source_id"] == "paper-new"


def test_strip_references_section_drops_heading_onward() -> None:
    body = "Intro text. " * 60 + "\n\n## Method\n\nMethod text. " * 20
    text = body + "\n\n## References\n\n[1] Some Author. Some Title. 2020."
    stripped = _strip_references_section(text)
    assert stripped == body.rstrip()
    assert "References" not in stripped
    assert "[1] Some Author" not in stripped


def test_strip_references_section_matches_numbered_bibliography_heading() -> None:
    body = "Intro text. " * 60
    text = body + "\n\n7 Bibliography\n\n[1] Some Author. Some Title. 2020."
    stripped = _strip_references_section(text)
    assert stripped == body.rstrip()


def test_strip_references_section_ignores_early_false_positive() -> None:
    text = "# References for X\n\nActual body text follows here without a real section."
    assert _strip_references_section(text) == text


def test_strip_references_section_no_match_returns_original() -> None:
    text = "Body text with no references section at all. " * 20
    assert _strip_references_section(text) == text


def test_experience_catalog_rebuilds_without_duplicate_papers(tmp_path: Path) -> None:
    _store(tmp_path, project_id="P-alpha", marker="first extraction")
    _store(tmp_path, project_id="P-alpha", marker="replacement extraction")

    content = experience_catalog_path(
        memory_dir=tmp_path, project_id="P-alpha"
    ).read_text(encoding="utf-8")

    assert content.count("## Catalyst Study") == 1
    assert "replacement extraction" in content
    assert "first extraction" not in content


def test_experience_catalog_remains_project_isolated(tmp_path: Path) -> None:
    _store(tmp_path, project_id="P-alpha", marker="alpha-only lesson")
    _store(tmp_path, project_id="P-beta", marker="beta-only lesson")

    alpha = experience_catalog_path(
        memory_dir=tmp_path, project_id="P-alpha"
    ).read_text(encoding="utf-8")
    beta = experience_catalog_path(memory_dir=tmp_path, project_id="P-beta").read_text(
        encoding="utf-8"
    )

    assert "alpha-only lesson" in alpha
    assert "beta-only lesson" not in alpha
    assert "beta-only lesson" in beta
    assert "alpha-only lesson" not in beta


def test_existing_experience_catalogs_are_backfilled(tmp_path: Path) -> None:
    _store(tmp_path, project_id="P-alpha", marker="alpha backfill")
    _store(tmp_path, project_id="P-beta", marker="beta backfill")
    alpha_catalog = experience_catalog_path(memory_dir=tmp_path, project_id="P-alpha")
    beta_catalog = experience_catalog_path(memory_dir=tmp_path, project_id="P-beta")
    alpha_catalog.unlink()
    beta_catalog.unlink()

    refreshed = refresh_all_experience_catalogs(memory_dir=tmp_path)

    assert refreshed == [alpha_catalog, beta_catalog]
    assert "alpha backfill" in alpha_catalog.read_text(encoding="utf-8")
    assert "beta backfill" in beta_catalog.read_text(encoding="utf-8")


def test_experience_catalog_is_not_injected_as_profile_context(tmp_path: Path) -> None:
    memories = tmp_path / "memories"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    middleware = memory_middleware.create_memory_middleware(
        str(memories), workspace_dir=str(workspace)
    )
    _store(
        memories,
        project_id=middleware.project_id,
        marker="catalog-only experience",
    )

    profile_context = middleware._read_profile_memory()

    assert EXPERIENCE_CATALOG_FILENAME not in profile_context
    assert "catalog-only experience" not in profile_context


def test_enqueue_is_idempotent_across_calls(tmp_path: Path) -> None:
    first, created = enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id="arXiv:2401.01234v2",
        url="https://arxiv.org/pdf/2401.01234v2.pdf",
        title="Paper",
    )
    second, created_again = enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id="arXiv:2401.01234v2",
        url="https://arxiv.org/pdf/2401.01234v2.pdf",
        title="Paper",
    )

    assert created is True
    assert created_again is False
    assert first.task_id == second.task_id
    assert (
        len(list_tasks(memory_dir=tmp_path, project_id="P-alpha", status="pending"))
        == 1
    )


def test_enqueue_tool_batches_and_launches_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launches: list[str] = []
    monkeypatch.setattr(
        "EvoScientist.tools.paper_experience_queue.launch_paper_experience_worker",
        lambda project_id: launches.append(project_id) or object(),
    )
    tool = create_paper_experience_queue_tool(memory_dir=tmp_path, project_id="P-alpha")

    result = json.loads(
        tool.run(
            {
                "papers": [
                    {
                        "url": "https://example.test/one",
                        "paper_id": "paper-1",
                        "title": "One",
                    },
                    {
                        "url": "https://example.test/two",
                        "paper_id": "paper-2",
                        "title": "Two",
                    },
                ]
            }
        )
    )

    assert result["enqueued"] == 2
    assert result["worker_launched"] is True
    assert launches == ["P-alpha"]


@pytest.mark.asyncio
async def test_active_extraction_returns_and_stores_experiences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def run_sync(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def download(_url: str) -> str:
        return "full paper text"

    async def extract(**_kwargs):
        return {
            "l1": _payload("paper-1", "active l1 result"),
            "l2": _payload("paper-1", "active l2 result"),
        }

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

    result = json.loads(
        await tool.arun(
            {
                "papers": [
                    {
                        "url": "https://example.test/paper",
                        "paper_id": "paper-1",
                        "title": "Paper One",
                    }
                ]
            }
        )
    )

    assert result["failed"] == []
    assert result["completed"][0]["cached"] is False
    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert {document.experience_level for document in documents} == {"l1", "l2"}


@pytest.mark.asyncio
async def test_active_extraction_reuses_project_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store(tmp_path, project_id="P-alpha", marker="cached experience")
    downloads = 0

    async def run_sync(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    async def download(_url: str) -> str:
        nonlocal downloads
        downloads += 1
        # Reached only by the full-text backfill, never by re-extraction.
        return "## Method\n\nBackfilled full text for the cached paper.\n"

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

    result = json.loads(
        await tool.arun(
            {
                "papers": [
                    {
                        "url": "https://arxiv.org/abs/2401.01234",
                        "paper_id": "2401.01234",
                        "title": "Catalyst Study",
                    }
                ]
            }
        )
    )

    # Cache hit: no extraction, but the full text is backfilled once.
    assert downloads == 1
    assert result["completed"][0]["cached"] is True
    assert "cached experience" in json.dumps(result, ensure_ascii=False)


@pytest.mark.asyncio
async def test_worker_graph_drains_empty_project(tmp_path: Path) -> None:
    graph = paper_experience_worker.build_paper_experience_worker_graph(
        memory_dir=tmp_path
    )
    result = await graph.ainvoke(
        {}, {"configurable": {"evomemory_project_id": "P-alpha"}}
    )
    assert result == {"processed": 0, "failed": 0}


@pytest.mark.asyncio
async def test_l1_and_l2_extraction_runs_concurrently() -> None:
    both_started = asyncio.Event()
    started = 0

    class Model:
        async def ainvoke(self, messages):
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            level = "l1" if messages[0].content == "l1 prompt" else "l2"
            return SimpleNamespace(
                content=json.dumps(_payload("paper-1", f"{level} result"))
            )

    result = await run_experience_extraction(
        paper_id="paper-1",
        paper_text="paper text",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
        model=Model(),
    )

    assert started == 2
    assert result["l1"]["experiences"][0]["narrative"] == "l1 result"
    assert result["l2"]["experiences"][0]["narrative"] == "l2 result"


@pytest.mark.asyncio
async def test_current_prompt_outputs_are_normalized_by_runtime() -> None:
    class Model:
        async def ainvoke(self, messages):
            level = "l1" if messages[0].content == "l1 prompt" else "l2"
            return SimpleNamespace(
                content=json.dumps(
                    _llm_payload("paper-1", f"{level} result", level=level)
                )
            )

    result = await run_experience_extraction(
        paper_id="paper-1",
        paper_text="paper text",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
        model=Model(),
        domain_arxiv="cs.AI",
    )

    assert result["l1"]["experiences"][0]["layer"] == "L1"
    assert result["l2"]["experiences"][0]["layer"] == "L2"
    assert result["l2"]["experiences"][0]["claim_type"] == "conditional"
    assert result["l1"]["experiences"][0]["evidence"][0]["source_id"] == "paper-1"


@pytest.mark.asyncio
async def test_extraction_uses_auxiliary_model_when_model_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class AuxiliaryModel:
        async def ainvoke(self, _messages):
            nonlocal calls
            calls += 1
            return SimpleNamespace(content=json.dumps(_payload("paper-1", "result")))

    auxiliary = AuxiliaryModel()
    monkeypatch.setattr(
        "EvoScientist.EvoScientist._ensure_auxiliary_chat_model",
        lambda: auxiliary,
    )

    result = await run_experience_extraction(
        paper_id="paper-1",
        paper_text="paper text",
        prompts={"l1": "l1 prompt", "l2": "l2 prompt"},
    )

    assert calls == 2
    assert result["l1"]["paper_id"] == "paper-1"


@pytest.mark.asyncio
async def test_worker_marks_download_failure_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        url="https://example.invalid/paper",
    )

    async def fail_download(_url: str) -> str:
        raise RuntimeError("download unavailable")

    monkeypatch.setattr(paper_experience_worker, "download_paper_text", fail_download)
    result = await paper_experience_worker.drain_paper_experience_queue(
        memory_dir=tmp_path,
        project_id="P-alpha",
        model=object(),
    )

    assert result == {"processed": 0, "failed": 1}
    failed = list_tasks(memory_dir=tmp_path, project_id="P-alpha", status="failed")
    assert len(failed) == 1
    assert failed[0].error == "download unavailable"

    retried, created = enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        url="https://example.invalid/paper",
    )
    assert created is True
    assert retried.status == "pending"
    assert list_tasks(memory_dir=tmp_path, project_id="P-alpha", status="failed") == []


@pytest.mark.asyncio
async def test_worker_stores_success_and_completes_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id="paper-1",
        url="https://example.test/paper",
        title="Paper One",
    )

    async def download(_url: str) -> str:
        return "full paper text"

    async def extract(**_kwargs):
        return {
            "l1": _payload("paper-1", "l1 worker result"),
            "l2": _payload("paper-1", "l2 worker result"),
        }

    monkeypatch.setattr(paper_experience_worker, "download_paper_text", download)
    monkeypatch.setattr(
        paper_experience_worker,
        "load_experience_prompts",
        lambda: {"l1": "l1 prompt", "l2": "l2 prompt"},
    )
    monkeypatch.setattr(paper_experience_worker, "run_experience_extraction", extract)

    result = await paper_experience_worker.drain_paper_experience_queue(
        memory_dir=tmp_path,
        project_id="P-alpha",
        model=object(),
    )

    assert result == {"processed": 1, "failed": 0}
    assert (
        len(list_tasks(memory_dir=tmp_path, project_id="P-alpha", status="completed"))
        == 1
    )
    documents = list_experience_documents(memory_dir=tmp_path, project_id="P-alpha")
    assert {document.experience_level for document in documents} == {"l1", "l2"}


@pytest.mark.asyncio
async def test_worker_recovers_interrupted_running_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    enqueue_paper(
        memory_dir=tmp_path,
        project_id="P-alpha",
        paper_id="paper-1",
        url="https://example.test/paper",
    )
    claimed = claim_next_task(memory_dir=tmp_path, project_id="P-alpha")
    assert claimed is not None

    async def download(_url: str) -> str:
        return "full paper text"

    async def extract(**_kwargs):
        return {
            "l1": _payload("paper-1", "recovered l1"),
            "l2": _payload("paper-1", "recovered l2"),
        }

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
    completed = list_tasks(
        memory_dir=tmp_path, project_id="P-alpha", status="completed"
    )
    assert completed[0].attempts == 2
