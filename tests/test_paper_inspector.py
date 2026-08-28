"""Tests for the /debug/papers chunk-store inspector.

The Starlette app is exercised directly through TestClient -- the point is the
handlers and the rendering, not langgraph dev's hosting of them.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from EvoScientist.langgraph_dev.http import app
from EvoScientist.memory.papers.store import paper_storage_key, store_paper_text

# Each section is padded past MIN_CHUNK_CHARS so the text really splits into
# several chunks under distinct section paths. A short paper collapses into one
# chunk headed by the title alone, which would not exercise the rendering.
_FILLER = "Supporting detail sentence that pads the section body. " * 8

PAPER_TEXT = (
    "# Contrastive Pretraining for Catalysts\n\n"
    f"Intro paragraph about the setting. {_FILLER}\n\n"
    "## Method\n\n"
    "We pretrain with a contrastive objective over reaction graphs. "
    f"{_FILLER}\n\n"
    "### Regularisation\n\n"
    "Dropout of 0.1 <script>alert(1)</script> keeps the encoder honest. "
    f"{_FILLER}\n\n"
    "## Results\n\n"
    "Accuracy improves by four points over the supervised baseline. "
    f"{_FILLER}\n"
)

PROJECT = "P-1234567890abcdef"
PAPER_ID = "arXiv:2401.00001"
PAPER_URL = "https://arxiv.org/abs/2401.00001"


@pytest.fixture
def store(tmp_path: Path):
    """Seed one stored paper and point the inspector at that directory."""
    store_paper_text(
        memory_dir=tmp_path,
        project_id=PROJECT,
        paper_id=PAPER_ID,
        url=PAPER_URL,
        title="Contrastive Pretraining for Catalysts",
        paper_text=PAPER_TEXT,
    )
    with patch(
        "EvoScientist.langgraph_dev.paper_inspector._memories_dir",
        return_value=tmp_path,
    ):
        yield tmp_path


@pytest.fixture
def client():
    return TestClient(app)


def test_overview_lists_the_project_and_paper(client, store):
    resp = client.get("/debug/papers")
    assert resp.status_code == 200
    body = resp.text
    assert PROJECT in body
    assert "Contrastive Pretraining for Catalysts" in body
    # Chunk counts are the whole point: they prove ingestion ran.
    assert str(len(PAPER_TEXT)) in body


def test_overview_says_so_when_the_store_is_empty(client, tmp_path):
    with patch(
        "EvoScientist.langgraph_dev.paper_inspector._memories_dir",
        return_value=tmp_path / "nothing-here",
    ):
        resp = client.get("/debug/papers")
    assert resp.status_code == 200
    # An empty page would read as a bug in the inspector rather than an empty
    # store, so the distinction is spelled out.
    assert "No paper full text stored yet" in resp.text


def test_paper_view_shows_chunk_rows_and_text(client, store):
    key = paper_storage_key(PAPER_ID, PAPER_URL)
    resp = client.get(f"/debug/papers?project={PROJECT}&paper={key}")
    assert resp.status_code == 200
    body = resp.text
    assert "contrastive objective over reaction graphs" in body
    assert "Method &gt; Regularisation" in body
    assert "C-" in body


def test_paper_text_is_escaped_not_interpolated(client, store):
    """Stored text comes from a downloaded paper: never trusted as markup."""
    key = paper_storage_key(PAPER_ID, PAPER_URL)
    body = client.get(f"/debug/papers?project={PROJECT}&paper={key}").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body


def test_unknown_paper_returns_404(client, store):
    resp = client.get(f"/debug/papers?project={PROJECT}&paper=not-a-paper")
    assert resp.status_code == 404
    assert "No stored paper matched" in resp.text


def test_json_overview_reports_the_switch_and_counts(client, store):
    resp = client.get("/debug/papers.json")
    assert resp.status_code == 200
    body = resp.json()
    assert "fulltext_enabled" in body
    project = next(p for p in body["projects"] if p["project_id"] == PROJECT)
    assert project["paper_count"] == 1
    assert project["chunk_total"] >= 1
    assert project["papers"][0]["chunk_count"] == project["chunk_total"]


def test_json_paper_view_omits_chunk_text(client, store):
    """Metadata for scripted checks; text stays on the page."""
    key = paper_storage_key(PAPER_ID, PAPER_URL)
    body = client.get(f"/debug/papers.json?project={PROJECT}&paper={key}").json()
    assert body["chunks"]
    for chunk in body["chunks"]:
        assert "text" not in chunk
        assert chunk["text_chars"] > 0
        assert chunk["chunk_id"].startswith("C-")


def test_the_overview_link_resolves_to_the_paper_view(client, store):
    """The rendered href is the only way in, so it has to round-trip."""
    body = client.get("/debug/papers").text
    match = re.search(r'href="(/debug/papers\?project=[^"]+)"', body)
    assert match, body
    link = html.unescape(match.group(1))
    resp = client.get(link)
    assert resp.status_code == 200
    assert "contrastive objective over reaction graphs" in resp.text


def test_a_disabled_switch_still_shows_what_was_stored(client, store):
    """Turning ingestion off does not hide already-stored text."""
    with patch(
        "EvoScientist.langgraph_dev.paper_inspector._fulltext_enabled",
        return_value=False,
    ):
        resp = client.get("/debug/papers")
    assert resp.status_code == 200
    assert "DISABLED" in resp.text
    assert "Contrastive Pretraining for Catalysts" in resp.text
