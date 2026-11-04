"""S2 is off by default in this build: a missing key must fail fast, not retry.

Without a key Semantic Scholar throttles hard; the old behaviour was to sit
through four 429 retries (about 45 s per search) before falling back to arXiv.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "paper-navigator" / "scripts"


@pytest.fixture
def pn(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in ("utils", "scholar_search"):
        sys.modules.pop(name, None)
    import scholar_search
    import utils

    yield utils, scholar_search
    for name in ("utils", "scholar_search"):
        sys.modules.pop(name, None)


def test_s2_call_without_key_fails_fast(pn, monkeypatch) -> None:
    utils, _ = pn
    monkeypatch.delenv("S2_API_KEY", raising=False)
    start = time.monotonic()
    with httpx.Client() as client, pytest.raises(utils.S2NotConfigured):
        utils.request_with_retry(
            client, f"{utils.S2_BASE}/paper/search", {"query": "x"}
        )
    assert time.monotonic() - start < 1.0


def test_s2_not_configured_is_a_rate_limit_for_existing_fallbacks(pn) -> None:
    utils, _ = pn
    assert issubclass(utils.S2NotConfigured, utils.RateLimitExhausted)


def test_search_falls_back_to_arxiv_without_key(pn, monkeypatch, capsys) -> None:
    _, scholar_search = pn
    monkeypatch.delenv("S2_API_KEY", raising=False)
    monkeypatch.setattr(
        scholar_search,
        "_fallback_arxiv_search",
        lambda *a, **k: [{"title": "from-arxiv"}],
    )
    assert scholar_search.search("duorc") == [{"title": "from-arxiv"}]
    assert "not configured" in capsys.readouterr().err


def test_non_s2_urls_are_unaffected(pn, monkeypatch) -> None:
    utils, _ = pn
    monkeypatch.delenv("S2_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert utils.request_with_retry(client, "https://example.org/x") == {"ok": True}


def test_config_has_no_s2_key() -> None:
    from EvoScientist.config.settings import EvoScientistConfig

    assert not hasattr(EvoScientistConfig(), "s2_api_key")


# ── No key: degrade at once, never wait ───────────────────────────────


@pytest.fixture
def readers(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.delenv("S2_API_KEY", raising=False)
    for name in ("utils", "fetch_paper", "download_paper"):
        sys.modules.pop(name, None)
    import download_paper
    import fetch_paper
    import utils

    def no_http(*a, **k):
        raise AssertionError("no HTTP call is expected without a key")

    monkeypatch.setattr(fetch_paper, "request_with_retry", no_http)
    monkeypatch.setattr(download_paper, "request_with_retry", no_http)
    yield utils, fetch_paper, download_paper
    for name in ("utils", "fetch_paper", "download_paper"):
        sys.modules.pop(name, None)


def test_fetch_by_arxiv_id_goes_straight_to_arxiv(readers) -> None:
    _, fetch_paper, _ = readers
    url, meta = fetch_paper.resolve_paper_url("ArXiv:1710.10723v2")
    assert url == "https://arxiv.org/abs/1710.10723"
    assert meta["externalIds"] == {"ArXiv": "1710.10723"}
    header = fetch_paper.format_metadata(meta)
    assert "Citations: n/a" in header and "arXiv: `1710.10723`" in header


def test_fetch_by_arxiv_url_goes_straight_to_arxiv(readers) -> None:
    _, fetch_paper, _ = readers
    url, _ = fetch_paper.resolve_paper_url("https://arxiv.org/pdf/1804.07927v1.pdf")
    assert url == "https://arxiv.org/abs/1804.07927"


def test_download_by_arxiv_id_picks_the_arxiv_pdf(readers) -> None:
    _, _, download_paper = readers
    meta = download_paper.resolve_metadata("arxiv:2101.06326")
    assert download_paper.choose_pdf_url(meta) == (
        "https://arxiv.org/pdf/2101.06326.pdf",
        "2101.06326",
    )


def test_ids_only_s2_can_resolve_fail_at_once_with_a_hint(readers) -> None:
    utils, fetch_paper, _ = readers
    with pytest.raises(utils.S2NotConfigured, match="arXiv ID, a DOI or a URL"):
        fetch_paper.resolve_paper_url("CorpusId:12345")


def test_with_a_key_the_s2_lookup_is_still_used(readers, monkeypatch) -> None:
    _, fetch_paper, _ = readers
    monkeypatch.setenv("S2_API_KEY", "k")
    calls = []

    def fake(client, url, params=None, headers=None, **kw):
        calls.append(url)
        return {"paperId": "p", "externalIds": {"ArXiv": "1"}, "title": "t"}

    monkeypatch.setattr(fetch_paper, "request_with_retry", fake)
    url, _ = fetch_paper.resolve_paper_url("ArXiv:1")
    assert calls and "semanticscholar" in calls[0]
    assert url == "https://arxiv.org/abs/1"


@pytest.mark.parametrize(
    "script,args",
    [
        ("trending.py", ["--query", "x"]),
        ("match_paper_by_title.py", ["--title", "x"]),
    ],
)
def test_s2_only_scripts_exit_with_one_line_not_a_traceback(script, args) -> None:
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "S2_API_KEY"}
    start = time.monotonic()
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=SCRIPTS,
        timeout=30,
    )
    assert time.monotonic() - start < 10
    assert done.returncode == 3, done.stderr
    assert "Traceback" not in done.stderr
    assert "no S2_API_KEY" in done.stderr
