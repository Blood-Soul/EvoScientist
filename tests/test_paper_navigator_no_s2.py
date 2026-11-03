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
