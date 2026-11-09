"""fetch_via_jina: shared full-text chain first, legacy Jina fallback second."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "skills" / "paper-navigator" / "scripts"


@pytest.fixture
def fp(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in ("utils", "fetch_paper"):
        sys.modules.pop(name, None)
    import fetch_paper

    yield fetch_paper
    for name in ("utils", "fetch_paper"):
        sys.modules.pop(name, None)


class _Shared:
    """Stand-in for the shared ``fulltext`` module."""

    def __init__(self, text: str | None = None, exc: Exception | None = None):
        self._text = text
        self._exc = exc

    def download_fulltext_sync(self, url):
        if self._exc is not None:
            raise self._exc

        class Result:
            backend = "pymupdf"

            def __init__(self, text):
                self.text = text

        return Result(self._text)


def test_shared_chain_is_preferred_and_truncated(fp, monkeypatch, capsys) -> None:
    body = "cookie banner " * 40 + "\n## Abstract\n" + "x" * 300
    monkeypatch.setattr(fp, "_load_shared_fulltext", lambda: _Shared(text=body))
    out = fp.fetch_via_jina("https://arxiv.org/abs/1710.10723", 100)
    assert out.startswith("## Abstract")
    assert "Truncated at 100" in out
    assert "Full text via pymupdf" in capsys.readouterr().err


def test_shared_chain_failure_is_surfaced(fp, monkeypatch) -> None:
    exc = RuntimeError("full-text fetch failed (arxiv-pdf: 404; jina: 402)")
    monkeypatch.setattr(fp, "_load_shared_fulltext", lambda: _Shared(exc=exc))
    with pytest.raises(RuntimeError, match=r"full-text chain failed.*402"):
        fp.fetch_via_jina("https://arxiv.org/abs/1710.10723")


def test_arxiv_pdf_url_from_abs_and_pdf_urls(fp) -> None:
    assert (
        fp._arxiv_pdf_url("https://arxiv.org/abs/1710.10723")
        == "https://arxiv.org/pdf/1710.10723"
    )
    assert (
        fp._arxiv_pdf_url("https://arxiv.org/pdf/1710.10723.pdf")
        == "https://arxiv.org/pdf/1710.10723"
    )
    assert (
        fp._arxiv_pdf_url("https://arxiv.org/abs/cs/0112017v2")
        == "https://arxiv.org/pdf/cs/0112017v2"
    )
    assert fp._arxiv_pdf_url("https://example.com/paper.pdf") == ""


class _Boom:
    """httpx.Client stand-in whose GET always fails like an unreachable host."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, *a, **k):
        import httpx

        raise httpx.ConnectError("jina down")


@pytest.fixture
def legacy(fp, monkeypatch):
    """Force the legacy Jina-then-pypdf path (standalone skill installs)."""
    monkeypatch.setattr(fp, "_load_shared_fulltext", lambda: None)
    return fp


def test_legacy_jina_failure_uses_local_pdf_for_arxiv(legacy, monkeypatch, capsys) -> None:
    monkeypatch.setattr(legacy.httpx, "Client", _Boom)
    monkeypatch.setattr(
        legacy, "fetch_arxiv_pdf_text", lambda url: "## Abstract\n" + "x" * 300
    )
    out = legacy.fetch_via_jina("https://arxiv.org/abs/1710.10723", 100)
    assert out.startswith("## Abstract")
    assert "Truncated at 100" in out
    assert "extracted locally" in capsys.readouterr().err


def test_legacy_systemexit_from_the_http_layer_also_falls_back(legacy, monkeypatch) -> None:
    class _Exit(_Boom):
        def get(self, *a, **k):
            raise SystemExit("Error: 403")

    monkeypatch.setattr(legacy.httpx, "Client", _Exit)
    monkeypatch.setattr(legacy, "fetch_arxiv_pdf_text", lambda url: "body " * 10)
    assert legacy.fetch_via_jina("https://arxiv.org/abs/1710.10723").startswith("body")


def test_legacy_jina_failure_on_other_hosts_still_raises(legacy, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("jina down")

    monkeypatch.setattr(legacy, "request_with_retry", boom)
    with pytest.raises(RuntimeError, match="jina down"):
        legacy.fetch_via_jina("https://example.com/paper.pdf")


def test_legacy_both_paths_failing_reports_both_errors(legacy, monkeypatch) -> None:
    def pdf_boom(url):
        raise RuntimeError("no pypdf")

    monkeypatch.setattr(legacy.httpx, "Client", _Boom)
    monkeypatch.setattr(legacy, "fetch_arxiv_pdf_text", pdf_boom)
    with pytest.raises(RuntimeError, match=r"jina down.*no pypdf"):
        legacy.fetch_via_jina("https://arxiv.org/abs/1710.10723")
