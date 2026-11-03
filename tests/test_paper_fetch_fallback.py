"""fetch_via_jina falls back to a local arXiv PDF extraction when Jina is unreachable."""

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


def test_jina_failure_uses_local_pdf_for_arxiv(fp, monkeypatch, capsys) -> None:
    monkeypatch.setattr(fp.httpx, "Client", _Boom)
    monkeypatch.setattr(
        fp, "fetch_arxiv_pdf_text", lambda url: "## Abstract\n" + "x" * 300
    )
    out = fp.fetch_via_jina("https://arxiv.org/abs/1710.10723", 100)
    assert out.startswith("## Abstract")
    assert "Truncated at 100" in out
    assert "extracted locally" in capsys.readouterr().err


def test_systemexit_from_the_http_layer_also_falls_back(fp, monkeypatch) -> None:
    class _Exit(_Boom):
        def get(self, *a, **k):
            raise SystemExit("Error: 403")

    monkeypatch.setattr(fp.httpx, "Client", _Exit)
    monkeypatch.setattr(fp, "fetch_arxiv_pdf_text", lambda url: "body " * 10)
    assert fp.fetch_via_jina("https://arxiv.org/abs/1710.10723").startswith("body")


def test_jina_failure_on_other_hosts_still_raises(fp, monkeypatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("jina down")

    monkeypatch.setattr(fp, "request_with_retry", boom)
    with pytest.raises(RuntimeError, match="jina down"):
        fp.fetch_via_jina("https://example.com/paper.pdf")


def test_both_paths_failing_reports_both_errors(fp, monkeypatch) -> None:
    def pdf_boom(url):
        raise RuntimeError("no pypdf")

    monkeypatch.setattr(fp.httpx, "Client", _Boom)
    monkeypatch.setattr(fp, "fetch_arxiv_pdf_text", pdf_boom)
    with pytest.raises(RuntimeError, match=r"jina down.*no pypdf"):
        fp.fetch_via_jina("https://arxiv.org/abs/1710.10723")
