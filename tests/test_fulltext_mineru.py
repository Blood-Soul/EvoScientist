"""MinerU path of the full-text chain: URL suffix, cleaning, transient-failure retry."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from EvoScientist.memory.experiences import fulltext as ft

RAW = """# Title

## 1 Intro

Plain paragraph with E<sup>2</sup> and H<sub>2</sub>O.

![](images/abc.jpg)

Figure 1: a caption that is real text.

<table><tr><td>Model</td><td>F1</td></tr><tr><td>ALBERT</td><td>76.4</td></tr></table>



## References

[1] Someone. A paper.
"""


def test_clean_drops_figures_flattens_tables_keeps_text():
    out = ft.clean_mineru_markdown(RAW)
    assert "![" not in out and "<table" not in out and "<sup" not in out
    assert "Figure 1: a caption that is real text." in out
    assert "E2 and H2O" in out
    assert "Model | F1" in out and "ALBERT | 76.4" in out
    assert "\n\n\n" not in out


def test_mineru_url_gets_pdf_suffix_only_when_missing():
    assert ft._mineru_pdf_url("https://arxiv.org/pdf/2101.06326") == "https://arxiv.org/pdf/2101.06326.pdf"
    assert ft._mineru_pdf_url("https://arxiv.org/pdf/2101.06326v2.pdf") == "https://arxiv.org/pdf/2101.06326v2.pdf"
    assert ft._mineru_pdf_url("https://example.org/a.pdf?x=1") == "https://example.org/a.pdf?x=1"


def test_retryable_classification():
    assert ft._mineru_retryable(RuntimeError("MinerU task failed: parsing failed, please try again later"))
    assert ft._mineru_retryable(TimeoutError("still running"))
    assert ft._mineru_retryable(httpx.ConnectError("boom"))
    assert not ft._mineru_retryable(RuntimeError("MinerU task failed: unsupported file type"))
    assert not ft._mineru_retryable(RuntimeError("MinerU submit rejected: token expired"))


def _run_chain(monkeypatch, outcomes):
    calls = {"n": 0}

    async def fake_precise(client, url, key, timeout_s):
        calls["n"] += 1
        outcome = outcomes[min(calls["n"] - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def fake_pdf(client, pdf_url):
        return b"%PDF-1.4 fake"

    async def no_sleep(_):
        return None

    monkeypatch.setenv("MINERU_API_KEY", "token")
    monkeypatch.setattr(ft, "_mineru_precise", fake_precise)
    monkeypatch.setattr(ft, "_fetch_pdf_bytes", fake_pdf)
    monkeypatch.setattr(ft, "pdf_bytes_to_text", lambda data: "local text " * 100)
    monkeypatch.setattr(ft.asyncio, "sleep", no_sleep)
    result = asyncio.run(ft.download_fulltext("https://arxiv.org/abs/2101.06326"))
    return result, calls["n"]


def test_transient_failure_is_retried_then_mineru_wins(monkeypatch):
    again = RuntimeError("MinerU task failed: parsing failed, please try again later")
    result, calls = _run_chain(monkeypatch, [again, "mineru text " * 100])
    assert result.backend == "mineru" and calls == 2
    assert any(a.get("retry") for a in result.attempts)


def test_permanent_failure_falls_back_without_retry(monkeypatch):
    result, calls = _run_chain(monkeypatch, [RuntimeError("MinerU task failed: unsupported file type")])
    assert result.backend == "pymupdf" and calls == 1


def test_persistent_transient_failure_gives_up_after_three_tries(monkeypatch):
    again = RuntimeError("MinerU task failed: please try again later")
    result, calls = _run_chain(monkeypatch, [again])
    assert result.backend == "pymupdf" and calls == 3


@pytest.fixture(autouse=True)
def _no_env_leak(monkeypatch):
    monkeypatch.delenv("EVOSCIENTIST_PAPER_FULLTEXT_BACKEND", raising=False)
    monkeypatch.delenv("EVOSCIENTIST_MINERU_AGENT_API", raising=False)
