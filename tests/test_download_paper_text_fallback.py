"""download_paper_text: arXiv PDF fallback when Jina Reader is unreachable."""

from __future__ import annotations

import httpx
import pytest

from EvoScientist.memory.experiences import extraction

BODY = "## Method\n\n" + "We train a reader with hard negatives. " * 30


def _patch_client(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(extraction.httpx, "AsyncClient", factory)
    monkeypatch.setattr(extraction, "_JINA_RETRY_DELAYS_S", (0.0, 0.0))


def test_arxiv_pdf_url():
    assert extraction._arxiv_pdf_url("https://arxiv.org/abs/1804.07927") == (
        "https://arxiv.org/pdf/1804.07927"
    )
    assert extraction._arxiv_pdf_url("https://arxiv.org/pdf/1804.07927.pdf") == (
        "https://arxiv.org/pdf/1804.07927"
    )
    assert extraction._arxiv_pdf_url("https://example.test/paper") == ""


@pytest.mark.asyncio
async def test_falls_back_to_arxiv_pdf_when_jina_is_denied(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "r.jina.ai":
            return httpx.Response(403)
        return httpx.Response(200, content=b"%PDF-fake")

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(extraction, "_pdf_bytes_to_text", lambda data: BODY)
    text = await extraction.download_paper_text("https://arxiv.org/pdf/1804.07927")
    assert "hard negatives" in text
    assert seen[-1] == "https://arxiv.org/pdf/1804.07927"


@pytest.mark.asyncio
async def test_jina_success_does_not_touch_the_pdf(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "r.jina.ai"
        return httpx.Response(200, text=BODY)

    _patch_client(monkeypatch, handler)
    assert "hard negatives" in await extraction.download_paper_text(
        "https://arxiv.org/abs/1804.07927"
    )


@pytest.mark.asyncio
async def test_non_arxiv_url_keeps_the_old_error(monkeypatch):
    _patch_client(monkeypatch, lambda request: httpx.Response(403))
    with pytest.raises(httpx.HTTPStatusError):
        await extraction.download_paper_text("https://example.test/paper")
