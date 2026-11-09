"""download_paper_text: the shared full-text chain and its fallbacks.

Order under test: MinerU precise API (token) -> arXiv PDF parsed locally ->
Jina Reader -> direct text/markdown. Every attempt is recorded, so the tests
also assert that a failure in one backend is not hidden by another.
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from EvoScientist.memory.experiences import extraction, fulltext

BODY = "## Method\n\n" + "We train a reader with hard negatives. " * 30
PDF_BYTES = b"%PDF-1.7 fake pdf body"


def _patch_client(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(fulltext.httpx, "AsyncClient", factory)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(fulltext.asyncio, "sleep", no_sleep)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "MINERU_API_KEY",
        "EVOSCIENTIST_MINERU_AGENT_API",
        "EVOSCIENTIST_PAPER_FULLTEXT_BACKEND",
        "JINA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def _zip_with_markdown(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("full.md", text)
    return buffer.getvalue()


def test_arxiv_pdf_url():
    assert fulltext.arxiv_pdf_url("https://arxiv.org/abs/1804.07927") == (
        "https://arxiv.org/pdf/1804.07927"
    )
    assert fulltext.arxiv_pdf_url("https://arxiv.org/pdf/1804.07927.pdf") == (
        "https://arxiv.org/pdf/1804.07927"
    )
    assert fulltext.arxiv_pdf_url("https://example.test/paper") == ""


@pytest.mark.asyncio
async def test_arxiv_pdf_is_parsed_locally_before_jina(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        assert request.url.host == "arxiv.org"
        return httpx.Response(200, content=PDF_BYTES)

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(fulltext, "pdf_bytes_to_text", lambda data: BODY)
    text = await extraction.download_paper_text("https://arxiv.org/abs/1804.07927")
    assert "hard negatives" in text
    assert seen == ["arxiv.org"]  # Jina never contacted


@pytest.mark.asyncio
async def test_falls_back_to_jina_when_the_pdf_is_unavailable(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "arxiv.org":
            return httpx.Response(404)
        assert request.url.host == "r.jina.ai"
        return httpx.Response(200, text=BODY)

    _patch_client(monkeypatch, handler)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "jina"
    assert [a["backend"] for a in result.attempts] == ["arxiv-pdf", "jina"]
    assert seen == ["arxiv.org", "r.jina.ai"]


@pytest.mark.asyncio
async def test_html_is_never_accepted(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(403)
        return httpx.Response(200, text="<html>blocked</html>", headers={"content-type": "text/html"})

    _patch_client(monkeypatch, handler)
    with pytest.raises(fulltext.FulltextError) as excinfo:
        await fulltext.download_fulltext("https://example.test/paper")
    detail = str(excinfo.value)
    assert "jina" in detail
    assert "refusing HTML" in detail
    assert [a["backend"] for a in excinfo.value.attempts] == ["jina", "direct"]


@pytest.mark.asyncio
async def test_mineru_precise_api_wins_when_a_token_is_set(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    zip_bytes = _zip_with_markdown(BODY)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "arxiv.org":
            return httpx.Response(200, content=PDF_BYTES)
        if request.url.path == "/api/v4/extract/task" and request.method == "POST":
            assert request.headers["authorization"] == "Bearer test-token"
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "t1"}})
        if request.url.path == "/api/v4/extract/task/t1":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "state": "done",
                        "full_zip_url": "https://cdn-mineru.openxlab.org.cn/out.zip",
                    },
                },
            )
        if request.url.host == "cdn-mineru.openxlab.org.cn":
            return httpx.Response(200, content=zip_bytes)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    _patch_client(monkeypatch, handler)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "mineru"
    assert "hard negatives" in result.text
    assert [a["backend"] for a in result.attempts] == ["arxiv-pdf", "mineru"]


@pytest.mark.asyncio
async def test_mineru_failure_still_returns_the_local_pdf(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "arxiv.org":
            return httpx.Response(200, content=PDF_BYTES)
        if request.url.host == "mineru.net":
            return httpx.Response(500)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(fulltext, "pdf_bytes_to_text", lambda data: BODY)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "pymupdf"
    assert result.attempts[1]["backend"] == "mineru"
    assert "error" in result.attempts[1]


@pytest.mark.asyncio
async def test_short_text_is_rejected(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "arxiv.org":
            return httpx.Response(200, content=PDF_BYTES)
        raise AssertionError("Jina should not be contacted")

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(fulltext, "pdf_bytes_to_text", lambda data: "too short")
    with pytest.raises(ValueError, match="unexpectedly short"):
        await extraction.download_paper_text("https://arxiv.org/abs/1804.07927")
