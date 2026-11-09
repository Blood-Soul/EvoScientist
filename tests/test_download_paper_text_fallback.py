"""download_paper_text: the shared full-text chain and its fallbacks.

Order under test: Jina Reader (retried) -> MinerU precise API (token, one attempt)
-> arXiv PDF parsed locally -> direct text/markdown. Every attempt is recorded, so
the tests also assert that a failure in one backend is not hidden by another.
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
async def test_jina_is_tried_first_and_the_pdf_is_never_downloaded(monkeypatch):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        assert request.url.host == "r.jina.ai"
        return httpx.Response(200, text=BODY)

    _patch_client(monkeypatch, handler)
    text = await extraction.download_paper_text("https://arxiv.org/abs/1804.07927")
    assert "hard negatives" in text
    assert seen == ["r.jina.ai"]  # arxiv.org untouched: no PDF fetch when Jina works


@pytest.mark.asyncio
async def test_jina_is_retried_with_backoff(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text=BODY)

    _patch_client(monkeypatch, handler)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "jina" and calls["n"] == 3


@pytest.mark.asyncio
async def test_jina_out_of_credit_is_not_retried_and_falls_through(monkeypatch):
    calls = {"jina": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            calls["jina"] += 1
            return httpx.Response(402)
        assert request.url.host == "arxiv.org"
        return httpx.Response(200, content=PDF_BYTES)

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(fulltext, "pdf_bytes_to_text", lambda data: BODY)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "pymupdf" and calls["jina"] == 1
    assert [a["backend"] for a in result.attempts] == ["jina", "arxiv-pdf", "pymupdf"]
    assert "402" in result.attempts[0]["error"]  # the real cause stays visible


@pytest.mark.asyncio
async def test_local_pdf_is_the_safety_net_when_jina_and_mineru_are_unavailable(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(402)
        return httpx.Response(200, content=PDF_BYTES)

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(fulltext, "pdf_bytes_to_text", lambda data: BODY)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "pymupdf"


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
async def test_mineru_runs_after_jina_fails_and_is_cleaned(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    raw = BODY + "\n\n![](images/a.jpg)\n\n<table><tr><td>A</td><td>1</td></tr></table>\n"
    zip_bytes = _zip_with_markdown(raw)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(402)
        if request.url.path == "/api/v4/extract/task" and request.method == "POST":
            assert request.headers["authorization"] == "Bearer test-token"
            assert json_url(request) == "https://arxiv.org/pdf/1804.07927.pdf"
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "t1"}})
        if request.url.path == "/api/v4/extract/task/t1":
            return httpx.Response(
                200,
                json={"code": 0, "data": {"state": "done", "full_zip_url": "https://cdn-mineru.openxlab.org.cn/out.zip"}},
            )
        if request.url.host == "cdn-mineru.openxlab.org.cn":
            return httpx.Response(200, content=zip_bytes)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    _patch_client(monkeypatch, handler)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "mineru"
    assert "hard negatives" in result.text and "![" not in result.text and "<table" not in result.text
    assert "A | 1" in result.text
    assert [a["backend"] for a in result.attempts] == ["jina", "mineru"]


def json_url(request: httpx.Request) -> str:
    import json

    return json.loads(request.content)["url"]


@pytest.mark.asyncio
async def test_mineru_is_attempted_once_then_the_local_pdf_is_used(monkeypatch):
    monkeypatch.setenv("MINERU_API_KEY", "test-token")
    calls = {"mineru": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(402)
        if request.url.host == "arxiv.org":
            return httpx.Response(200, content=PDF_BYTES)
        if request.url.host == "mineru.net":
            calls["mineru"] += 1
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "t1"}}) if request.method == "POST" else httpx.Response(
                200, json={"code": 0, "data": {"state": "failed", "err_msg": "parsing failed, please try again later"}}
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(fulltext, "pdf_bytes_to_text", lambda data: BODY)
    result = await fulltext.download_fulltext("https://arxiv.org/abs/1804.07927")
    assert result.backend == "pymupdf"
    assert calls["mineru"] == 2  # one submit + one poll, no second task
    mineru = next(a for a in result.attempts if a["backend"] == "mineru")
    assert "try again later" in mineru["error"]


@pytest.mark.asyncio
async def test_short_text_counts_as_failure_and_the_error_names_every_backend(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "r.jina.ai":
            return httpx.Response(200, text="Just a moment...")
        if request.url.host == "arxiv.org":
            return httpx.Response(200, content=PDF_BYTES)
        return httpx.Response(200, text="<html/>", headers={"content-type": "text/html"})

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(fulltext, "pdf_bytes_to_text", lambda data: "too short")
    with pytest.raises(RuntimeError) as excinfo:
        await extraction.download_paper_text("https://arxiv.org/abs/1804.07927")
    message = str(excinfo.value)
    assert "jina: only" in message and "pymupdf: only" in message
