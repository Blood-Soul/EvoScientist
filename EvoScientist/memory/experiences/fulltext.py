"""One full-text acquisition chain for papers, shared by the experience worker and paper-navigator.

The chain, in order, with every attempt recorded:

1. **MinerU cloud** (optional, preferred when ``MINERU_API_KEY`` is set).
   The precise API accepts a PDF URL, parses layout, tables and formulas, and
   returns a zip containing ``full.md``. A token-free agent API also exists; it
   is opt-in (``EVOSCIENTIST_MINERU_AGENT_API=1``) because its shared queue can
   leave tasks pending for many minutes.
2. **The arXiv PDF, parsed locally** with pymupdf4llm (layout-aware Markdown),
   then pymupdf's plain text, then pypdf. This path needs no network service
   beyond arxiv.org and no GPU.
3. **Jina Reader** (``r.jina.ai``), only when the PDF path is unavailable or
   fails; it converts both PDFs and HTML pages to Markdown.
4. **A direct GET**, accepted only when the server returns ``text/plain`` or
   ``text/markdown`` — raw HTML is never handed to the extractor.

This module deliberately imports nothing from the EvoScientist package: the
paper-navigator skill loads it by file path from an interpreter that may not
have the package installed. Configuration is therefore read from environment
variables only:

``MINERU_API_KEY``
    MinerU precise-API token. Without it, MinerU is skipped unless the agent API
    is explicitly enabled.
``EVOSCIENTIST_PAPER_FULLTEXT_BACKEND``
    ``auto`` (default), ``mineru``, ``pdf`` or ``jina``.
``EVOSCIENTIST_MINERU_MODEL``
    MinerU model version for the precise API (default ``vlm``).
``EVOSCIENTIST_MINERU_TIMEOUT_S``
    Per-paper polling budget in seconds (default 600).
``EVOSCIENTIST_MINERU_AGENT_API``
    Set to ``1`` to use the token-free agent API. Off by default.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any

import httpx

_USER_AGENT = "EvoScientist/0.2 paper-fulltext"

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([^?#\s]+?)(?:\.pdf)?(?:[?#].*)?$", re.IGNORECASE
)

MINERU_PRECISE_BASE = "https://mineru.net/api/v4"
MINERU_AGENT_BASE = "https://mineru.net/api/v1/agent"

_PYMUPDF_QUIET = False


def _silence_pymupdf(pymupdf: Any) -> None:
    """Route PyMuPDF's parser/OCR chatter away from stdout.

    ``pymupdf.message`` and pymupdf4llm's layout messages default to stdout,
    which the paper-navigator skill hands to the agent as its result. Set the
    sink once per process; failures still surface as exceptions.
    """
    global _PYMUPDF_QUIET
    if not _PYMUPDF_QUIET:
        setter = getattr(pymupdf, "set_messages", None)
        if setter is not None:  # present from PyMuPDF 1.24.4; older just prints
            setter(stream=io.StringIO())
        _PYMUPDF_QUIET = True



class FulltextError(RuntimeError):
    """No backend produced usable text; ``attempts`` says what each one saw."""

    def __init__(self, url: str, attempts: list[dict[str, Any]]):
        self.url = url
        self.attempts = attempts
        detail = "; ".join(
            f"{a['backend']}: {a.get('error') or 'empty text'}" for a in attempts
        )
        super().__init__(f"full-text fetch failed for {url} ({detail})")


@dataclass
class Fulltext:
    text: str
    backend: str
    attempts: list[dict[str, Any]] = field(default_factory=list)


def arxiv_pdf_url(url: str) -> str:
    """Return the arXiv PDF URL for an abs/pdf URL, else ''."""
    match = _ARXIV_ID_RE.search(url)
    return f"https://arxiv.org/pdf/{match.group(1)}" if match else ""


def _looks_like_pdf_url(url: str) -> bool:
    path = url.split("?", 1)[0].split("#", 1)[0].casefold()
    return path.endswith(".pdf") or "arxiv.org/pdf/" in path


def _backend() -> str:
    choice = os.environ.get("EVOSCIENTIST_PAPER_FULLTEXT_BACKEND", "auto").strip().casefold()
    return choice if choice in {"auto", "mineru", "pdf", "jina"} else "auto"


def _record(
    attempts: list[dict[str, Any]], backend: str, started: float, error: Any = None, **extra: Any
) -> None:
    entry = {
        "backend": backend,
        "seconds": round(time.monotonic() - started, 2),
        **extra,
    }
    if error is not None:
        entry["error"] = str(error)[:800]
    attempts.append(entry)


# ------------------------------------------------------------------ local pdf


def pdf_bytes_to_text(data: bytes) -> str:
    """Layout-aware Markdown from PDF bytes, degrading to plain text then pypdf."""
    if not data.lstrip().startswith(b"%PDF"):
        raise ValueError("not a PDF")
    try:
        import pymupdf
        import pymupdf4llm

        _silence_pymupdf(pymupdf)
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            return pymupdf4llm.to_markdown(doc).strip()
        finally:
            doc.close()
    except ImportError:
        pass
    try:
        import pymupdf

        _silence_pymupdf(pymupdf)
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            pages = [page.get_text("text").strip() for page in doc]
        finally:
            doc.close()
        return "\n\n".join(page for page in pages if page).strip()
    except ImportError:
        pass
    from pypdf import PdfReader

    pages = [(p.extract_text() or "").strip() for p in PdfReader(io.BytesIO(data)).pages]
    return "\n\n".join(page for page in pages if page).strip()


async def _fetch_pdf_bytes(client: httpx.AsyncClient, pdf_url: str) -> bytes:
    response = await client.get(pdf_url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").casefold()
    if "pdf" not in content_type and not response.content.lstrip().startswith(b"%PDF"):
        raise ValueError(f"not a PDF (content-type {content_type or 'unknown'})")
    return response.content


# --------------------------------------------------------------------- MinerU


def _zip_markdown(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        target = next((n for n in names if n.endswith("full.md")), None)
        if target is None:
            target = next((n for n in names if n.casefold().endswith(".md")), None)
        if target is None:
            raise ValueError(f"no Markdown in MinerU zip (members: {', '.join(names[:8])})")
        return archive.read(target).decode("utf-8", "replace").strip()


async def _mineru_poll(
    client: httpx.AsyncClient, task_url: str, headers: dict[str, str], timeout_s: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    state = ""
    while time.monotonic() < deadline:
        await asyncio.sleep(3)
        response = await client.get(task_url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        state = str(data.get("state") or "")
        if state == "done":
            return data
        if state == "failed":
            raise RuntimeError(f"MinerU task failed: {data.get('err_msg') or data.get('err_code')}")
    raise TimeoutError(f"MinerU task still {state or 'unfinished'} after {timeout_s:.0f}s")


async def _mineru_precise(
    client: httpx.AsyncClient, pdf_url: str, key: str, timeout_s: float
) -> str:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "url": pdf_url,
        "model_version": os.environ.get("EVOSCIENTIST_MINERU_MODEL", "vlm"),
        "language": "en",
        "enable_table": True,
        "enable_formula": True,
    }
    response = await client.post(f"{MINERU_PRECISE_BASE}/extract/task", json=body, headers=headers)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"MinerU submit rejected: {payload.get('msg')}")
    task_id = (payload.get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"MinerU submit returned no task_id: {payload}")
    data = await _mineru_poll(client, f"{MINERU_PRECISE_BASE}/extract/task/{task_id}", headers, timeout_s)
    zip_url = data.get("full_zip_url")
    if not zip_url:
        raise RuntimeError("MinerU finished without full_zip_url")
    archive = await client.get(zip_url)
    archive.raise_for_status()
    return _zip_markdown(archive.content)


async def _mineru_agent(
    client: httpx.AsyncClient, pdf_bytes: bytes, file_name: str, timeout_s: float
) -> str:
    response = await client.post(
        f"{MINERU_AGENT_BASE}/parse/file",
        json={"file_name": file_name, "language": "en"},
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"MinerU agent submit rejected: {payload.get('msg')}")
    data = payload.get("data") or {}
    task_id, upload_url = data.get("task_id"), data.get("file_url")
    if not task_id or not upload_url:
        raise RuntimeError(f"MinerU agent submit incomplete: {payload}")
    upload = await client.put(upload_url, content=pdf_bytes)
    upload.raise_for_status()
    result = await _mineru_poll(client, f"{MINERU_AGENT_BASE}/parse/{task_id}", {}, timeout_s)
    markdown_url = result.get("markdown_url")
    if not markdown_url:
        raise RuntimeError("MinerU agent finished without markdown_url")
    markdown = await client.get(markdown_url)
    markdown.raise_for_status()
    return markdown.text.strip()


# ----------------------------------------------------------------------- Jina


async def _jina(client: httpx.AsyncClient, url: str, attempts: list[dict[str, Any]]) -> str:
    key = os.environ.get("JINA_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    reader_url = url if url.startswith("https://r.jina.ai/") else f"https://r.jina.ai/{url}"
    last: Exception | None = None
    for delay in (2.0, 5.0, None):
        started = time.monotonic()
        try:
            response = await client.get(reader_url, headers=headers)
            response.raise_for_status()
            text = response.text.strip()
            if text:
                _record(attempts, "jina", started, chars=len(text))
                return text
            last = ValueError("empty response")
        except Exception as exc:
            last = exc
            _record(attempts, "jina", started, error=exc)
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            if status is not None and status != 429 and status < 500:
                break
        if delay is not None:
            await asyncio.sleep(delay)
    assert last is not None
    raise last


# ----------------------------------------------------------------------- main


async def download_fulltext(
    url: str,
    *,
    timeout: float = 90.0,
    backend: str | None = None,
) -> Fulltext:
    """Fetch a paper's full text, recording every backend that was tried."""
    choice = (backend or _backend()).casefold()
    if choice not in {"auto", "mineru", "pdf", "jina"}:
        choice = "auto"
    pdf_url = arxiv_pdf_url(url) or (url if _looks_like_pdf_url(url) else "")
    key = os.environ.get("MINERU_API_KEY", "").strip()
    mineru_timeout = float(os.environ.get("EVOSCIENTIST_MINERU_TIMEOUT_S", "600"))
    agent_api = os.environ.get("EVOSCIENTIST_MINERU_AGENT_API", "").strip() == "1"
    attempts: list[dict[str, Any]] = []

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/markdown,text/plain;q=0.9,application/pdf;q=0.8,*/*;q=0.1",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        # The PDF is cheap to fetch and is the fallback for every MinerU mode.
        pdf_bytes: bytes | None = None
        if pdf_url:
            started = time.monotonic()
            try:
                pdf_bytes = await _fetch_pdf_bytes(client, pdf_url)
                _record(attempts, "arxiv-pdf", started, bytes=len(pdf_bytes), url=pdf_url)
            except Exception as exc:
                _record(attempts, "arxiv-pdf", started, error=exc, url=pdf_url)

        if choice in {"auto", "mineru"} and (key or agent_api):
            started = time.monotonic()
            try:
                if key:
                    text = await _mineru_precise(client, pdf_url or url, key, mineru_timeout)
                    _record(attempts, "mineru", started, chars=len(text))
                    return Fulltext(text=text, backend="mineru", attempts=attempts)
                if pdf_bytes is not None:
                    name = (pdf_url.rsplit("/", 1)[-1] or "paper.pdf").split("?")[0]
                    text = await _mineru_agent(client, pdf_bytes, name, mineru_timeout)
                    _record(attempts, "mineru-agent", started, chars=len(text))
                    return Fulltext(text=text, backend="mineru-agent", attempts=attempts)
                raise ValueError("agent API needs a downloaded PDF")
            except Exception as exc:
                _record(attempts, "mineru" if key else "mineru-agent", started, error=exc)

        if choice in {"auto", "pdf", "mineru"} and pdf_bytes is not None:
            started = time.monotonic()
            try:
                text = await asyncio.to_thread(pdf_bytes_to_text, pdf_bytes)
                _record(attempts, "pymupdf", started, chars=len(text))
                if text:
                    return Fulltext(text=text, backend="pymupdf", attempts=attempts)
                raise ValueError("local PDF parse returned no text")
            except Exception as exc:
                _record(attempts, "pymupdf", started, error=exc)

        if choice in {"auto", "pdf", "mineru", "jina"}:
            try:
                text = await _jina(client, url, attempts)
                return Fulltext(text=text, backend="jina", attempts=attempts)
            except Exception:
                pass

        started = time.monotonic()
        try:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").casefold()
            if "text/plain" not in content_type and "text/markdown" not in content_type:
                raise ValueError(f"refusing HTML response (content-type {content_type or 'unknown'})")
            text = response.text.strip()
            _record(attempts, "direct", started, chars=len(text))
            if text:
                return Fulltext(text=text, backend="direct", attempts=attempts)
            raise ValueError("empty response")
        except Exception as exc:
            _record(attempts, "direct", started, error=exc)

    raise FulltextError(url, attempts)


def download_fulltext_sync(url: str, *, timeout: float = 90.0, backend: str | None = None) -> Fulltext:
    """Blocking wrapper for the skill scripts."""
    return asyncio.run(download_fulltext(url, timeout=timeout, backend=backend))


__all__ = [
    "Fulltext",
    "FulltextError",
    "arxiv_pdf_url",
    "download_fulltext",
    "download_fulltext_sync",
    "pdf_bytes_to_text",
]
