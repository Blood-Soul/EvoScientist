"""One full-text acquisition chain for papers, shared by the experience worker and paper-navigator.

The chain, in order, with every attempt recorded:

1. **Jina Reader** (``r.jina.ai``), retried with backoff. It converts both PDFs and
   HTML pages to clean Markdown and is fast.
2. **MinerU cloud** (optional, used when ``MINERU_API_KEY`` is set), one attempt only:
   it parses layout, tables and formulas but can take minutes on a long paper. The
   result is reduced to the paper's text (figures dropped, tables flattened). A
   token-free agent API exists; it is opt-in (``EVOSCIENTIST_MINERU_AGENT_API=1``).
3. **The arXiv PDF, parsed locally** (the safety net) with pymupdf4llm, then
   pymupdf's plain text, then pypdf. Needs only arxiv.org and no GPU.
4. **A direct GET**, accepted only when the server returns ``text/plain`` or
   ``text/markdown`` — raw HTML is never handed to the extractor.

A backend whose text is under 500 characters counts as failed and the chain moves on.

This module deliberately imports nothing from the EvoScientist package: the
paper-navigator skill loads it by file path from an interpreter that may not
have the package installed. Configuration is therefore read from environment
variables only:

``MINERU_API_KEY``
    MinerU precise-API token. Without it, MinerU is skipped unless the agent API
    is explicitly enabled.
``JINA_API_KEY``
    Jina Reader key (works without one at a much lower rate limit).
``EVOSCIENTIST_PAPER_FULLTEXT_BACKEND``
    ``auto`` (default: Jina, MinerU, local PDF), or the first stop: ``jina``,
    ``mineru`` or ``pdf``. The local PDF parse stays behind ``jina``/``mineru``.
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


_LANDING_MARKERS = (
    "skip to main content",
    "submission history",
    "bibliographic explorer",
    "bibliographic and citation tools",
    "access paper:",
    "cite as: arxiv",
    "just a moment",
    "enable javascript and cookies",
)
_BODY_WORD_RE = re.compile(r"\b(introduction|methods?|methodology|experiments?|results|conclusions?)\b", re.I)


def landing_page_reason(text: str, source_url: str = "") -> str:
    """Why this text is a landing/abstract page rather than a paper's body, else ''.

    The experience extractor must read the full text. A converter handed an arXiv
    ``/abs/`` link, a publisher page or a bot wall happily returns the page chrome plus
    an abstract (about 9 KB for arXiv), which is long enough to pass a length check.
    """
    head = text[:600].casefold()
    if "url source:" in head and "/abs/" in head.split("url source:", 1)[1].split("\n", 1)[0]:
        return "converted an /abs/ page (abstract only)"
    lowered = text.casefold()
    hits = [m for m in _LANDING_MARKERS if m in lowered]
    if len(hits) >= 2:
        return f"page chrome, not a paper body ({', '.join(hits[:3])})"
    if len(text) < 15000 and len(_BODY_WORD_RE.findall(text)) < 2:
        return "no body sections (introduction/methods/results...) found"
    return ""


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


_IMAGE_RE = re.compile(r"^[ \t]*!\[[^\]]*\]\([^)]*\)[ \t]*$", re.MULTILINE)
_INLINE_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"</?(?:sup|sub|b|i|u|em|strong|span|br)\b[^>]*>", re.IGNORECASE)


def _table_to_text(match: re.Match[str]) -> str:
    """One HTML table -> one line per row, cells joined by ' | ' (numbers kept)."""
    rows = []
    for row in _ROW_RE.findall(match.group(0)):
        cells = [_TAG_RE.sub("", re.sub(r"\s+", " ", c)).strip() for c in _CELL_RE.findall(row)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def clean_mineru_markdown(markdown: str) -> str:
    """Keep the paper's words: drop figures, flatten tables, tidy inline HTML.

    MinerU's ``full.md`` points at extracted figures (``![](images/...)``) and renders
    tables as HTML. The experience extractor reads text, so figures are dropped
    (their captions are ordinary lines and stay), each table becomes one text line per
    row, and ``<sup>``/``<sub>`` style tags are removed. The References section is
    stripped later, by the caller, the same way for every backend.
    """
    text = _TABLE_RE.sub(_table_to_text, markdown)
    text = _IMAGE_RE.sub("", text)
    text = _INLINE_IMAGE_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _mineru_pdf_url(url: str) -> str:
    """MinerU decides the file type from the URL, so an arXiv ``/pdf/<id>`` needs ``.pdf``."""
    base, sep, rest = url.partition("?")
    if "arxiv.org/pdf/" in base and not base.casefold().endswith(".pdf"):
        base += ".pdf"
    return base + sep + rest


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
        "url": _mineru_pdf_url(pdf_url),
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
    return clean_mineru_markdown(_zip_markdown(archive.content))


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
    return clean_mineru_markdown(markdown.text)


# ----------------------------------------------------------------------- Jina

_JINA_RETRY_DELAYS_S = (1.0, 2.0, None)  # 3 attempts, 3 s of waiting in total
_JINA_TIMEOUT = httpx.Timeout(45.0, connect=10.0)


async def _jina(client: httpx.AsyncClient, url: str, attempts: list[dict[str, Any]]) -> str:
    key = os.environ.get("JINA_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    reader_url = url if url.startswith("https://r.jina.ai/") else f"https://r.jina.ai/{url}"
    last: Exception | None = None
    # Short on purpose: slower backends and the local PDF parse sit behind Jina, so a
    # stuck request should cost seconds, not minutes. An unreachable host fails at the
    # 10 s connect limit; a slow conversion gets 45 s to read.
    for delay in _JINA_RETRY_DELAYS_S:
        started = time.monotonic()
        try:
            response = await client.get(reader_url, headers=headers, timeout=_JINA_TIMEOUT)
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


_MIN_TEXT_CHARS = 500  # shorter than this is an error page or a stub, not a paper

_ORDER = {
    "auto": ("jina", "mineru", "pdf"),
    "jina": ("jina", "pdf"),
    "mineru": ("mineru", "pdf"),
    "pdf": ("pdf",),
}


async def download_fulltext(
    url: str,
    *,
    timeout: float = 90.0,
    backend: str | None = None,
) -> Fulltext:
    """Fetch a paper's full text, recording every backend that was tried.

    ``auto`` order: Jina Reader (retried with backoff), then MinerU (one attempt:
    it is slow, and a retry costs minutes), then the arXiv PDF parsed locally, then
    a direct text/plain response. ``backend`` picks the first stop (``jina``,
    ``mineru`` or ``pdf``); the local PDF parse stays behind it as the safety net.
    A backend that returns under ``_MIN_TEXT_CHARS`` counts as failed.
    """
    choice = (backend or _backend()).casefold()
    order = _ORDER.get(choice, _ORDER["auto"])
    pdf_url = arxiv_pdf_url(url) or (url if _looks_like_pdf_url(url) else "")
    key = os.environ.get("MINERU_API_KEY", "").strip()
    mineru_timeout = float(os.environ.get("EVOSCIENTIST_MINERU_TIMEOUT_S", "600"))
    agent_api = os.environ.get("EVOSCIENTIST_MINERU_AGENT_API", "").strip() == "1"
    attempts: list[dict[str, Any]] = []

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "text/markdown,text/plain;q=0.9,application/pdf;q=0.8,*/*;q=0.1",
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=10.0), follow_redirects=True, headers=headers
    ) as client:
        pdf_cache: list[bytes | None] = []

        async def pdf_bytes() -> bytes | None:
            """The arXiv PDF, downloaded the first time a backend needs it."""
            if not pdf_cache:
                data: bytes | None = None
                if pdf_url:
                    started = time.monotonic()
                    try:
                        data = await _fetch_pdf_bytes(client, pdf_url)
                        _record(attempts, "arxiv-pdf", started, bytes=len(data), url=pdf_url)
                    except Exception as exc:
                        _record(attempts, "arxiv-pdf", started, error=exc, url=pdf_url)
                pdf_cache.append(data)
            return pdf_cache[0]

        def accept(text: str, name: str, started: float) -> bool:
            if len(text.strip()) < _MIN_TEXT_CHARS:
                _record(attempts, name, started, error=f"only {len(text.strip())} chars")
                return False
            reason = landing_page_reason(text)
            if reason:
                _record(attempts, name, started, error=f"not the full text: {reason}")
                return False
            return True

        for step in order:
            if step == "jina":
                started = time.monotonic()
                try:
                    # An arXiv /abs/ page converts to just the abstract (about 9 KB);
                    # the /pdf/ link converts to the whole paper.
                    text = await _jina(client, pdf_url or url, attempts)
                except Exception:
                    continue  # _jina recorded every attempt
                if accept(text, "jina", started):
                    return Fulltext(text=text.strip(), backend="jina", attempts=attempts)
            elif step == "mineru" and (key or agent_api):
                name = "mineru" if key else "mineru-agent"
                started = time.monotonic()
                try:
                    if key:
                        text = await _mineru_precise(client, pdf_url or url, key, mineru_timeout)
                    else:
                        data = await pdf_bytes()
                        if data is None:
                            raise ValueError("agent API needs a downloaded PDF")
                        file_name = (pdf_url.rsplit("/", 1)[-1] or "paper.pdf").split("?")[0]
                        text = await _mineru_agent(client, data, file_name, mineru_timeout)
                    _record(attempts, name, started, chars=len(text))
                    if accept(text, name, started):
                        return Fulltext(text=text.strip(), backend=name, attempts=attempts)
                except Exception as exc:
                    _record(attempts, name, started, error=exc)
            elif step == "pdf":
                data = await pdf_bytes()
                if data is None:
                    continue
                started = time.monotonic()
                try:
                    text = await asyncio.to_thread(pdf_bytes_to_text, data)
                    _record(attempts, "pymupdf", started, chars=len(text))
                    if accept(text, "pymupdf", started):
                        return Fulltext(text=text.strip(), backend="pymupdf", attempts=attempts)
                except Exception as exc:
                    _record(attempts, "pymupdf", started, error=exc)

        started = time.monotonic()
        try:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").casefold()
            if "text/plain" not in content_type and "text/markdown" not in content_type:
                raise ValueError(f"refusing HTML response (content-type {content_type or 'unknown'})")
            text = response.text.strip()
            _record(attempts, "direct", started, chars=len(text))
            if accept(text, "direct", started):
                return Fulltext(text=text, backend="direct", attempts=attempts)
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
    "landing_page_reason",
    "pdf_bytes_to_text",
]
