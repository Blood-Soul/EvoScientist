#!/usr/bin/env python3
"""Fetch a paper's full text without depending on r.jina.ai.

``fetch_paper.py`` routes every full-text fetch through Jina Reader, which is
blocked on some networks and hangs until the caller's timeout. This script
prefers paths that work directly:

    1. arXiv HTML   (arxiv.org/html -> markdownify)   -- no PDF library needed
    2. arXiv PDF    (arxiv.org/pdf  -> PyMuPDF)       -- used when HTML is absent
    3. Jina Reader                                    -- last-resort fallback

It also **re-executes itself inside the project virtualenv** when the current
interpreter lacks the needed libraries: the agent's shell runs a bare
``python3`` that has neither ``markdownify`` nor ``pymupdf`` installed, which
would otherwise silently degrade every fetch back to Jina.

Prints the saved Markdown path on stdout; progress goes to stderr.

Usage:
    python fetch_paper_direct.py --paper-id 2210.03629 --papers-dir artifacts/papers
    python fetch_paper_direct.py --url https://arxiv.org/abs/2210.03629 --papers-dir artifacts/papers
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

VENV_CANDIDATES = (
    "/mnt/f/EvoScientist-main/EvoScientist-main/.venv/bin/python",
    str(Path.home() / ".venv/bin/python"),
)
USER_AGENT = "EvoScientist/1.0 (paper-navigator direct fetch)"
_REEXEC_FLAG = "_PAPER_FETCH_REEXEC"


def _ensure_capable_interpreter() -> None:
    """Re-exec in a venv that has httpx + markdownify, if this one doesn't."""
    if os.environ.get(_REEXEC_FLAG):
        return  # already re-executed once; run with whatever we have
    try:
        import httpx  # noqa: F401
        import markdownify  # noqa: F401

        return  # current interpreter is fine
    except ImportError:
        pass
    for candidate in VENV_CANDIDATES:
        if not Path(candidate).is_file():
            continue
        env = dict(os.environ, **{_REEXEC_FLAG: "1", "PYTHONUTF8": "1"})
        print(f"[re-exec via {candidate}]", file=sys.stderr)
        os.execve(candidate, [candidate, os.path.abspath(__file__), *sys.argv[1:]], env)
    print(
        "Warning: no interpreter with httpx+markdownify found; "
        "full-text conversion may be limited.",
        file=sys.stderr,
    )


_ensure_capable_interpreter()

import httpx  # noqa: E402


def arxiv_id_from(value: str) -> str | None:
    """Return a bare arXiv id from an id or any arxiv.org URL, else None."""
    text = value.strip()
    parsed = urlparse(text)
    if (parsed.hostname or "").lower().endswith("arxiv.org"):
        match = re.match(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?$", parsed.path)
        if match:
            text = match.group(1)
    text = re.sub(r"^arxiv:", "", text, flags=re.IGNORECASE).strip()
    if re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?", text, re.IGNORECASE):
        return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)
    return None


def _get(url: str, timeout: float = 60.0) -> httpx.Response:
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        return client.get(url, headers={"User-Agent": USER_AGENT})


def fetch_arxiv_html(arxiv_id: str) -> str:
    """arXiv's native HTML rendering -> Markdown. No PDF library required."""
    from markdownify import markdownify

    last_error: Exception | None = None
    for url in (
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    ):
        try:
            response = _get(url)
            if response.status_code != 200 or len(response.text) < 20000:
                continue
            text = markdownify(response.text, heading_style="ATX")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if len(text) >= 5000:
                print(f"[html] {url} -> {len(text)} chars", file=sys.stderr)
                return text
        except Exception as exc:  # try the next mirror
            last_error = exc
    raise RuntimeError(f"no usable arXiv HTML rendering ({last_error})")


def fetch_arxiv_pdf(arxiv_id: str) -> str:
    """arXiv PDF -> text via PyMuPDF."""
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # legacy module name

    import tempfile

    response = _get(f"https://arxiv.org/pdf/{arxiv_id}.pdf", timeout=120)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError("arXiv did not return a PDF")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(response.content)
        tmp = handle.name
    try:
        with pymupdf.open(tmp) as doc:
            text = "\n\n".join(page.get_text() for page in doc).strip()
    finally:
        os.unlink(tmp)
    print(f"[pdf] {len(text)} chars", file=sys.stderr)
    return text


def fetch_via_jina(url: str) -> str:
    """Last resort: Jina Reader (blocked on some networks)."""
    key = os.environ.get("JINA_API_KEY", "")
    headers = {"User-Agent": USER_AGENT}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        response = client.get(f"https://r.jina.ai/{url}", headers=headers)
        response.raise_for_status()
    print(f"[jina] {len(response.text)} chars", file=sys.stderr)
    return response.text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--paper-id", "-p", help="arXiv id, DOI, or S2 id")
    source.add_argument("--url", "-u", help="Paper URL")
    parser.add_argument("--papers-dir", default="artifacts/papers")
    parser.add_argument(
        "--limit-chars",
        type=int,
        default=120000,
        help="Truncate the saved text at N chars (default 120000)",
    )
    args = parser.parse_args()

    raw = args.paper_id or args.url
    arxiv_id = arxiv_id_from(raw)

    text = ""
    errors: list[str] = []
    if arxiv_id:
        for label, fetcher in (("html", fetch_arxiv_html), ("pdf", fetch_arxiv_pdf)):
            try:
                text = fetcher(arxiv_id)
                break
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                print(f"[{label} failed] {exc}", file=sys.stderr)

    if not text:
        url = args.url or (
            f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else str(raw)
        )
        try:
            text = fetch_via_jina(url)
        except Exception as exc:
            errors.append(f"jina: {exc}")
            print("All fetch strategies failed:", file=sys.stderr)
            for line in errors:
                print(f"  - {line}", file=sys.stderr)
            raise SystemExit(1)

    if len(text) > args.limit_chars:
        text = text[: args.limit_chars] + "\n\n---\n*[Truncated]*"

    slug = re.sub(r"[^\w.-]+", "_", arxiv_id or str(raw))[:60]
    papers_dir = Path(args.papers_dir)
    papers_dir.mkdir(parents=True, exist_ok=True)
    out = papers_dir / f"{slug}.md"
    header = f"# arXiv:{arxiv_id}\n\n" if arxiv_id else ""
    out.write_text(header + text + "\n", encoding="utf-8")

    print(f"chars: {len(text)}", file=sys.stderr)
    print(out)


if __name__ == "__main__":
    main()
