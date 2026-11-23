#!/usr/bin/env python3
"""Embed a project's experience records and paper chunks ahead of time, so the first search does not wait for it.

Uses the same client, texts and cache as retrieval and clustering (``EvoScientist/memory/embedding.py``), so what
this writes is what they read. Needs ``EVOSCIENTIST_EMBEDDING_URL`` pointing at a running service
(``scripts/embedding_server.py``).

    EVOSCIENTIST_EMBEDDING_URL=http://127.0.0.1:18765/v1 python scripts/embed_library.py --project P-xxxx
    ... --memory-dir ~/.evoscientist/memories   # default: the configured memory directory
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from EvoScientist.memory import dense, embedding
    from EvoScientist.memory.experiences import cross_paper
    from EvoScientist.memory.experiences.store import list_experience_documents
    from EvoScientist.memory.papers.store import list_paper_chunk_documents

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--project", required=True, help="the project id (P-...)")
    ap.add_argument("--memory-dir", default="", help="defaults to the configured memory directory")
    ap.add_argument("--skip-chunks", action="store_true", help="embed the experience records only")
    args = ap.parse_args(argv)
    if not embedding.configured():
        print(f"{embedding.URL_ENV} is not set: start scripts/embedding_server.py and point it there", file=sys.stderr)
        return 2
    if args.memory_dir:
        memory_dir = Path(args.memory_dir).expanduser()
    else:
        from EvoScientist import paths

        memory_dir = paths.MEMORIES_DIR
    jobs = [
        ("experience records", [dense.document_text(d) for d in
                                list_experience_documents(memory_dir=memory_dir, project_id=args.project)]),
        ("L1 clustering views", [cross_paper._view_text(r["item"]) for r in
                                 cross_paper.load_l1_records(memory_dir, args.project)]),
    ]
    if not args.skip_chunks:
        jobs.append(("paper chunks", [dense.document_text(d) for d in
                                      list_paper_chunk_documents(memory_dir=memory_dir, project_id=args.project)]))
    for name, texts in jobs:
        started = time.monotonic()
        out = embedding.embed_documents(texts, memory_dir=memory_dir) if texts else None
        if texts and out is None:
            print(f"{name}: the embedding service failed; see the log", file=sys.stderr)
            return 1
        print(f"{name}: {len(texts)} embedded or already cached in {time.monotonic() - started:.1f}s")
    print(f"cache: {embedding._cache_path(memory_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
