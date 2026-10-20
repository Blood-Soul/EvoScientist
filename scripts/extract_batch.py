#!/usr/bin/env python3
"""Extract L1/L2 experiences for a small batch of papers into a scratch bank.

For trying an extraction-prompt change on real papers before committing to it.
Writes to whatever `--bank` you point at, so the production bank stays untouched:
point it at a scratch directory, extract the same papers under the old and new
prompts, and score both with `score_experiences.py`.

The production path is the queue-driven worker
(`EvoScientist/memory/agents/paper_experience_worker.py`); this is deliberately
simpler -- no queue, no dedup, no promotion to long-term memory -- because a
handful of papers does not need any of that.

Usage:
    scripts/extract_batch.py --bank .scratch/bank_v5 \\
        --corpus ./papers --limit 4
    scripts/extract_batch.py --bank .scratch/bank_v5 --arxiv 2005.13366
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from EvoScientist.memory.experiences.extraction import (
        download_paper_text,
        load_experience_prompts,
        run_experience_extraction,
    )
    from EvoScientist.memory.experiences.store import store_paper_experiences
    from EvoScientist.memory.experiences.task_context import (
        TaskContext,
        TaskContextError,
    )
except ImportError as _exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"cannot import the experience layer ({_exc}); run from the repo root "
        "with its virtualenv, e.g. .venv/bin/python scripts/extract_batch.py"
    ) from _exc


def _local_corpus(corpus: Path, limit: int) -> list[dict[str, str]]:
    """Papers from a directory of `<paper_id>.md` full texts beside `<paper_id>.json`.

    This is the corpus the production bank was built from, already converted from
    PDF. Reading it avoids re-downloading, which both removes the slowest and
    least reliable step and keeps the paper set identical to the existing records
    -- so a prompt change is the only difference between the two extractions.
    """
    if not corpus.is_dir():
        raise SystemExit(f"corpus dir not found: {corpus}")
    out = []
    for md_path in sorted(corpus.glob("*.md")):
        paper_id = md_path.stem
        meta: dict[str, Any] = {}
        meta_path = md_path.with_suffix(".json")
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        arxiv = str(meta.get("arXiv") or "").strip()
        out.append(
            {
                "paper_id": paper_id,
                # A URL is still recorded for provenance, but nothing fetches it.
                "url": f"https://arxiv.org/abs/{arxiv}"
                if arxiv
                else f"local://{paper_id}",
                "title": str(meta.get("title") or ""),
                "domain_arxiv": "",
                "text_path": str(md_path),
            }
        )
        if len(out) >= limit:
            break
    return out


async def _extract_one(
    paper: dict[str, str],
    *,
    bank: Path,
    project: str,
    task_context: Any | None,
    prompts: dict[str, str],
    model: Any,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    paper_id = paper["paper_id"]
    started = time.monotonic()
    async with sem:
        try:
            cached = paper.get("text_path")
            if cached and Path(cached).is_file():
                text = Path(cached).read_text(encoding="utf-8")
                source = "cached"
            else:
                text = await download_paper_text(paper["url"])
                source = "downloaded"

            payloads = await run_experience_extraction(
                paper_id=paper_id,
                paper_text=text,
                prompts=prompts,
                model=model,
                domain_arxiv=paper.get("domain_arxiv") or None,
                task_context=task_context,
            )
            store_paper_experiences(
                memory_dir=bank,
                project_id=project,
                paper_id=paper_id,
                url=paper["url"],
                title=paper.get("title") or paper_id,
                paper_text=text,
                prompts={k: v for k, v in prompts.items() if k in ("l1", "l2")},
                payloads=payloads,
                domain_arxiv=paper.get("domain_arxiv") or None,
            )
        except Exception as exc:  # one paper must not stop the batch
            # Type, not just str(exc): httpx timeout exceptions carry an empty
            # message, so a bare str() reports the failure as "".
            detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            print(f"  FAILED {paper_id}: {detail}", file=sys.stderr, flush=True)
            return {"paper_id": paper_id, "error": detail}

    l1 = payloads.get("l1", {}).get("experiences", [])
    l2 = payloads.get("l2", {}).get("experiences", [])
    induced = [r for r in l2 if r.get("source_l1_ids")]
    row = {
        "paper_id": paper_id,
        "source": source,
        "chars": len(text),
        "l1": len(l1),
        "l2_direct": len(l2) - len(induced),
        "l2_induced": len(induced),
        "seconds": round(time.monotonic() - started, 1),
        # Whether the new fields actually came through is the whole point of a
        # trial batch, so surface it per paper rather than making it a later query.
        "with_trigger_context": sum(
            1 for r in (*l1, *l2) if str(r.get("trigger_context") or "").strip()
        ),
        "with_evidence_scope": sum(1 for r in (*l1, *l2) if r.get("evidence_scope")),
        "total_records": len(l1) + len(l2),
    }
    print(
        f"  {paper_id}  L1={row['l1']} L2={row['l2_direct']}+{row['l2_induced']}i  "
        f"trigger={row['with_trigger_context']}/{row['total_records']} "
        f"scope={row['with_evidence_scope']}/{row['total_records']}  "
        f"{row['seconds']}s",
        flush=True,
    )
    return row


async def _main_async(args: argparse.Namespace) -> None:
    if args.model or args.provider:
        from EvoScientist.llm.models import get_chat_model

        model = get_chat_model(args.model, provider=args.provider)
    else:
        from EvoScientist.EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()

    prompts = load_experience_prompts()
    print(f"prompts loaded: {sorted(prompts)}")

    task_context = None
    if args.task_context:
        try:
            task_context = TaskContext.from_json_file(args.task_context)
        except TaskContextError as exc:
            raise SystemExit(f"bad --task-context: {exc}") from exc
        print(f"task-oriented extraction for: {task_context.task}")
    else:
        print("bench-blind extraction (no --task-context)")

    if args.arxiv:
        papers = [
            {
                "paper_id": f"arxiv_{a}",
                "url": f"https://arxiv.org/abs/{a}",
                "title": "",
                "domain_arxiv": "",
                "text_path": "",
            }
            for a in args.arxiv
        ]
    else:
        papers = _local_corpus(args.corpus, args.limit)
    if not papers:
        raise SystemExit("no papers to extract")
    print(f"extracting {len(papers)} paper(s) into {args.bank}")

    sem = asyncio.Semaphore(args.concurrency)
    rows = await asyncio.gather(
        *(
            _extract_one(
                p,
                bank=args.bank,
                project=args.project_out,
                task_context=task_context,
                prompts=prompts,
                model=model,
                sem=sem,
            )
            for p in papers
        )
    )

    ok = [r for r in rows if "error" not in r]
    summary = {
        "papers_attempted": len(rows),
        "papers_ok": len(ok),
        "records_total": sum(r["total_records"] for r in ok),
        "l1_total": sum(r["l1"] for r in ok),
        "l2_direct_total": sum(r["l2_direct"] for r in ok),
        "l2_induced_total": sum(r["l2_induced"] for r in ok),
        "records_with_trigger_context": sum(r["with_trigger_context"] for r in ok),
        "records_with_evidence_scope": sum(r["with_evidence_scope"] for r in ok),
        "failures": [r for r in rows if "error" in r],
    }
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\nscore with:\n  scripts/score_experiences.py --bank {args.bank} "
        f"--project {args.project_out} --level l2 --limit 10"
    )
    if args.graph_output:
        from EvoScientist.memory._atomic import atomic_write_json
        from EvoScientist.memory.experiences.graph import build_experience_graph

        graph_prompt = (REPO_ROOT / "prompt/experience_linker.md").read_text(
            encoding="utf-8"
        )
        previous = (
            json.loads(args.graph_output.read_text(encoding="utf-8"))
            if args.graph_output.exists()
            else None
        )
        graph = await build_experience_graph(
            memory_dir=args.bank,
            project_id=args.project_out,
            prompt=graph_prompt,
            model=model,
            max_pairs=None if args.graph_max_pairs == 0 else args.graph_max_pairs,
            candidate_strategy=args.graph_candidate_strategy,
            previous=previous,
        )
        atomic_write_json(args.graph_output, graph)
        if args.graph_candidate_strategy == "field":
            atomic_write_json(
                args.graph_output.with_name(
                    args.graph_output.stem + ".routing-index.json"
                ),
                {
                    "project_id": args.project_out,
                    **graph["candidate_state"]["routing_index"],
                },
            )
        print(
            f"graph sidecar: {args.graph_output} "
            f"({len(graph['edges'])} edges, "
            f"{sum('error' in row for row in graph['decisions'])} linker errors)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True, help="output bank dir")
    parser.add_argument(
        "--project-out", default="P-trial", help="project id to write under"
    )
    parser.add_argument("--arxiv", nargs="*", help="arXiv ids to download fresh")
    parser.add_argument(
        "--corpus",
        type=Path,
        help="directory of already-converted <paper_id>.md full texts "
        "(e.g. ./papers)",
    )
    parser.add_argument("--limit", type=int, default=5, help="how many papers")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--task-context",
        type=Path,
        help="JSON file describing the downstream task, to bias what gets "
        "extracted (see prompt/task_context.schema.json). Omit for "
        "bench-blind extraction with no added task block.",
    )
    parser.add_argument("--model", help="override model for this run only")
    parser.add_argument("--provider", help="override provider for this run only")
    parser.add_argument(
        "--graph-output",
        type=Path,
        help="optionally build a reviewable experience-graph sidecar after extraction",
    )
    parser.add_argument(
        "--graph-max-pairs",
        type=int,
        default=0,
        help="Optional graph pair cap; default 0 uses per-record nomination limits",
    )
    parser.add_argument(
        "--graph-candidate-strategy", choices=["lexical", "field"], default="lexical"
    )
    args = parser.parse_args()
    if not args.arxiv and not args.corpus:
        parser.error("pass --arxiv, or --corpus")
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
