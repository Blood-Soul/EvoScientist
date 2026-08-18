#!/usr/bin/env python3
"""Build the two comparison contexts for solution-ab.

For each requested paper it collects:
  A · the extraction tool's rendered L1/L2 experiences
  B · the paper body, with References/Appendix stripped

then concatenates each side into one context file. Stripping the tail matters:
references and appendices are 15-46% of a fetched paper and contribute nothing to
solution design, so leaving them in would both waste budget and make the size
comparison misleading.

Writes ``artifacts/ab/context_A.md`` and ``artifacts/ab/context_B.md`` and prints a
size report. Flags the pair when the two sides differ by more than a tolerance, so
the caller can say so instead of pretending the comparison was size-matched.

Usage (workspace root):
    python build_contexts.py --paper-ids 2303.11366,2502.13172 [--out-dir artifacts/ab]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Sections that end the usable body. Matched as a Markdown heading only, so a
# mid-sentence mention of "references" cannot truncate the paper.
TAIL_HEADINGS = re.compile(
    r"^#{1,3}\s*(?:"
    r"references?|bibliography|appendix|appendices"
    r"|acknowledge?ments?|acknowledgments?"
    r"|instructions?\s+for\s+reporting\s+errors"
    r")\b",
    re.IGNORECASE,
)

DEFAULT_TOLERANCE = 35.0  # percent
MEMORY_SESSIONS = (
    Path.home() / ".evoscientist/memories/paper_experiences/sessions"
)


def strip_tail(text: str) -> tuple[str, int]:
    """Drop everything from the first References/Appendix heading onward."""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if TAIL_HEADINGS.match(line.strip()):
            kept = "".join(lines[:index]).rstrip()
            return kept + "\n", len(text) - len(kept)
    return text, 0


def find_experience_file(paper_id: str) -> Path | None:
    """Newest rendered.md for this paper across all extraction sessions."""
    if not MEMORY_SESSIONS.is_dir():
        return None
    hits = [
        path
        for path in MEMORY_SESSIONS.glob(f"*/*{paper_id}*/rendered.md")
        if path.is_file()
    ]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def find_paper_file(paper_id: str, papers_dir: Path) -> Path | None:
    direct = papers_dir / f"{paper_id}.md"
    if direct.is_file():
        return direct
    hits = sorted(papers_dir.glob(f"*{paper_id}*.md"), key=lambda p: -p.stat().st_size)
    return hits[0] if hits else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-ids",
        required=True,
        help="Comma-separated arXiv ids (1-3 recommended)",
    )
    parser.add_argument("--papers-dir", default="artifacts/papers")
    parser.add_argument("--out-dir", default="artifacts/ab")
    parser.add_argument(
        "--keep-tail",
        action="store_true",
        help="Keep References/Appendix (default strips them)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Flag when sides differ by more than this %% (default {DEFAULT_TOLERANCE})",
    )
    args = parser.parse_args()

    paper_ids = [p.strip() for p in args.paper_ids.split(",") if p.strip()]
    if not paper_ids:
        parser.error("--paper-ids is empty")
    if len(paper_ids) > 3:
        print(
            f"Warning: {len(paper_ids)} papers requested; merged contexts may be "
            "large. 3 or fewer is recommended.",
            file=sys.stderr,
        )

    papers_dir = Path(args.papers_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_parts: list[str] = []
    b_parts: list[str] = []
    rows: list[tuple[str, int, int, int]] = []
    missing: list[str] = []

    for paper_id in paper_ids:
        a_path = find_experience_file(paper_id)
        b_path = find_paper_file(paper_id, papers_dir)

        if a_path is None or b_path is None:
            which = []
            if a_path is None:
                which.append("experiences (run paper-experience first)")
            if b_path is None:
                which.append(f"full text (expected {papers_dir}/{paper_id}.md)")
            missing.append(f"{paper_id}: missing {' and '.join(which)}")
            continue

        a_text = a_path.read_text(encoding="utf-8")
        b_text = b_path.read_text(encoding="utf-8")
        stripped = 0
        if not args.keep_tail:
            b_text, stripped = strip_tail(b_text)

        if len(b_text) < 10000:
            print(
                f"Warning: {paper_id} body is only {len(b_text)} chars — an /abs/ "
                "page may have been fetched instead of the full paper. Group B "
                "would lose for reasons unrelated to the comparison.",
                file=sys.stderr,
            )

        a_parts.append(f"# === Paper {paper_id} · 抽取经验 ===\n\n{a_text.strip()}\n")
        b_parts.append(f"# === Paper {paper_id} · 论文正文 ===\n\n{b_text.strip()}\n")
        rows.append((paper_id, len(a_text), len(b_text), stripped))

    if missing:
        print("Cannot build contexts:", file=sys.stderr)
        for line in missing:
            print(f"  - {line}", file=sys.stderr)
        raise SystemExit(1)

    a_ctx = out_dir / "context_A.md"
    b_ctx = out_dir / "context_B.md"
    a_ctx.write_text("\n\n".join(a_parts), encoding="utf-8")
    b_ctx.write_text("\n\n".join(b_parts), encoding="utf-8")

    total_a = sum(r[1] for r in rows)
    total_b = sum(r[2] for r in rows)

    print(f"papers: {len(rows)}")
    print(f"{'paper':<14}{'A(经验)':>10}{'B(正文)':>10}{'尾部已删':>10}")
    for paper_id, size_a, size_b, stripped in rows:
        print(f"{paper_id:<14}{size_a:>10}{size_b:>10}{stripped:>10}")
    print(f"{'TOTAL':<14}{total_a:>10}{total_b:>10}")

    smaller = min(total_a, total_b) or 1
    diff_pct = abs(total_a - total_b) / smaller * 100
    longer = "A" if total_a > total_b else "B"
    print(f"\nA/B ratio: {total_a / (total_b or 1) * 100:.1f}%")
    print(f"size gap:  {diff_pct:.1f}% ({longer} is longer)")
    if diff_pct > args.tolerance:
        print(
            f"NOTE size gap exceeds {args.tolerance:.0f}% — report this in the output "
            "so the reader knows the two sides were not size-matched.",
        )
    else:
        print(f"OK size gap within {args.tolerance:.0f}% tolerance.")

    print(f"\n{a_ctx}")
    print(f"{b_ctx}")


if __name__ == "__main__":
    main()
