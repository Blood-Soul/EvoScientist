"""Import the offline paper experience bank (L1/L2 JSONL) into EvoMemory.

Reads L1 practical + L2 inductive experiences extracted from papers and writes
each as a global observation via ``record_observation_file`` — so they become
searchable through the native EvoMemory preflight (``search_observations`` +
per-turn index injection) across all research sessions.

Data is correct UTF-8 on disk; all reads are explicit ``encoding="utf-8"`` and
the process should run with ``PYTHONUTF8=1`` on Windows to avoid cp936 fallback.

Usage (run inside WSL so MEMORIES_DIR resolves to the real ~/.evoscientist):
    PYTHONUTF8=1 python scripts/import_experience_bank.py \
        --base /mnt/f/experience-bank-v2 [--limit N] [--l2-only] \
        [--include-l1-fine] [--scope global] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(_repo_root()))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for experience_mapping

from EvoScientist.memory import record_observation_file  # noqa: E402
from EvoScientist.memory.types import (  # noqa: E402
    MemoryScope,
    MemorySourceType,
    MemoryType,
)


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


from experience_mapping import (  # noqa: E402
    clean as _clean,
    l1_to_observation as _l1_to_observation,
    l2_to_observation as _l2_to_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=os.environ.get("EXP_BANK_BASE", "/mnt/f/experience-bank-v2"),
        help="Experience bank root (contains _out/l1_batch, _out/l2_batch)",
    )
    parser.add_argument("--memory-dir", default=None, help="Override MEMORIES_DIR")
    parser.add_argument("--limit", type=int, default=None, help="Cap records per level")
    parser.add_argument("--l2-only", action="store_true")
    parser.add_argument(
        "--include-l1-fine",
        action="store_true",
        help="Also import fine-grained L1 (default: only coarse/medium)",
    )
    parser.add_argument(
        "--scope", choices=["global", "project"], default="global"
    )
    parser.add_argument("--project-id", default="", help="Required when scope=project")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.scope == "project" and not args.project_id:
        parser.error("--project-id is required when --scope project")

    scope = MemoryScope(args.scope)
    base = Path(args.base)

    if args.memory_dir:
        memory_dir = Path(args.memory_dir)
    else:
        from EvoScientist import paths

        memory_dir = paths.MEMORIES_DIR
    print(f"MEMORIES_DIR: {memory_dir}")
    print(f"Scope: {scope.value}  Base: {base}")
    print(f"L1 fine included: {args.include_l1_fine}  L2-only: {args.l2_only}\n")

    jobs: list[tuple[str, dict]] = []

    if not args.l2_only:
        l1_path = base / "_out" / "l1_batch" / "_all_experiences.jsonl"
        l1 = _read_jsonl(l1_path)
        kept = 0
        for rec in l1:
            gran = _clean(rec.get("granularity"))
            if not args.include_l1_fine and gran == "fine":
                continue
            obs = _l1_to_observation(rec)
            if obs is None:
                continue
            obs["memory_type"] = MemoryType.PROCEDURAL
            jobs.append(("l1", obs))
            kept += 1
            if args.limit and kept >= args.limit:
                break
        print(f"L1: {len(l1)} read → {kept} queued")

    l2_path = base / "_out" / "l2_batch" / "_all_experiences.jsonl"
    l2 = _read_jsonl(l2_path)
    kept = 0
    for rec in l2:
        obs = _l2_to_observation(rec)
        if obs is None:
            continue
        obs["memory_type"] = MemoryType.SEMANTIC
        jobs.append(("l2", obs))
        kept += 1
        if args.limit and kept >= args.limit:
            break
    print(f"L2: {len(l2)} read → {kept} queued\n")

    created = 0
    duplicate = 0
    skipped = 0
    for level, obs in jobs:
        if args.dry_run:
            skipped += 1
            continue
        result = record_observation_file(
            memory_dir=memory_dir,
            project_id=args.project_id,
            memory_type=obs["memory_type"],
            summary=obs["summary"],
            observation=obs["observation"],
            why_it_matters=obs["why_it_matters"],
            scope=scope,
            source_type=MemorySourceType.TURN,
            source_session_id=f"offline-import-{level}",
            source_agent="experience-bank",
            evidence=obs["evidence"],
        )
        if result["created"]:
            created += 1
        else:
            duplicate += 1

    print("=== Summary ===")
    print(f"Queued:    {len(jobs)}")
    if args.dry_run:
        print(f"Dry-run:   {skipped} (nothing written)")
    else:
        print(f"Created:   {created}")
        print(f"Duplicate: {duplicate} (already existed, idempotent)")


if __name__ == "__main__":
    main()
