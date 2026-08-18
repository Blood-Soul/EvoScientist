"""Promote this turn's extracted paper experiences into long-term EvoMemory.

The ``extract_paper_experiences`` tool caches parsed L1/L2 JSON per session+paper
under ``<MEMORIES_DIR>/paper_experiences/sessions/<session>/<paper>/{l1,l2}.json``.
This script reads that cache for a given paper and writes each experience as a
global EvoMemory observation (L2 -> semantic, L1 -> procedural), so future
sessions retrieve them via ``search_observations``. Idempotent by content hash.

Reuses the paper_experience module's own path/key helpers so the cache location
matches exactly, and the same field mapping as ``scripts/import_experience_bank.py``.

Usage (run inside WSL; PYTHONUTF8=1 avoids Windows cp936 mojibake):
    PYTHONUTF8=1 python /skills/paper-experience/scripts/promote_to_memory.py \
        --paper-id <ID> [--session <thread_id>] [--scope global] [--dry-run]

If --session is omitted, the script scans all session dirs for the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_CANDIDATES = ("/mnt/f/EvoScientist-main/EvoScientist-main",)
VENV_CANDIDATES = (
    "/mnt/f/EvoScientist-main/EvoScientist-main/.venv/bin/python",
    str(Path.home() / ".venv/bin/python"),
)
_REEXEC_FLAG = "_PROMOTE_MEMORY_REEXEC"


def _bootstrap_import():
    """Make the EvoScientist package importable, re-executing if necessary.

    Walking up from ``__file__`` only works when the skill lives inside the repo.
    Once installed to ``~/.evoscientist/skills/`` there is no repo above it, and
    the agent's bare ``python3`` has no EvoScientist package either — so the
    import dies with ModuleNotFoundError. Try the known repo paths first, then
    re-exec into the project venv, which has the package installed.
    """
    # Put the repo on sys.path so ``EvoScientist`` resolves at all.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "EvoScientist" / "__init__.py").exists():
            sys.path.insert(0, str(parent))
            break
    else:
        for repo in REPO_CANDIDATES:
            if (Path(repo) / "EvoScientist" / "__init__.py").exists():
                sys.path.insert(0, repo)
                break

    if os.environ.get(_REEXEC_FLAG):
        return  # already re-executed once; let any import error surface

    # Finding the package is not enough: importing EvoScientist.memory pulls in
    # langchain, which only exists in the project venv. Probe that third-party
    # dependency — probing the top-level package would succeed here and skip the
    # re-exec, then fail deeper with ModuleNotFoundError: langchain.
    try:
        import langchain  # noqa: F401

        return
    except ImportError:
        pass
    for candidate in VENV_CANDIDATES:
        if Path(candidate).is_file():
            env = dict(os.environ, **{_REEXEC_FLAG: "1", "PYTHONUTF8": "1"})
            print(f"[re-exec via {candidate}]", file=sys.stderr)
            os.execve(
                candidate,
                [candidate, os.path.abspath(__file__), *sys.argv[1:]],
                env,
            )


_bootstrap_import()

from EvoScientist import paths  # noqa: E402
from EvoScientist.memory import record_observation_file  # noqa: E402
from EvoScientist.memory.types import (  # noqa: E402
    MemoryScope,
    MemorySourceType,
    MemoryType,
)
from EvoScientist.tools.paper_experience import (  # noqa: E402
    _canonical_paper_key,
    _safe_storage_segment,
    paper_identifier,
)


# Mapping lives in the repo's scripts/ so the offline-bank importer and this
# promoter cannot drift apart on which fields reach the store.
for _cand in (
    Path(__file__).resolve().parents[3] / "scripts",
    Path("/mnt/f/EvoScientist-main/EvoScientist-main/scripts"),
):
    if (_cand / "experience_mapping.py").is_file():
        sys.path.insert(0, str(_cand))
        break

from experience_mapping import (  # noqa: E402
    clean as _clean,
    l1_to_observation as _l1_to_observation,
    l2_to_observation as _l2_to_observation,
)


def _find_paper_dirs(memory_dir: Path, paper_id: str, session: str | None) -> list[Path]:
    """Locate cache dirs for this paper (matching the tool's own key algorithm)."""
    sessions_root = memory_dir / "paper_experiences" / "sessions"
    if not sessions_root.is_dir():
        return []
    paper_key = _safe_storage_segment(
        _canonical_paper_key(paper_id), prefix="paper"
    )
    if session:
        session_key = _safe_storage_segment(session, prefix="session")
        candidate = sessions_root / session_key / paper_key
        return [candidate] if candidate.is_dir() else []
    # No session given: scan every session dir for this paper_key.
    return [
        session_dir / paper_key
        for session_dir in sessions_root.iterdir()
        if (session_dir / paper_key).is_dir()
    ]


def _load_records(paper_dir: Path, level: str) -> list[dict]:
    path = paper_dir / f"{level}.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    exps = payload.get("experiences") if isinstance(payload, dict) else None
    return [e for e in exps if isinstance(e, dict)] if isinstance(exps, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", required=True, help="Canonical id you extracted")
    parser.add_argument("--session", default=None, help="Thread id (omit to scan all)")
    parser.add_argument("--memory-dir", default=None, help="Override MEMORIES_DIR")
    parser.add_argument("--scope", choices=["global", "project"], default="global")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.scope == "project" and not args.project_id:
        parser.error("--project-id is required when --scope project")

    memory_dir = Path(args.memory_dir) if args.memory_dir else paths.MEMORIES_DIR
    scope = MemoryScope(args.scope)
    canonical = paper_identifier(args.paper_id)

    paper_dirs = _find_paper_dirs(memory_dir, args.paper_id, args.session)
    if not paper_dirs:
        print(
            f"No cached experiences found for paper {canonical!r} under "
            f"{memory_dir}/paper_experiences/sessions/"
            + (f" session {args.session!r}" if args.session else " (any session)")
        )
        print("Run extract_paper_experiences on this paper first, in this session.")
        sys.exit(1)

    jobs: list[tuple[str, dict]] = []
    for paper_dir in paper_dirs:
        for rec in _load_records(paper_dir, "l1"):
            obs = _l1_to_observation(rec, canonical)
            if obs:
                obs["memory_type"] = MemoryType.PROCEDURAL
                jobs.append(("l1", obs))
        for rec in _load_records(paper_dir, "l2"):
            obs = _l2_to_observation(rec, canonical)
            if obs:
                obs["memory_type"] = MemoryType.SEMANTIC
                jobs.append(("l2", obs))

    print(f"Paper: {canonical}")
    print(f"MEMORIES_DIR: {memory_dir}")
    print(f"Cache dirs: {len(paper_dirs)}  Experiences queued: {len(jobs)}\n")

    created = duplicate = 0
    for level, obs in jobs:
        if args.dry_run:
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
            source_session_id=f"paper-experience-{level}",
            source_agent="paper-experience",
            evidence=obs["evidence"],
        )
        if result["created"]:
            created += 1
        else:
            duplicate += 1

    print("=== Summary ===")
    if args.dry_run:
        print(f"Dry-run: {len(jobs)} would be written (nothing written)")
    else:
        print(f"Created:   {created}")
        print(f"Duplicate: {duplicate} (already existed, idempotent)")


if __name__ == "__main__":
    main()
