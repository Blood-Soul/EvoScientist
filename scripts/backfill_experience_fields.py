#!/usr/bin/env python3
"""Backfill `transferable_core` and `bindings` onto already-extracted records.

Both fields were added to the extraction prompts after the library already held
records, so every pre-existing `E-*` record lacks them. They are optional on
read, but the reuse layer runs degraded without them:

  * `policy.select.transferable_core()` falls back to the first 200 characters
    of `statement` as the rerank descriptor -- a truncated paragraph opening,
    which is what the reranker sees instead of the claim.
  * The policy writer's `rebind` step has to mine source-fixed values out of
    prose rather than reading a structured list, and the A/B harness cannot
    count stale-binding hits without one.

This derives both from fields the records already have (`statement`, `scope`,
`action`, `effect`), so it does not re-download papers or re-run full
extraction. Nothing else about a record is touched: the same statement, the same
evidence, the same confidence.

Safety and scale properties, in order of how much they matter here:

  * **Idempotent.** A record that already has a non-empty field is skipped, so
    an interrupted run is resumed by re-running it. `--overwrite` forces
    regeneration.
  * **Resumable and batched.** Work is committed per paper, not per run, and
    `--limit` bounds one invocation. A library of any size is walked in as many
    passes as needed.
  * **Never destructive.** Each level file is rewritten via a temporary file in
    the same directory and an atomic replace, and only after every record in it
    parsed and validated. A crash mid-write leaves the original intact. Pass
    `--backup` to keep a `.bak` copy as well.
  * **`--dry-run` by default is not assumed**: the script requires either
    `--apply` or `--dry-run` explicitly, because it edits the live memory store.

Usage:
    scripts/backfill_experience_fields.py --dry-run
    scripts/backfill_experience_fields.py --apply --limit 5
    scripts/backfill_experience_fields.py --apply --project P-xxxx --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from EvoScientist.memory.experiences.extraction import (
        ExperienceOutputError,
        _validate_bindings,
    )
    from EvoScientist.memory.experiences.store import reset_experience_parse_cache
except ImportError as _exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"cannot import the experience layer ({_exc}); run this from the repo "
        "root with its virtualenv, e.g. "
        ".venv/bin/python scripts/backfill_experience_fields.py --dry-run"
    ) from _exc

LEVELS: tuple[str, ...] = ("l1", "l2")
TARGET_FIELDS = ("transferable_core", "bindings")

# Derived fields are only as good as their inputs. A record whose `statement` is
# a stub cannot yield a meaningful core, and writing a bad one is worse than
# leaving the documented fallback in place, so such records are reported as
# skipped rather than filled.
MIN_STATEMENT_CHARS = 40

SYSTEM_PROMPT = """\
You rewrite one already-extracted research experience record into two \
reuse-oriented fields. You are NOT re-extracting the paper and you have no \
access to it: work only from the record's own text, and never introduce a fact \
it does not contain.

Return only a JSON object, no prose and no Markdown fences:

{"transferable_core": "...", "bindings": [{"name": "...", "kind": "..."}]}

`transferable_core`: at most 60 words. Restate the record's claim with every \
source-specific value removed -- dataset names, model names, specific \
hyperparameter values, and scale mentions -- while keeping the causal \
structure ("when X, doing Y yields Z"). Example: given "on ImageNet with \
ResNet-50, a cosine LR schedule reduced overfitting by 3.2%", the core is \
"when training large vision models, using a cosine LR schedule reduces \
overfitting". If the claim is a pure ablation of one method's own \
hyperparameters and does not transfer, return an empty string for this field.

`bindings`: every source-fixed value the record names, each tagged with a \
`kind` from exactly this set: dataset, model, scale, hyperparam, baseline, \
metric, toolchain, other. Use the value as the record writes it. Return an \
empty array if the record names none.

A binding is a NAMED entity that a later reader could wrongly carry over into a \
different task -- "ImageNet", "GPT-4", "HumanEval", "AdamW", "8xA100". It is \
NOT a measured outcome. Do not emit bare numbers or percentages such as "91%", \
"3.2%", or "80%": those are results, they are meaningless as names, and they \
match unrelated text. When a number only matters together with what it \
measures, either name the metric alone ("pass@1") or omit it. Prefer the \
`metric` kind for the metric's name, never for its value.
"""


@dataclass
class Stats:
    """What one run did, reported per record rather than per file."""

    papers_seen: int = 0
    papers_changed: int = 0
    records_seen: int = 0
    records_filled: int = 0
    records_already_had: int = 0
    records_too_thin: int = 0
    records_failed: int = 0
    failures: list[str] = field(default_factory=list)

    def render(self, *, applied: bool) -> str:
        verb = "wrote" if applied else "would write"
        lines = [
            f"papers: {self.papers_seen} seen, {self.papers_changed} {verb}",
            f"records: {self.records_seen} seen, {self.records_filled} {verb}, "
            f"{self.records_already_had} already populated, "
            f"{self.records_too_thin} too thin to derive, "
            f"{self.records_failed} failed",
        ]
        if self.failures:
            lines.append("failures:")
            lines.extend(f"  {item}" for item in self.failures[:20])
            if len(self.failures) > 20:
                lines.append(f"  ... and {len(self.failures) - 20} more")
        return "\n".join(lines)


def _needs_backfill(item: Mapping[str, Any], *, overwrite: bool) -> bool:
    """Report whether this record is missing either derived field."""
    if overwrite:
        return True
    for name in TARGET_FIELDS:
        value = item.get(name)
        if value in (None, "", [], {}):
            return True
    return False


def _record_context(item: Mapping[str, Any]) -> str:
    """Render the record's own text for the model, and nothing else."""
    parts = []
    for name in ("domain", "task", "statement", "scope", "action", "effect"):
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            parts.append(f"[{name}] {' '.join(value.split())}")
    for name in ("applicable_when", "not_applicable_when"):
        value = item.get(name)
        if isinstance(value, list) and value:
            joined = "; ".join(str(entry) for entry in value if entry)
            if joined:
                parts.append(f"[{name}] {joined}")
    return "\n".join(parts)


def _is_usable_binding(name: str) -> bool:
    """Report whether a binding name can identify a source-fixed value.

    A binding exists so a later reader can be caught carrying a source value
    into a different task, which means it has to be a name. Measured outcomes
    fail at that twice over: `91%` identifies nothing, and as a substring it
    fires on any text that happens to contain those characters. The prompt asks
    for names, but the check is here too -- the consumer's correctness should
    not rest on the model having complied.
    """
    stripped = name.strip()
    if len(stripped) < 3:
        return False
    # Anything with no letter is a bare number, percentage, or ratio.
    return any(character.isalpha() for character in stripped)


def _parse_response(raw: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse and validate the model's two fields, raising on anything odd."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExperienceOutputError(f"backfill output is not JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ExperienceOutputError("backfill output must be a JSON object")

    core = payload.get("transferable_core", "")
    if not isinstance(core, str):
        raise ExperienceOutputError("transferable_core must be a string")
    core = " ".join(core.split())

    bindings = payload.get("bindings", [])
    # Reuse the extraction validator so a backfilled record cannot hold a
    # binding shape that a freshly extracted one would have been rejected for.
    _validate_bindings(bindings, level="l1")
    normalized = []
    for row in bindings:
        name = " ".join(str(row["name"]).split())
        if not _is_usable_binding(name):
            continue
        normalized.append(
            {"name": name, "kind": str(row.get("kind") or "").strip().casefold()}
        )
    return core, normalized


async def _derive_one(
    item: Mapping[str, Any], *, model: Any, semaphore: asyncio.Semaphore
) -> tuple[str, list[dict[str, Any]]]:
    """Ask the model for one record's two fields."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from EvoScientist.utils import format_message_content

    async with semaphore:
        response = await model.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=_record_context(item)),
            ]
        )
    return _parse_response(format_message_content(response))


def _write_atomically(path: Path, payload: Mapping[str, Any], *, backup: bool) -> None:
    """Replace one level file only after the whole new document is on disk.

    The store is the library; a half-written `l1.json` loses every record in it.
    The temporary file is created in the same directory so the replace is atomic
    rather than a cross-device copy.
    """
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_bytes(path.read_bytes())
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


async def _backfill_level(
    path: Path,
    *,
    model: Any,
    semaphore: asyncio.Semaphore,
    stats: Stats,
    overwrite: bool,
    apply: bool,
    backup: bool,
) -> bool:
    """Backfill one level file, returning whether it changed."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        stats.records_failed += 1
        stats.failures.append(f"{path}: unreadable ({exc})")
        return False
    experiences = payload.get("experiences")
    if not isinstance(experiences, list):
        stats.failures.append(f"{path}: no experiences array")
        return False

    pending: list[int] = []
    for index, item in enumerate(experiences):
        if not isinstance(item, Mapping):
            continue
        stats.records_seen += 1
        if not _needs_backfill(item, overwrite=overwrite):
            stats.records_already_had += 1
            continue
        statement = item.get("statement")
        if (
            not isinstance(statement, str)
            or len(statement.strip()) < MIN_STATEMENT_CHARS
        ):
            stats.records_too_thin += 1
            continue
        pending.append(index)

    if not pending:
        return False
    if not apply:
        stats.records_filled += len(pending)
        return True

    results = await asyncio.gather(
        *(
            _derive_one(experiences[index], model=model, semaphore=semaphore)
            for index in pending
        ),
        return_exceptions=True,
    )

    changed = False
    for index, result in zip(pending, results, strict=True):
        if isinstance(result, BaseException):
            stats.records_failed += 1
            record_id = experiences[index].get("id", f"index {index}")
            stats.failures.append(
                f"{path.parent.name}/{path.name} {record_id}: {result}"
            )
            continue
        core, bindings = result
        item = dict(experiences[index])
        # An empty core means the model judged the claim untransferable. Record
        # the bindings anyway and leave `transferable_core` absent so the
        # documented `statement` fallback still applies -- writing "" would make
        # the field present but useless.
        if core:
            item["transferable_core"] = core
        if bindings:
            item["bindings"] = bindings
        if item == experiences[index]:
            continue
        experiences[index] = item
        stats.records_filled += 1
        changed = True

    if changed:
        _write_atomically(path, payload, backup=backup)
    return changed


def _project_dirs(experiences_root: Path, project_id: str | None) -> list[Path]:
    """Return the project directories to walk."""
    projects = experiences_root / "projects"
    if not projects.is_dir():
        return []
    if project_id:
        candidate = projects / project_id
        return [candidate] if candidate.is_dir() else []
    return sorted(path for path in projects.iterdir() if path.is_dir())


async def _run(args: argparse.Namespace) -> Stats:
    stats = Stats()
    model: Any | None = None
    if args.apply:
        from EvoScientist.EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    memory_dir = Path(args.memory_dir).expanduser()
    for project_dir in _project_dirs(memory_dir / "experiences", args.project):
        for paper_dir in sorted(
            path for path in project_dir.iterdir() if path.is_dir()
        ):
            # `--limit` bounds the papers this run actually WORKS ON, not the
            # papers it walks past. Counting visits instead would make every
            # pass re-spend its budget on the already-finished prefix, so a
            # library larger than one batch could never be finished by repeated
            # passes -- which is the whole point of the flag.
            if args.limit and stats.papers_changed >= args.limit:
                break
            level_paths = [
                paper_dir / f"{level}.json"
                for level in LEVELS
                if (paper_dir / f"{level}.json").exists()
            ]
            if not level_paths:
                continue
            stats.papers_seen += 1
            changed = False
            for path in level_paths:
                changed |= await _backfill_level(
                    path,
                    model=model,
                    semaphore=semaphore,
                    stats=stats,
                    overwrite=args.overwrite,
                    apply=args.apply,
                    backup=args.backup,
                )
            if changed:
                stats.papers_changed += 1
                if args.apply:
                    # The store caches parsed level files on mtime and size;
                    # drop it so a later read in this process sees the rewrite.
                    reset_experience_parse_cache()
    return stats


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from EvoScientist import paths

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Call the model and rewrite level files in place.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without calling the model or writing.",
    )
    parser.add_argument(
        "--memory-dir",
        default=str(paths.MEMORIES_DIR),
        help="Memory root to walk (default: the configured store).",
    )
    parser.add_argument(
        "--project",
        default="",
        help="Only this project id (default: every project in the store).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after backfilling this many papers; 0 means no limit. "
        "Papers an earlier run already finished do not count against it, so "
        "repeated passes walk the whole library.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Concurrent model calls (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate both fields even where they are already populated.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a .bak copy of each rewritten level file.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    stats = asyncio.run(_run(args))
    print(stats.render(applied=args.apply))
    return 1 if stats.records_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
