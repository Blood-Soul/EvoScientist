#!/usr/bin/env python3
"""A/B harness: does the policy layer stop agents copying source-paper values?

Three conditions, same target tasks, same model:

  A `none`   no memory at all -- the floor. Anything memory adds must beat this.
  B `raw`    full `E-*` statement prose injected verbatim. Today's behaviour,
             and the QCR study's "Full Trajectory" condition.
  C `policy` `derive_policy()` output injected instead of the prose.

Metric: **stale-binding rate**. Every fixture experience declares its
source-fixed values in `bindings`, so a hit is a deterministic string match --
the plan names a source value (`ImageNet`, `GPT-3`) that the target task never
mentioned. No LLM judge, so the number is reproducible and cheap.

Reported alongside it:
  * `rebind_rate` -- plans that name the value *and* say to re-derive it.
    Mentioning `ImageNet` to explain provenance is correct; silently training on
    it is the bug. Only unqualified mentions count as stale.
  * `target_hit_rate` -- plans naming the target's own values, guarding against
    a policy that avoids staleness by saying nothing useful.

Deliberately decoupled from the run harness: it seeds a throwaway memory store
from a JSON fixture and calls `derive_policy()` directly, so it runs without a
live project, and a regression in graph wiring cannot silently change the
numbers. It measures the reuse layer in isolation, not a full agent session.

Usage:
    scripts/policy_ab.py --dry-run                 # prompts only, no API calls
    scripts/policy_ab.py --repeats 3               # 3 samples per cell
    scripts/policy_ab.py --fixture my.json --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from EvoScientist.memory.policy import derive_policy
    from EvoScientist.memory.policy.select import gather_candidates
except ImportError as _exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"cannot import the policy layer ({_exc}); run this from the repo root "
        "with its virtualenv, e.g. .venv/bin/python scripts/policy_ab.py"
    ) from _exc

CONDITIONS = ("none", "raw", "policy")
PROJECT_ID = "policy-ab"

# Phrases that turn a source value from an instruction into provenance. Checked
# in a window around the mention: "validated on ImageNet, pick your own" is
# correct reuse, "train on ImageNet" is a stale binding.
REBIND_MARKERS = (
    "re-derive",
    "rederive",
    "rebind",
    "your own",
    "your task",
    "target task",
    "instead of",
    "not ",
    "rather than",
    "was validated",
    "was evaluated",
    "in the source",
    "source paper",
    "originally",
    "the paper used",
    "substitute",
    "replace",
    "analogous",
    "equivalent for",
    "adapt",
)
REBIND_WINDOW = 160


ACTOR_PROMPT = """\
You are planning one concrete experiment. Write the plan the team will execute.

Target task:
{task}

Setting for this task:
{setting}
{memory_block}
Write 6-10 numbered steps. Each step must name the datasets, models, and
hyperparameter values it uses -- a step that defers every value is not a plan.
Output the steps only, no preamble.
"""

MEMORY_BLOCK_RAW = """
Relevant prior experience from the literature:

{records}
"""

MEMORY_BLOCK_POLICY = """
A reuse policy derived for this specific task:

{policy}
"""


@dataclass
class Task:
    """One target task, described so its own values differ from every source."""

    name: str
    description: str
    setting: str
    # Values a correct plan should land on. Substring match, case-insensitive.
    target_values: list[str] = field(default_factory=list)


@dataclass
class Sample:
    """One (task, condition) trial."""

    task: str
    condition: str
    output: str
    stale: list[str] = field(default_factory=list)
    rebound: list[str] = field(default_factory=list)
    target_hits: list[str] = field(default_factory=list)
    # How many target values existed to hit, so target_hit_rate has a denominator.
    target_total: int = 0
    policy_verdict: str = ""
    error: str = ""

    @property
    def is_stale(self) -> bool:
        return bool(self.stale)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


def _mention_pattern(name: str) -> re.Pattern[str]:
    """Match ``name`` as a whole token, not as a substring.

    Plain `in` matching reports "PPO" inside "su*ppo*rt" and "SFT" inside
    "shifts", which would inflate every condition's stale count with noise. The
    guards are conditional on the name's own edges so values whose boundary is
    punctuation still match: `1e-5`, `nDCG@10`, `5% warmup`.
    """
    body = re.escape(name)
    prefix = r"(?<![0-9a-z])" if name[:1].isalnum() else ""
    suffix = r"(?![0-9a-z])" if name[-1:].isalnum() else ""
    return re.compile(f"{prefix}{body}{suffix}")


def _find_mention(haystack: str, name: str) -> int:
    match = _mention_pattern(name).search(haystack)
    return match.start() if match else -1


def _binding_names(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Collect (name, kind) from the fixture's declared bindings, deduplicated."""
    seen: dict[str, str] = {}
    for record in records:
        for row in record.get("bindings") or []:
            name = str(row.get("name") or "").strip()
            if len(name) < 3:
                # Too short to match without firing on unrelated prose.
                continue
            seen.setdefault(_normalize(name), str(row.get("kind") or "other"))
    return sorted(seen.items())


def score_output(
    *,
    output: str,
    task: Task,
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Classify each source binding mentioned in ``output``.

    Returns ``(stale, rebound, target_hits)``. A binding counts as stale only
    when the plan names it and *no* rebinding marker sits nearby: naming
    `ImageNet` to explain where a result came from is correct reuse, while
    naming it as the data to train on is the failure this layer exists to stop.
    Values the target task itself supplies are never stale.
    """
    haystack = _normalize(output)
    task_text = _normalize(f"{task.description} {task.setting}")
    stale: list[str] = []
    rebound: list[str] = []
    for name, kind in _binding_names(records):
        if _find_mention(task_text, name) >= 0:
            # The target task names this value too, so a mention proves nothing.
            continue
        position = _find_mention(haystack, name)
        if position < 0:
            continue
        window = haystack[
            max(0, position - REBIND_WINDOW) : position + len(name) + REBIND_WINDOW
        ]
        label = f"{kind}:{name}"
        if any(marker in window for marker in REBIND_MARKERS):
            rebound.append(label)
        else:
            stale.append(label)
    target_hits = [
        value for value in task.target_values if _normalize(value) in haystack
    ]
    return stale, rebound, target_hits


def load_fixture(path: Path) -> tuple[list[Task], list[dict[str, Any]]]:
    """Read tasks and papers, rejecting a fixture that cannot measure anything."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = [
        Task(
            name=str(row["name"]),
            description=str(row["description"]),
            setting=str(row.get("setting") or ""),
            target_values=[str(v) for v in row.get("target_values") or []],
        )
        for row in payload["tasks"]
    ]
    papers = list(payload["papers"])
    if not tasks or not papers:
        raise SystemExit(f"{path}: fixture needs at least one task and one paper")
    records = [
        record for paper in papers for level in ("l1", "l2") for record in paper[level]
    ]
    if not _binding_names(records):
        raise SystemExit(
            f"{path}: no usable bindings; the stale-binding metric would read 0 "
            "regardless of condition"
        )
    # A source value the task also names is unattributable, so warn rather than
    # silently reporting a metric with a blind spot.
    declared = {name for name, _ in _binding_names(records)}
    for task in tasks:
        task_text = _normalize(f"{task.description} {task.setting}")
        overlap = sorted(
            name for name in declared if _find_mention(task_text, name) >= 0
        )
        if overlap:
            print(
                f"  warning: task {task.name!r} names source bindings {overlap}; "
                "those are excluded from its stale count",
                file=sys.stderr,
            )
    return tasks, papers


def seed_store(memory_dir: Path, papers: list[dict[str, Any]]) -> list[str]:
    """Write fixture papers into a throwaway memory store; return experience IDs."""
    from EvoScientist.memory.experiences.store import (
        list_experience_documents,
        store_paper_experiences,
    )

    for paper in papers:
        payloads = {
            level: {
                "paper_id": paper["paper_id"],
                "experiences": _stamp(paper[level], paper["paper_id"], level),
            }
            for level in ("l1", "l2")
        }
        store_paper_experiences(
            memory_dir=memory_dir,
            project_id=PROJECT_ID,
            paper_id=paper["paper_id"],
            url=paper.get("url", ""),
            title=paper.get("title", ""),
            paper_text=paper.get("title", ""),
            prompts={"l1": "fixture", "l2": "fixture"},
            payloads=payloads,
        )
    return [
        document.observation_id
        for document in list_experience_documents(
            memory_dir=memory_dir, project_id=PROJECT_ID
        )
    ]


def _stamp(
    records: list[dict[str, Any]], paper_id: str, level: str
) -> list[dict[str, Any]]:
    """Add the runtime-injected fields the store expects, without re-extracting."""
    stamped = []
    for index, record in enumerate(records, start=1):
        item = dict(record)
        item.setdefault("id", f"{level}_{paper_id.replace('.', '_')}_{index:02d}")
        item.setdefault("layer", level.upper())
        item.setdefault("domain_arxiv", None)
        item.setdefault("utility", None)
        item.setdefault("confidence", 0.6)
        item["evidence"] = [
            {"source_id": paper_id, **row} for row in item.get("evidence") or []
        ]
        stamped.append(item)
    return stamped


def render_raw_records(records: list[dict[str, Any]], limit: int) -> str:
    """Condition B: full statement prose, the way it reaches an agent today."""
    blocks = []
    for index, record in enumerate(records[:limit], start=1):
        # Retrieved records carry the store's `E-*` id; fixture records read
        # straight off disk (dry run) have none, so fall back to an ordinal.
        label = record.get("id") or f"record-{index}"
        blocks.append(
            f"[{label}] {record.get('domain', '')} / "
            f"{record.get('task', '')}\n"
            f"{record.get('statement', '')}\n"
            f"Applicable when: {'; '.join(record.get('applicable_when') or [])}\n"
            f"Not applicable when: "
            f"{'; '.join(record.get('not_applicable_when') or [])}"
        )
    return "\n\n".join(blocks)


def render_policy(policy: dict[str, Any]) -> str:
    """Condition C: the derived policy, rendered the way the tool returns it."""
    lines = [f"verdict: {policy.get('verdict', '?')}"]
    if policy.get("reason"):
        lines.append(f"reason: {policy['reason']}")
    for field_name in ("procedure", "preconditions", "checks", "declines"):
        rows = policy.get(field_name) or []
        if rows:
            lines.append(f"{field_name}:")
            lines.extend(f"  - {row}" for row in rows)
    if policy.get("rebind"):
        lines.append("rebind (re-derive these; do not copy the source value):")
        for row in policy["rebind"]:
            source = (
                f" [source used: {row['source_value']}]"
                if row.get("source_value")
                else ""
            )
            lines.append(
                f"  - {row['name']} ({row.get('kind', 'other')}): "
                f"{row['how_to_obtain']}{source}"
            )
    for row in policy.get("conflicts") or []:
        lines.append(
            f"conflict between {', '.join(row.get('between') or [])}: "
            f"{row['disagreement']} -- {row.get('discriminator', '')}"
        )
    if policy.get("unsupported"):
        lines.append("memory says nothing about:")
        lines.extend(f"  - {row}" for row in policy["unsupported"])
    return "\n".join(lines)


def build_prompt(*, task: Task, condition: str, memory_text: str) -> str:
    if condition == "none":
        block = ""
    elif condition == "raw":
        block = MEMORY_BLOCK_RAW.format(records=memory_text)
    else:
        block = MEMORY_BLOCK_POLICY.format(policy=memory_text)
    return ACTOR_PROMPT.format(
        task=task.description, setting=task.setting, memory_block=block
    )


async def memory_text_for(
    *,
    condition: str,
    task: Task,
    memory_dir: Path,
    raw_limit: int,
    max_selected: int,
    policy_model: Any | None = None,
) -> tuple[str, str]:
    """Build the injected block for one condition. Returns (text, verdict).

    Condition B runs the *same* retrieval as C and differs only in what it does
    with the hits: raw prose versus a derived policy. Injecting fixture records
    in file order instead would confound retrieval quality with the prose/policy
    contrast this harness is meant to isolate.
    """
    if condition == "none":
        return "", ""
    if condition == "raw":
        candidates = await asyncio.to_thread(
            gather_candidates,
            memory_dir=memory_dir,
            project_id=PROJECT_ID,
            query=task.description,
            limit=raw_limit,
        )
        retrieved = [
            item["record"].get("experience", {}) | {"id": item["id"]}
            for item in candidates
        ]
        return render_raw_records(retrieved, raw_limit), ""
    report = await derive_policy(
        memory_dir=memory_dir,
        project_id=PROJECT_ID,
        task=task.description,
        state=task.setting,
        max_selected=max_selected,
        model=policy_model,
        # The cache key is (task, selected ids), so repeats would all read the
        # first synthesis back. Refresh so each repeat is an independent sample.
        refresh=True,
    )
    policy = report.get("policy")
    if not policy:
        # `no_candidates` / `no_reusable_memory` is a real outcome, not a crash:
        # the policy layer declining to inject is exactly what it should do when
        # memory has nothing, and the plan is then written unaided.
        return "", report.get("status", "none")
    return render_policy(policy), str(policy.get("verdict", ""))


async def run_cell(
    *,
    task: Task,
    condition: str,
    model: Any,
    memory_dir: Path,
    records: list[dict[str, Any]],
    raw_limit: int,
    max_selected: int,
    policy_model: Any | None = None,
) -> Sample:
    """Run one (task, condition) trial end to end."""
    from langchain_core.messages import HumanMessage

    try:
        memory_text, verdict = await memory_text_for(
            condition=condition,
            task=task,
            memory_dir=memory_dir,
            raw_limit=raw_limit,
            max_selected=max_selected,
            policy_model=policy_model,
        )
        prompt = build_prompt(task=task, condition=condition, memory_text=memory_text)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        output = _message_text(response)
    except Exception as exc:
        return Sample(
            task=task.name,
            condition=condition,
            output="",
            error=f"{type(exc).__name__}: {exc}",
        )
    stale, rebound, target_hits = score_output(
        output=output, task=task, records=records
    )
    return Sample(
        task=task.name,
        condition=condition,
        output=output,
        stale=stale,
        rebound=rebound,
        target_hits=target_hits,
        target_total=len(task.target_values),
        policy_verdict=verdict,
    )


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def summarize(samples: list[Sample]) -> dict[str, dict[str, Any]]:
    """Per-condition rates over the samples that produced output."""
    summary: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        cells = [s for s in samples if s.condition == condition]
        scored = [s for s in cells if not s.error]
        errors = len(cells) - len(scored)
        if not scored:
            summary[condition] = {"n": 0, "errors": errors}
            continue
        stale_counts = [len(s.stale) for s in scored]
        summary[condition] = {
            "n": len(scored),
            "errors": errors,
            # Fraction of plans containing at least one copied source value.
            "stale_binding_rate": sum(1 for c in stale_counts if c) / len(scored),
            # Copied values per plan; separates "one slip" from "wholesale copy".
            "stale_per_plan": statistics.mean(stale_counts),
            "rebind_rate": sum(1 for s in scored if s.rebound) / len(scored),
            "target_hit_rate": statistics.mean(
                len(s.target_hits) / s.target_total if s.target_total else 0.0
                for s in scored
            ),
            "stale_by_kind": dict(
                Counter(
                    label.split(":", 1)[0] for s in scored for label in s.stale
                ).most_common()
            ),
        }
    return summary


def print_report(summary: dict[str, dict[str, Any]], samples: list[Sample]) -> None:
    labels = {
        "none": "A no memory",
        "raw": "B raw E-* prose",
        "policy": "C derived policy",
    }
    print()
    print(
        f"{'condition':<20} {'n':>3} {'stale%':>7} {'/plan':>6} "
        f"{'rebind%':>8} {'target%':>8}"
    )
    print("-" * 56)
    for condition in CONDITIONS:
        row = summary[condition]
        if not row.get("n"):
            print(f"{labels[condition]:<20} {'-':>3}  (no scored samples)")
            continue
        print(
            f"{labels[condition]:<20} {row['n']:>3} "
            f"{row['stale_binding_rate'] * 100:>6.1f}% "
            f"{row['stale_per_plan']:>6.2f} "
            f"{row['rebind_rate'] * 100:>7.1f}% "
            f"{row['target_hit_rate'] * 100:>7.1f}%"
        )
    print()
    raw_rate = summary["raw"].get("stale_binding_rate")
    policy_rate = summary["policy"].get("stale_binding_rate")
    if raw_rate is not None and policy_rate is not None:
        delta = raw_rate - policy_rate
        print(
            f"stale-binding delta (B - C): {delta * 100:+.1f} points "
            f"({'policy helps' if delta > 0 else 'no improvement'})"
        )
    for condition in CONDITIONS:
        by_kind = summary[condition].get("stale_by_kind") or {}
        if by_kind:
            print(f"  {labels[condition]} stale by kind: {by_kind}")
    errors = [s for s in samples if s.error]
    if errors:
        print(f"\n{len(errors)} cell(s) failed:")
        for sample in errors[:5]:
            print(f"  {sample.condition}/{sample.task}: {sample.error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).with_name("policy_ab_fixture.json"),
        help="tasks and source experiences (default: policy_ab_fixture.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="actor model for all three conditions (default: project main model)",
    )
    parser.add_argument("--provider", default=None, help="actor model provider")
    parser.add_argument(
        "--policy-model",
        default=None,
        help=(
            "model for rerank and synthesis, i.e. condition C's intermediate work "
            "(default: the project auxiliary model). Pin this to keep the actor "
            "and the reuse layer as independent variables."
        ),
    )
    parser.add_argument(
        "--policy-provider", default=None, help="provider for --policy-model"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="samples per (task, condition) cell; LLM output varies (default: 1)",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help=f"comma-separated subset of {','.join(CONDITIONS)}",
    )
    parser.add_argument(
        "--raw-limit",
        type=int,
        default=4,
        help="records injected in condition B, matched to policy max-selected",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=4,
        help="records the policy layer may select (default: 4)",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=None,
        help="seed store location (default: a temp dir, removed on exit)",
    )
    parser.add_argument("--json", type=Path, default=None, help="write full report")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print condition A and B prompts and exit; no API calls",
    )
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> int:
    tasks, papers = load_fixture(args.fixture)
    records = [
        record for paper in papers for level in ("l1", "l2") for record in paper[level]
    ]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise SystemExit(f"unknown condition(s): {unknown}")

    if args.dry_run:
        # Condition C needs a live model to derive a policy, so a dry run shows
        # only the two conditions whose prompts are known without an API call. We
        # don't have a memory store yet, so render the fixture records directly.
        for task in tasks[:1]:
            for condition in ("none", "raw"):
                if condition not in conditions:
                    continue
                text = (
                    render_raw_records(records, args.raw_limit)
                    if condition == "raw"
                    else ""
                )
                print("=" * 72)
                print(f"condition {condition} / task {task.name}")
                print("=" * 72)
                print(build_prompt(task=task, condition=condition, memory_text=text))
        print(
            f"\n{len(tasks)} tasks x {len(conditions)} conditions x "
            f"{args.repeats} repeats = "
            f"{len(tasks) * len(conditions) * args.repeats} actor calls when run.",
            file=sys.stderr,
        )
        return 0

    from EvoScientist.llm import get_chat_model

    model = get_chat_model(model=args.model, provider=args.provider)
    policy_model = (
        get_chat_model(model=args.policy_model, provider=args.policy_provider)
        if args.policy_model
        else None
    )

    owned_temp = args.memory_dir is None
    memory_dir = Path(
        tempfile.mkdtemp(prefix="policy-ab-") if owned_temp else args.memory_dir
    )
    try:
        experience_ids = seed_store(memory_dir, papers)
        print(
            f"seeded {len(experience_ids)} experience records from "
            f"{len(papers)} papers into {memory_dir}",
            file=sys.stderr,
        )
        cells = [
            (task, condition)
            for _ in range(args.repeats)
            for task in tasks
            for condition in conditions
        ]
        print(f"running {len(cells)} cells...", file=sys.stderr)
        samples = await asyncio.gather(
            *(
                run_cell(
                    task=task,
                    condition=condition,
                    model=model,
                    memory_dir=memory_dir,
                    records=records,
                    raw_limit=args.raw_limit,
                    max_selected=args.max_selected,
                    policy_model=policy_model,
                )
                for task, condition in cells
            )
        )
    finally:
        if owned_temp:
            shutil.rmtree(memory_dir, ignore_errors=True)

    summary = summarize(list(samples))
    print_report(summary, list(samples))
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "fixture": str(args.fixture),
                    "actor_model": args.model or "project default",
                    "policy_model": args.policy_model or "project auxiliary",
                    "repeats": args.repeats,
                    "summary": summary,
                    "samples": [vars(s) for s in samples],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nfull report written to {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
