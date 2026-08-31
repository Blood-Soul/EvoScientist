#!/usr/bin/env python3
"""Pretty-print the `apply_experience` debug trace for a meeting walkthrough.

`derive_policy()` writes one JSON line per stage (see
`EvoScientist/memory/policy/trace.py`) when `EVOSCIENTIST_POLICY_TRACE=1`:
`request` -> `retrieve` -> `rerank` -> `synthesize` -> `report`, all sharing one
`call_id`. This script groups those lines back into calls and renders each one
as a readable "what happened" narrative: the task asked, which `E-*` records
were recalled, what the reranker kept and why, the synthesis model's raw output
before parsing next to the normalized policy, and the final report handed back
to the agent.

This is a throwaway dev tool, not a supported CLI: no config wiring, no tests
beyond what's already covering `trace.py` itself. Delete it once prompt
iteration on the policy layer settles.

Usage:
    EVOSCIENTIST_POLICY_TRACE=1 <run the agent>
    scripts/policy_trace_view.py                       # last call, default path
    scripts/policy_trace_view.py --memory-dir ~/.evoscientist/memories/proj
    scripts/policy_trace_view.py --last 3               # last 3 calls
    scripts/policy_trace_view.py --call-id a1b2c3d4e5f6
    scripts/policy_trace_view.py --follow                # like tail -f
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_RELATIVE_PATH = "policies/trace.jsonl"


def resolve_path(*, memory_dir: Path | None, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    if memory_dir is not None:
        return memory_dir / DEFAULT_RELATIVE_PATH
    raise SystemExit("pass --memory-dir or --path so I know which trace to read")


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"no trace file at {path}. Was EVOSCIENTIST_POLICY_TRACE=1 set, and "
            "has apply_experience actually run since?"
        )
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def group_by_call(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group events by `call_id`, preserving file order within each call.

    Older lines predating `call_id` (if any survive on disk) are dropped --
    they cannot be joined into a single call's narrative.
    """
    calls: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        call_id = event.get("call_id")
        if not call_id:
            continue
        calls.setdefault(call_id, []).append(event)
    return calls


def _truncate(text: str, limit: int = 2000) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return (
        text[:limit] + f"\n... [{len(text) - limit} more chars, truncated for display]"
    )


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def render_call(call_id: str, events: list[dict[str, Any]], *, full: bool) -> str:
    """Render one `apply_experience` invocation as a step-by-step narrative."""
    by_event = {e["event"]: e for e in events}
    limit = 10**9 if full else 2000
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(f"call {call_id}")
    lines.append("=" * 78)

    request = by_event.get("request")
    if request:
        lines.append(f"task:    {request.get('task', '')}")
        state = request.get("state") or "[no project state provided]"
        lines.append(f"state:   {state}")
        lines.append(
            f"params:  max_selected={request.get('max_selected')} "
            f"refresh={request.get('refresh')} project={request.get('project_id')}"
        )

    retrieve = by_event.get("retrieve")
    lines.append("\n-- 1. retrieve (TF-IDF over stored E-* records) --")
    if retrieve:
        ids = retrieve.get("candidate_ids") or []
        if not ids:
            lines.append("  no candidates matched this query")
        else:
            descriptors = retrieve.get("descriptors") or {}
            for eid in ids:
                lines.append(f"  [{eid}] {descriptors.get(eid, '')}")
    else:
        lines.append("  (no retrieve event -- call ended before or during retrieval)")

    rerank = by_event.get("rerank")
    lines.append("\n-- 2. rerank (aux model selects the working set) --")
    if rerank:
        lines.append(f"  selected: {rerank.get('selected_ids')}")
        lines.append(f"  reason:   {rerank.get('reason')}")
        lines.append("  raw model output:")
        lines.append(_indent(_truncate(rerank.get("raw_output", ""), limit)))
    else:
        lines.append("  (no rerank event)")

    synthesize = by_event.get("synthesize")
    lines.append(
        "\n-- 3. synthesize (aux model rewrites selected records into a policy) --"
    )
    if synthesize:
        lines.append(f"  parsed ok: {synthesize.get('parsed')}")
        lines.append("  raw model output (the actual 改写, before parsing):")
        lines.append(_indent(_truncate(synthesize.get("raw_output", ""), limit)))
    else:
        lines.append(
            "  (no synthesize event -- served from cache, or call stopped earlier)"
        )

    report = by_event.get("report")
    lines.append(
        "\n-- 4. final report (what apply_experience hands back to the agent) --"
    )
    if report:
        lines.append(
            f"  status: {report.get('status')}  cached: {report.get('cached')}"
        )
        policy = report.get("policy")
        if policy is not None:
            lines.append(
                "  policy (this is what gets injected into the agent's context):"
            )
            lines.append(
                _indent(
                    _truncate(
                        json.dumps(
                            policy, ensure_ascii=False, indent=2, sort_keys=True
                        ),
                        limit,
                    )
                )
            )
        elif report.get("hint"):
            lines.append(f"  hint: {report.get('hint')}")
    else:
        lines.append("  (no report event -- call did not finish)")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--memory-dir", type=Path, default=None, help="project memory dir"
    )
    parser.add_argument(
        "--path", type=Path, default=None, help="explicit trace file path"
    )
    parser.add_argument(
        "--last", type=int, default=1, help="show the last N calls (default: 1)"
    )
    parser.add_argument("--call-id", default=None, help="show one specific call")
    parser.add_argument(
        "--full", action="store_true", help="don't truncate long fields"
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="like tail -f: print each new call as it completes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = resolve_path(memory_dir=args.memory_dir, explicit_path=args.path)

    if args.follow:
        print(f"following {path} (ctrl-C to stop)...", file=sys.stderr)
        seen: set[str] = set()
        try:
            while True:
                if path.exists():
                    events = read_events(path)
                    calls = group_by_call(events)
                    for call_id, call_events in calls.items():
                        if call_id in seen:
                            continue
                        if any(e["event"] == "report" for e in call_events):
                            print(render_call(call_id, call_events, full=args.full))
                            print()
                            seen.add(call_id)
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    events = read_events(path)
    calls = group_by_call(events)
    if args.call_id:
        if args.call_id not in calls:
            raise SystemExit(f"no call {args.call_id} in {path}")
        print(render_call(args.call_id, calls[args.call_id], full=args.full))
        return 0

    # File order is chronological (append-only), so the last N keys are the
    # most recent calls.
    ordered_ids = list(calls.keys())[-args.last :]
    for call_id in ordered_ids:
        print(render_call(call_id, calls[call_id], full=args.full))
        print()
    if not ordered_ids:
        print(f"trace file {path} has no complete calls yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
