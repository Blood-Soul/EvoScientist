"""Plans and analyses: the two documents around a node.

A *plan* is written by a sub-agent (the plan-writer) for one direction and
reviewed by the main agent before a node is opened on it. An *analysis* is
written by another sub-agent (the node-analyst) after a node has been scored:
what was worth keeping, what is wrong, what to change. The main agent records
both with `tree_plan` and `tree_analysis`, pasting the sub-agent's text as it
came back, so the reply it reviewed is the reply that is kept.

Files live in `reports/` at the workspace root and are excluded from git, so
they never mix into a node's commit. Whenever a document is rewritten the
version it replaces is copied to `.research/reports_history/`, so earlier
versions stay readable.
"""

from __future__ import annotations

from pathlib import Path

from .gitlog import STATE_DIR

REPORTS_DIR = "reports"
HISTORY_DIR = "reports_history"
MAX_PLAN_REVISIONS = 2  # rewrites of one plan after its first version
PLAN_DECISIONS = ("accepted", "rejected", "revise")


def report_path(root: Path, kind: str, doc_id: str) -> Path:
    """`reports/plans/P1.md` or `reports/analyses/A1.md`."""
    return root / REPORTS_DIR / kind / f"{doc_id}.md"


def write_report(root: Path, kind: str, doc_id: str, text: str, version: int) -> Path:
    """Write the document; keep the version it replaces. `version` counts from 1."""
    path = report_path(root, kind, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and version > 1:
        old = root / STATE_DIR / HISTORY_DIR / f"{doc_id}.v{version - 1}.md"
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text(path.read_text())
    path.write_text(text.rstrip() + "\n")
    return path


def short(text: str, limit: int = 160) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[: limit - 1] + "…"


def render_plans(plans: dict[str, dict]) -> list[str]:
    """Status lines for the plans, oldest first."""
    lines = []
    for p in sorted(plans.values(), key=lambda p: int(p["id"][1:])):
        tail = f" -> {p['node']}" if p.get("node") else ""
        rounds = f", rewritten {p['rounds']}x" if p.get("rounds") else ""
        lines.append(
            f"  {p['id']} {p['status']:<8} family {p.get('cluster') or '-'} | from "
            f"{','.join(p['parents']) or 'root'}{rounds}{tail} | {short(p['title'], 80)}"
        )
    return lines


def render_analyses(analyses: dict[str, dict]) -> list[str]:
    lines = []
    for a in sorted(analyses.values(), key=lambda a: int(a["id"][1:])):
        lines.append(
            f"  {a['id']} of {a['node']}: {len(a['flaws'])} flaws, "
            f"{len(a['improvements'])} improvements"
            + (f" | first flaw: {short(a['flaws'][0], 90)}" if a["flaws"] else "")
        )
    return lines


def render_analysis_for_brief(a: dict) -> str:
    """What a refine node's sub-agent is told about the parent's analysis."""
    parts = [f"Analysis {a['id']} of {a['node']}:"]
    if a["strengths"]:
        parts.append("- worth keeping: " + "; ".join(a["strengths"]))
    if a["flaws"]:
        parts.append("- flaws: " + "; ".join(a["flaws"]))
    if a["improvements"]:
        parts.append("- improvements suggested: " + "; ".join(a["improvements"]))
    return "\n".join(parts)
