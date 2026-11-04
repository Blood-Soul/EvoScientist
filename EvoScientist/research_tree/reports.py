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

import re
from pathlib import Path

from .errors import TreeError
from .gitlog import STATE_DIR

REPORTS_DIR = "reports"
HISTORY_DIR = "reports_history"
MAX_PLAN_REVISIONS = 2  # rewrites of one plan after its first version
PLAN_DECISIONS = ("accepted", "rejected", "revise")


EXPERIENCE_ID = re.compile(r"\bE-[0-9a-f]{6,}\b")
_HEADING = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
_BULLET = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+(.*)$")
_EMPTY_ITEMS = {"none", "n/a", "nothing", "-"}


def read_draft(root: Path, rel: str) -> str:
    """Text of a document a sub-agent wrote, addressed relative to the workspace root."""
    if not rel or not rel.strip():
        raise TreeError("path is required: where the sub-agent wrote the document")
    base = Path(root).resolve()
    target = (base / rel.strip().lstrip("/")).resolve()
    if not target.is_relative_to(base):
        raise TreeError(f"{rel!r} is outside the workspace")
    if not target.is_file():
        raise TreeError(
            f"no file at {rel}; the sub-agent should have written its document there (ask it for the path)"
        )
    text = target.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        raise TreeError(f"{rel} is empty")
    return text


def section_items(text: str, *names: str) -> list[str]:
    """Bullet (or numbered) items under the first `## <name>` heading.

    Matching is case-insensitive on the start of the heading. A section written
    as plain prose becomes one item; items such as "none" are dropped.
    """
    wanted = tuple(n.lower() for n in names)
    items: list[str] = []
    prose: list[str] = []
    inside = False
    for line in text.splitlines():
        head = _HEADING.match(line)
        if head:
            if inside:
                break
            inside = len(head.group(1)) == 2 and head.group(2).lower().startswith(wanted)
            continue
        if not inside:
            continue
        s = line.strip()
        bullet = _BULLET.match(s)
        if bullet:
            items.append(bullet.group(1).strip())
        elif s and items:
            items[-1] += " " + s  # a wrapped line of the previous item
        elif s:
            prose.append(s)
    if not items and prose:
        items = [" ".join(prose)]
    return [x for x in items if x.lower().strip(" .") not in _EMPTY_ITEMS]


def plan_title(text: str) -> str:
    m = re.search(r"^#\s*Plan:\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else ""


def experience_ids(text: str) -> list[str]:
    """E-ids named under `## Experience used`."""
    return sorted(set(EXPERIENCE_ID.findall("\n".join(section_items(text, "Experience used")))))


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
