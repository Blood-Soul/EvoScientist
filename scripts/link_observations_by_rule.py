"""Link bulk-imported observations so the knowledge graph is not 96% isolated.

The observation store's graph edges are written by the background
``observation_linker`` agent, which only sees observations produced during a
conversation. Everything imported by script — the offline experience bank and the
paper-experience promotions — bypassed it, leaving ~96% of nodes with no edges at
all. A graph of 360 loose points renders as noise regardless of layout.

This backfills edges from facts already in the data rather than guessing at
semantics:

  1. same-paper        — experiences extracted from one paper describe different
                         facets of the same system, so they are genuinely related.
  2. L1<->L2 bridge    — within a paper, the practical recipe and the inductive
                         claim it supports are the most informative pairing.

Both use ``complements``; nothing here can establish ``contradicts`` or
``supersedes``, which need semantic judgement the linker agent provides.

Usage:
    python scripts/link_observations_by_rule.py --dry-run
    python scripts/link_observations_by_rule.py [--max-per-paper 8]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from EvoScientist import paths  # noqa: E402
from EvoScientist.memory import link_observation_files  # noqa: E402
from EvoScientist.memory.types import ObservationRelation  # noqa: E402

ARXIV_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5})")
ID_RE = re.compile(r'^id:\s*"?(O-[0-9a-f]+)"?', re.MULTILINE)
TYPE_RE = re.compile(r"^memory_type:\s*(\w+)", re.MULTILINE)
SESSION_RE = re.compile(r'session_id:\s*"?([^"\n]+)"?')
AGENT_RE = re.compile(r'agent:\s*"([^"]+)"')
LINKED_RE = re.compile(r"^related_observations:", re.MULTILINE)
DOMAIN_RE = re.compile(r"^domain:\s*(agent_\w+)", re.MULTILINE)

BULK_AGENTS = {"experience-bank", "paper-experience"}


class Node:
    __slots__ = ("obs_id", "paper", "mtype", "level", "domain", "linked")

    def __init__(
        self,
        obs_id: str,
        paper: str,
        mtype: str,
        level: str,
        domain: str,
        linked: bool,
    ):
        self.obs_id = obs_id
        self.paper = paper
        self.mtype = mtype
        self.level = level
        self.domain = domain
        self.linked = linked


def scan(memory_dir: Path) -> list[Node]:
    root = Path(memory_dir) / "observations" / "global"
    nodes: list[Node] = []
    for path in sorted(root.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        agent = AGENT_RE.search(text)
        if not agent or agent.group(1) not in BULK_AGENTS:
            continue  # leave agent-authored nodes to the linker
        obs_id = ID_RE.search(text)
        paper = ARXIV_RE.search(text)
        if not obs_id or not paper:
            continue
        mtype = TYPE_RE.search(text)
        session = SESSION_RE.search(text)
        level = "l2" if (session and session.group(1).endswith("l2")) else "l1"
        domain = DOMAIN_RE.search(text)
        nodes.append(
            Node(
                obs_id.group(1),
                paper.group(1),
                mtype.group(1) if mtype else "?",
                level,
                domain.group(1) if domain else "",
                bool(LINKED_RE.search(text)),
            )
        )
    return nodes


def plan_edges(nodes: list[Node], max_per_paper: int) -> list[tuple[str, str, str]]:
    """Return (source, target, reason) edges, deduped and capped per paper."""
    by_paper: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        by_paper[node.paper].append(node)

    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    for paper, group in sorted(by_paper.items()):
        if len(group) < 2:
            continue
        l1 = [n for n in group if n.level == "l1"]
        l2 = [n for n in group if n.level == "l2"]

        # Bridge L1<->L2 first: the practice and the claim it supports are the
        # most informative pairing, so they get priority under the cap.
        ranked: list[tuple[Node, Node]] = []
        for a in l1:
            for b in l2:
                ranked.append((a, b))
        # Then same-level pairs, which are weaker but keep single-level papers
        # from staying isolated.
        for same in (l1, l2):
            ranked.extend(combinations(same, 2))

        added = 0
        for a, b in ranked:
            if added >= max_per_paper:
                break
            key = tuple(sorted((a.obs_id, b.obs_id)))
            if key in seen:
                continue
            seen.add(key)
            if a.level != b.level:
                reason = (
                    f"Both extracted from arXiv:{paper}: the {a.level.upper()} "
                    f"practice record and the {b.level.upper()} inductive claim "
                    "describe the same work from complementary angles "
                    "(what was done vs. what it generalizes to)."
                )
            else:
                reason = (
                    f"Both extracted from arXiv:{paper} at the {a.level.upper()} "
                    "level, covering different facets of the same system."
                )
            edges.append((a.obs_id, b.obs_id, reason))
            added += 1

    # Rule 3 — cross-paper, same domain. Same-paper edges only ever produce 51
    # disconnected islands; these are the edges that actually make the graph
    # navigable, letting a reader move between papers that attack one problem.
    # Chain within each domain rather than connecting all pairs: a clique over a
    # 40-node domain is 780 edges of hairball, a chain is 39 and still connected.
    by_domain: dict[str, list[Node]] = defaultdict(list)
    for node in nodes:
        if node.domain:
            by_domain[node.domain].append(node)

    for domain, group in sorted(by_domain.items()):
        # One representative per paper keeps the chain across papers, not within.
        seen_papers: set[str] = set()
        reps: list[Node] = []
        for node in sorted(group, key=lambda n: (n.paper, n.level)):
            if node.paper not in seen_papers:
                seen_papers.add(node.paper)
                reps.append(node)
        if len(reps) < 2:
            continue
        for a, b in zip(reps, reps[1:], strict=False):
            key = tuple(sorted((a.obs_id, b.obs_id)))
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                (
                    a.obs_id,
                    b.obs_id,
                    f"Both classified under {domain}, from arXiv:{a.paper} and "
                    f"arXiv:{b.paper} respectively — different papers addressing "
                    "the same problem area.",
                )
            )
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument(
        "--max-per-paper",
        type=int,
        default=8,
        help="Cap edges added per paper (default 8) to avoid dense hairballs",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    memory_dir = Path(args.memory_dir) if args.memory_dir else paths.MEMORIES_DIR
    nodes = scan(memory_dir)
    papers = {n.paper for n in nodes}
    already = sum(1 for n in nodes if n.linked)

    print(f"MEMORIES_DIR: {memory_dir}")
    print(f"bulk-imported nodes: {len(nodes)}  papers: {len(papers)}")
    print(f"  already linked: {already}   isolated: {len(nodes) - already}")

    edges = plan_edges(nodes, args.max_per_paper)
    print(f"planned edges: {len(edges)} (cap {args.max_per_paper}/paper)\n")

    if args.dry_run:
        for src, tgt, reason in edges[:5]:
            print(f"  {src} <-> {tgt}\n    {reason[:100]}…")
        print(f"\nDry-run: {len(edges)} edges would be written (nothing written).")
        return

    written = failed = 0
    for src, tgt, reason in edges:
        try:
            link_observation_files(
                memory_dir=memory_dir,
                project_id="",
                source_observation_id=src,
                target_observation_id=tgt,
                reason=reason,
                relation=ObservationRelation.COMPLEMENTS,
                bidirectional=True,
            )
            written += 1
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"  failed {src} <-> {tgt}: {exc}", file=sys.stderr)

    print("=== Summary ===")
    print(f"edges written: {written}")
    if failed:
        print(f"failed:        {failed}")


if __name__ == "__main__":
    main()
