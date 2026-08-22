"""Unit tests for the knowledge-graph edge planner.

Covers scripts/link_observations_by_rule.py plan_edges() — the rule-based
backfill that links bulk-imported observations (same-paper L1<->L2 bridges and
cross-paper same-domain chains) so the graph isn't 96% isolated nodes. Pure
in-memory logic; no store access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

linker = pytest.importorskip("link_observations_by_rule")
plan_edges = linker.plan_edges
Node = linker.Node


def _node(oid, paper, level="l2", domain="agent_learning"):
    mtype = "procedural" if level == "l1" else "semantic"
    return Node(oid, paper, mtype, level, domain, False)


def test_same_paper_l1_l2_bridged():
    nodes = [_node("O-1", "2201.1", "l1"), _node("O-2", "2201.1", "l2")]
    edges = plan_edges(nodes, max_per_paper=8)
    assert len(edges) >= 1
    assert any("2201.1" in reason for _, _, reason in edges)


def test_cross_paper_same_domain_linked():
    nodes = [_node("O-1", "paperA"), _node("O-2", "paperB")]
    edges = plan_edges(nodes, max_per_paper=8)
    assert any(
        "paperA" in reason and "paperB" in reason for _, _, reason in edges
    )


def test_per_paper_cap_enforced():
    many = [_node(f"O-{i}", "P") for i in range(20)]
    edges = plan_edges(many, max_per_paper=8)
    same_paper = [e for e in edges if "different papers" not in e[2]]
    assert len(same_paper) <= 8


def test_no_duplicate_edges():
    many = [_node(f"O-{i}", "P") for i in range(20)]
    edges = plan_edges(many, max_per_paper=8)
    pairs = [(a, b) for a, b, _ in edges]
    assert len(pairs) == len(set(pairs))
