"""Unit tests for the experience-bank importer/promoter mapping layer.

Covers scripts/experience_mapping.py — the v3-first, legacy-fallback field
mapping shared by the offline importer (import_experience_bank.py) and the online
promoter (promote_to_memory.py). No LLM calls; pure field-shape logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

experience_mapping = pytest.importorskip("experience_mapping")
l1_to_observation = experience_mapping.l1_to_observation
l2_to_observation = experience_mapping.l2_to_observation
_quote = experience_mapping._quote
_first_sentence = experience_mapping._first_sentence


# --- L1 ---------------------------------------------------------------------

def test_l1_v3_requires_statement():
    """A v3 L1 record with no statement/narrative is unusable."""
    assert l1_to_observation({"task": "x"}, "p") is None


def test_l1_v3_maps_core_fields():
    rec = {
        "statement": "S" * 400,
        "task": "train",
        "evidence": [{"section": "results", "quote": "verbatim-quote"}],
        "domain": "agent_learning",
        "scope": "broad applicability",
    }
    obs = l1_to_observation(rec, "1234.5")
    assert obs is not None
    assert "verbatim-quote" in (obs["evidence"] or "")


def test_l1_v3_scope_is_not_granularity_prefix():
    """v3 `scope` is applicability, not a coarse/medium/fine grain — no prefix."""
    obs = l1_to_observation({"statement": "S" * 400, "scope": "broad"}, "p")
    assert not obs["summary"].startswith("[broad]")


def test_l1_legacy_fallback_and_granularity_prefix():
    rec = {
        "narrative": "N" * 400,
        "t": {"summary": "legacy task"},
        "source_quote": "old-quote",
        "granularity": "fine",
    }
    obs = l1_to_observation(rec, "p")
    assert obs is not None
    assert obs["summary"].startswith("[fine]")
    assert "old-quote" in (obs["evidence"] or "")


# --- L2 ---------------------------------------------------------------------

def test_l2_v3_confidence_and_rationale():
    rec = {
        "statement": "claim",
        "confidence": "high",
        "rationale": "because-mechanism",
        "claim_type": "trend",
    }
    obs = l2_to_observation(rec, "p")
    assert obs is not None
    assert "high" in obs["why_it_matters"]
    assert "because-mechanism" in obs["observation"]
    assert obs["summary"].startswith("[trend]")


def test_l2_legacy_mu_r_fallback():
    rec = {"declaration": "D", "μ": "medium", "r": "cause", "narrative": "body"}
    obs = l2_to_observation(rec, "p")
    assert obs is not None
    assert "medium" in obs["why_it_matters"]


def test_l2_empty_declaration_is_none():
    assert l2_to_observation({"confidence": "high"}, "p") is None


# --- helpers ----------------------------------------------------------------

def test_quote_joins_multiple_evidence():
    q = _quote({"evidence": [{"quote": "aa", "section": "s1"}, {"quote": "bb"}]})
    assert "aa" in q and "bb" in q


def test_quote_legacy_flat_field():
    assert _quote({"source_quote": "flat"}) == "flat"


def test_first_sentence_truncates():
    assert _first_sentence("First. Second.") == "First."
