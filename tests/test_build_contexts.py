"""Unit tests for solution-ab context building (denoising).

Covers skills/solution-ab/scripts/build_contexts.py strip_tail() — removing
References/Appendix tails so the A/B size comparison isn't skewed by
non-content. No network; pure text logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "solution-ab"
    / "scripts"
)
sys.path.insert(0, str(_SCRIPTS))

build_contexts = pytest.importorskip("build_contexts")
strip_tail = build_contexts.strip_tail


def test_strip_references_section():
    body = "# Intro\nreal content here\n## References\n[1] foo\n[2] bar\n"
    kept, removed = strip_tail(body)
    assert "real content here" in kept
    assert "[1] foo" not in kept
    assert removed > 0


def test_no_tail_keeps_everything():
    body = "# Intro\njust content\nno refs here\n"
    kept, removed = strip_tail(body)
    assert kept.strip() == body.strip()
    assert removed == 0


def test_strip_appendix():
    body = "main body\n## Appendix A\nextra appendix text\n"
    kept, _ = strip_tail(body)
    assert "main body" in kept
    assert "extra appendix text" not in kept


def test_inline_references_word_not_triggered():
    """The word 'references' mid-sentence must not truncate — heading-only match."""
    body = "we cite many references in this work\nmore real text\n"
    kept, removed = strip_tail(body)
    assert removed == 0
    assert "more real text" in kept
