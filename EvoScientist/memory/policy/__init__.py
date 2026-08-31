"""Experience-to-policy transformation layer.

This package implements the query-conditioned reuse (QCR) operation that
converts stored paper experiences into target-bound policies. It adds the
missing step between retrieval and action: an experience record describes what
one paper's authors did in their setting; a policy describes what the current
task should do, listing the source-side values that must be re-derived rather
than copied.

Entry point: ``derive_policy`` runs the full pipeline: retrieve → rerank →
synthesize → cache. The tool at ``tools/experience_policy.py`` wraps it.
"""

from __future__ import annotations

from .pipeline import derive_policy
from .prompts import load_rerank_prompt, load_writer_prompt
from .schema import BINDING_KINDS, VERDICTS, PolicyOutputError, normalize_policy
from .select import (
    DEFAULT_MAX_SELECTED,
    DEFAULT_RETRIEVE_LIMIT,
    gather_candidates,
    rerank_candidates,
    transferable_core,
)
from .store import STORE_VERSION, load_cached_policy, store_policy
from .synthesize import parse_policy_json, synthesize_policy

__all__ = [
    "BINDING_KINDS",
    "DEFAULT_MAX_SELECTED",
    "DEFAULT_RETRIEVE_LIMIT",
    "STORE_VERSION",
    "VERDICTS",
    "PolicyOutputError",
    "derive_policy",
    "gather_candidates",
    "load_cached_policy",
    "load_rerank_prompt",
    "load_writer_prompt",
    "normalize_policy",
    "parse_policy_json",
    "rerank_candidates",
    "store_policy",
    "synthesize_policy",
    "transferable_core",
]
