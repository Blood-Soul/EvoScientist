"""Policy storage: disk cache plus audit trail.

Synthesis is an aux-model call reading ~20K chars and writing ~2K. Repeating
that on every tool invocation for the same (task, selected-records) pair is
wasteful and makes the tool slow. Caching on a stable key makes the second ask
instant while preserving the audit trail for later outcome tracking.

The key hashes (task text, sorted selected E-* IDs). A task rephrased yields a
new key and a new synthesis pass, which is correct -- the writer reads the task
literally, so changed wording can change the policy. That boundary keeps the
cache simple while staying safe.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .._atomic import atomic_write_json, read_json

STORE_VERSION = 1
POLICY_DIR = "policies/projects"


def _policy_cache_key(*, task: str, selected_ids: Sequence[str]) -> str:
    """Stable hash of the synthesis inputs."""
    canonical = json.dumps(
        {"task": task.strip(), "selected": sorted(selected_ids)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _policy_path(*, memory_dir: str | Path, project_id: str, key: str) -> Path:
    return (
        Path(memory_dir).expanduser() / POLICY_DIR / project_id / f"policy-{key}.json"
    )


def load_cached_policy(
    *,
    memory_dir: str | Path,
    project_id: str,
    task: str,
    selected_ids: Sequence[str],
) -> dict[str, Any] | None:
    """Read one cached policy if it exists and matches the current inputs."""
    key = _policy_cache_key(task=task, selected_ids=selected_ids)
    path = _policy_path(memory_dir=memory_dir, project_id=project_id, key=key)
    stored = read_json(path)
    if stored is None or stored.get("store_version") != STORE_VERSION:
        return None
    return stored.get("policy")


def store_policy(
    *,
    memory_dir: str | Path,
    project_id: str,
    task: str,
    selected_ids: Sequence[str],
    policy: Mapping[str, Any],
    model_name: str,
) -> Path:
    """Persist one synthesis result, indexed by its stable cache key."""
    key = _policy_cache_key(task=task, selected_ids=selected_ids)
    path = _policy_path(memory_dir=memory_dir, project_id=project_id, key=key)
    atomic_write_json(
        path,
        {
            "store_version": STORE_VERSION,
            "cache_key": key,
            "project_id": project_id,
            "task": task.strip(),
            "selected_ids": sorted(selected_ids),
            "policy": dict(policy),
            "synthesized_at": datetime.now(UTC).isoformat(),
            "synthesized_by": model_name,
        },
    )
    return path


__all__ = [
    "STORE_VERSION",
    "load_cached_policy",
    "store_policy",
]
