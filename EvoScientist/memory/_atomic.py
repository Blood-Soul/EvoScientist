"""Atomic file writes shared by the file-backed memory stores.

Both the experience store and the paper full-text store persist derived
artifacts that must never be observed half-written by a concurrent reader
(a background worker writes while a live agent searches). Every write goes
to a uniquely-named temporary sibling first, then lands with ``os.replace``,
which is atomic on POSIX and on Windows for same-directory replacement.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as sorted, indented JSON, replacing ``path`` atomically."""
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` as UTF-8 text, replacing ``path`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Read one JSON object, returning None for missing or malformed files."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None
