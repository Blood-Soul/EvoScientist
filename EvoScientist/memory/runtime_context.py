"""Shared resolution of the active project id from tool runtime config.

Every project-scoped tool needs the same answer to the same question: which
project is this call for? The id is injected through LangGraph's
``configurable`` map rather than passed as a model-visible argument, so each
tool would otherwise repeat the same defensive unwrapping. It lives here once.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def runtime_config_value(runtime: Any | None, key: str) -> str | None:
    """Read one optional non-empty string from runtime configurable config."""
    if runtime is None:
        return None
    config = getattr(runtime, "config", None) or {}
    if not isinstance(config, Mapping):
        return None
    configurable = config.get("configurable", {})
    if not isinstance(configurable, Mapping):
        return None
    value = configurable.get(key)
    return value if isinstance(value, str) and value.strip() else None


def runtime_project_id(runtime: Any | None, default_project_id: str) -> str:
    """Return the runtime's project id, falling back to the build-time default.

    The default is the project resolved when the tool was constructed, which is
    correct for a single-workspace CLI run; the runtime value wins so one server
    process can serve several projects.
    """
    return runtime_config_value(runtime, "evomemory_project_id") or default_project_id
