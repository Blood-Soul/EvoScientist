"""Plain structured data (dicts, lists, scalars) as indented text for a model to read.

For results whose shape has no dedicated renderer. A mapping becomes `key: value` lines, a list becomes `-` items,
and a multi-line string keeps its line breaks instead of being escaped into one line. Empty values are dropped so the
reader does not wade through `null` and `[]`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_PAD = "  "


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(getattr(value, "value", value))


def _empty(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, (Mapping, Sequence)) and not isinstance(value, str) and len(value) == 0)


def render_data(value: Any, *, indent: int = 0) -> str:
    """Return `value` as text; see the module docstring."""
    pad = _PAD * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            if _empty(item):
                continue
            if isinstance(item, Mapping) or (isinstance(item, Sequence) and not isinstance(item, str | bytes)):
                lines.append(f"{pad}{key}:")
                lines.append(render_data(item, indent=indent + 1))
            else:
                text = _scalar(item)
                if "\n" in text:
                    lines.append(f"{pad}{key}:")
                    lines.extend(f"{pad}{_PAD}{part}" for part in text.splitlines())
                else:
                    lines.append(f"{pad}{key}: {text}")
        return "\n".join(lines)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        lines = []
        for item in value:
            if _empty(item):
                continue
            if isinstance(item, Mapping):
                body = render_data(item, indent=indent + 1).splitlines()
                if body:
                    lines.append(f"{pad}- {body[0].lstrip()}")
                    lines.extend(body[1:])
            elif isinstance(item, Sequence) and not isinstance(item, str | bytes):
                lines.append(f"{pad}-")
                lines.append(render_data(item, indent=indent + 1))
            else:
                text = _scalar(item)
                first, *rest = text.splitlines() or [""]
                lines.append(f"{pad}- {first}")
                lines.extend(f"{pad}{_PAD}{part}" for part in rest)
        return "\n".join(lines)
    return f"{pad}{_scalar(value)}"


__all__ = ["render_data"]
