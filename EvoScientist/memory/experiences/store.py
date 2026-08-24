"""File-backed, project-isolated paper experience storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..types import (
    ExperienceLevel,
    MemoryScope,
    MemoryType,
    ObservationReadResult,
    ObservationSearchDocument,
)

EXPERIENCE_DIR = "experiences/projects"
EXPERIENCE_CATALOG_FILENAME = "PAPER_EXPERIENCES.md"
STORE_VERSION = 1


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_paper_identifier(value: str) -> str:
    """Normalize common paper URLs just enough for stable deduplication."""
    normalized = value.strip()
    normalized = re.sub(r"v\d+(?=(?:\.pdf)?(?:[?#].*)?$)", "", normalized)
    normalized = normalized.removesuffix(".pdf").rstrip("/")
    return normalized.casefold()


def paper_storage_key(paper_id: str, url: str) -> str:
    canonical = canonical_paper_identifier(paper_id or url)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", canonical).strip("._-") or "paper"
    return f"{slug[:48]}-{_sha256(canonical)[:12]}"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _paper_dir(
    memory_dir: str | Path, *, project_id: str, paper_id: str, url: str
) -> Path:
    return (
        Path(memory_dir).expanduser()
        / EXPERIENCE_DIR
        / project_id
        / paper_storage_key(paper_id, url)
    )


def experience_catalog_path(*, memory_dir: str | Path, project_id: str) -> Path:
    """Return the WebUI-visible derived catalog for one project's experiences."""
    return (
        Path(memory_dir).expanduser()
        / "profile"
        / "projects"
        / project_id
        / EXPERIENCE_CATALOG_FILENAME
    )


def _catalog_scalar(value: Any, *, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    compact = " ".join(value.split())
    return compact.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def refresh_experience_catalog(*, memory_dir: str | Path, project_id: str) -> Path:
    """Rebuild the read-only Markdown projection shown by the existing WebUI."""
    root = Path(memory_dir).expanduser()
    project_root = root / EXPERIENCE_DIR / project_id
    papers: list[tuple[str, Path, dict[str, Any]]] = []
    try:
        paper_dirs = sorted(path for path in project_root.iterdir() if path.is_dir())
    except OSError:
        paper_dirs = []

    for directory in paper_dirs:
        metadata = _read_json(directory / "metadata.json")
        if metadata is None or metadata.get("project_id") != project_id:
            continue
        extracted_at = str(metadata.get("extracted_at") or "")
        papers.append((extracted_at, directory, metadata))
    papers.sort(key=lambda item: (item[0], item[1].name), reverse=True)

    lines = [
        "# Paper experiences",
        "",
        "Extracted paper experience saved for this project. This file is a "
        "read-only catalog generated from the isolated experience store.",
        "",
        f"Papers: {len(papers)}",
        "",
    ]
    for extracted_at, directory, metadata in papers:
        title = _catalog_scalar(
            metadata.get("title"),
            fallback=_catalog_scalar(metadata.get("paper_id"), fallback=directory.name),
        )
        paper_id = _catalog_scalar(metadata.get("paper_id"))
        url = _catalog_scalar(metadata.get("url"))
        lines.extend([f"## {title}", ""])
        if paper_id:
            lines.append(f"- Paper ID: `{paper_id.replace('`', '')}`")
        if url:
            lines.append(f"- URL: {url}")
        if extracted_at:
            lines.append(f"- Extracted: {extracted_at}")
        lines.append("")

        found_experiences = False
        for level in ("l1", "l2"):
            payload = _read_json(directory / f"{level}.json")
            experiences = payload.get("experiences") if payload else None
            items = experiences if isinstance(experiences, list) else []
            lines.extend([f"### {level.upper()} experiences ({len(items)})", ""])
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                found_experiences = True
                experience_id = _experience_id(
                    project_id=project_id,
                    paper_key=directory.name,
                    level=level,
                    index=index,
                )
                summary = _catalog_scalar(
                    _summary(item, title=title, level=level),
                    fallback=f"{level.upper()} experience",
                )
                lines.append(f"- `{experience_id}`: {summary}")
            if not items:
                lines.append("- No experience records.")
            lines.append("")
        if not found_experiences:
            lines.extend(["No valid experience records were parsed.", ""])

    target = experience_catalog_path(memory_dir=root, project_id=project_id)
    _atomic_write_text(target, "\n".join(lines).rstrip() + "\n")
    return target


def refresh_all_experience_catalogs(*, memory_dir: str | Path) -> list[Path]:
    """Backfill WebUI catalogs for every project already holding experiences."""
    root = Path(memory_dir).expanduser()
    projects_root = root / EXPERIENCE_DIR
    try:
        project_dirs = sorted(
            path
            for path in projects_root.iterdir()
            if path.is_dir() and not path.is_symlink()
        )
    except OSError:
        return []

    catalogs: list[Path] = []
    for project_dir in project_dirs:
        try:
            catalogs.append(
                refresh_experience_catalog(
                    memory_dir=root,
                    project_id=project_dir.name,
                )
            )
        except OSError:
            continue
    return catalogs


def store_paper_experiences(
    *,
    memory_dir: str | Path,
    project_id: str,
    paper_id: str,
    url: str,
    title: str,
    paper_text: str,
    prompts: Mapping[ExperienceLevel, str],
    payloads: Mapping[ExperienceLevel, Mapping[str, Any]],
) -> Path:
    """Atomically persist prompt-compatible L1 and L2 JSON payloads."""
    directory = _paper_dir(
        memory_dir, project_id=project_id, paper_id=paper_id, url=url
    )
    now = datetime.now(UTC).isoformat()
    for level in ("l1", "l2"):
        _atomic_write_json(directory / f"{level}.json", payloads[level])
    _atomic_write_json(
        directory / "metadata.json",
        {
            "store_version": STORE_VERSION,
            "project_id": project_id,
            "paper_id": paper_id,
            "canonical_paper_id": canonical_paper_identifier(paper_id or url),
            "title": title,
            "url": url,
            "paper_sha256": _sha256(paper_text),
            "prompt_sha256": {level: _sha256(prompts[level]) for level in ("l1", "l2")},
            "extracted_at": now,
        },
    )
    refresh_experience_catalog(memory_dir=memory_dir, project_id=project_id)
    return directory


def _experience_id(
    *, project_id: str, paper_key: str, level: ExperienceLevel, index: int
) -> str:
    return f"E-{_sha256(f'{project_id}:{paper_key}:{level}:{index}')[:16]}"


def _summary(item: Mapping[str, Any], *, title: str, level: ExperienceLevel) -> str:
    candidates = (
        item.get("declaration"),
        item.get("keywords_summary"),
        item.get("narrative"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            compact = " ".join(value.split())
            return compact[:240]
    return f"{level.upper()} experience from {title or 'paper'}"


def _experience_text(
    *,
    experience_id: str,
    level: ExperienceLevel,
    metadata: Mapping[str, Any],
    item: Mapping[str, Any],
) -> str:
    return json.dumps(
        {
            "id": experience_id,
            "record_kind": "experience",
            "experience_level": level,
            "scope": "project",
            "project_id": metadata.get("project_id"),
            "paper": {
                "paper_id": metadata.get("paper_id"),
                "title": metadata.get("title"),
                "url": metadata.get("url"),
            },
            "experience": item,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def list_experience_documents(
    *, memory_dir: str | Path, project_id: str
) -> list[ObservationSearchDocument]:
    """Expose project experiences as documents consumable by EvoMemory search."""
    root = Path(memory_dir).expanduser()
    project_root = root / EXPERIENCE_DIR / project_id
    documents: list[ObservationSearchDocument] = []
    try:
        paper_dirs = sorted(path for path in project_root.iterdir() if path.is_dir())
    except OSError:
        return []
    for directory in paper_dirs:
        metadata = _read_json(directory / "metadata.json")
        if metadata is None or metadata.get("project_id") != project_id:
            continue
        for level in ("l1", "l2"):
            payload = _read_json(directory / f"{level}.json")
            experiences = payload.get("experiences") if payload else None
            if not isinstance(experiences, list):
                continue
            for index, item in enumerate(experiences):
                if not isinstance(item, dict):
                    continue
                experience_id = _experience_id(
                    project_id=project_id,
                    paper_key=directory.name,
                    level=level,
                    index=index,
                )
                text = _experience_text(
                    experience_id=experience_id,
                    level=level,
                    metadata=metadata,
                    item=item,
                )
                relative = (directory / f"{level}.json").relative_to(root).as_posix()
                documents.append(
                    ObservationSearchDocument(
                        observation_id=experience_id,
                        path=f"/memories/{relative}",
                        memory_type=MemoryType.SEMANTIC,
                        scope=MemoryScope.PROJECT,
                        summary=_summary(
                            item,
                            title=str(metadata.get("title") or ""),
                            level=level,
                        ),
                        body=text,
                        text=text,
                        record_kind="experience",
                        experience_level=level,
                    )
                )
    return documents


def read_experience_file(
    *, memory_dir: str | Path, project_id: str, experience_id: str
) -> ObservationReadResult | None:
    """Read one stable experience record in the active project."""
    for document in list_experience_documents(
        memory_dir=memory_dir, project_id=project_id
    ):
        if document.observation_id != experience_id.strip():
            continue
        return {
            "observation_id": document.observation_id,
            "path": document.path,
            "memory_type": document.memory_type,
            "scope": document.scope,
            "summary": document.summary,
            "text": document.text,
            "record_kind": "experience",
            "experience_level": document.experience_level or "l1",
        }
    return None
