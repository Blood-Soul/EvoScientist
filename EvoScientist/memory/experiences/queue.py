"""Durable file queue for paper experience extraction."""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .store import canonical_paper_identifier

QUEUE_DIR = "paper_experience_queue/projects"
QueueStatus = Literal["pending", "running", "completed", "failed"]


class PaperExperienceTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    project_id: str
    paper_id: str = ""
    url: str = Field(min_length=1)
    title: str = ""
    enqueued_at: str
    status: QueueStatus = "pending"
    attempts: int = 0
    updated_at: str | None = None
    error: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _task_id(project_id: str, paper_id: str, url: str) -> str:
    canonical = canonical_paper_identifier(paper_id or url)
    digest = hashlib.sha256(f"{project_id}:{canonical}".encode()).hexdigest()[:20]
    return f"PX-{digest}"


def _status_dir(memory_dir: str | Path, project_id: str, status: QueueStatus) -> Path:
    return Path(memory_dir).expanduser() / QUEUE_DIR / project_id / status


def _write_task(path: Path, task: PaperExperienceTask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        task.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_task(path: Path) -> PaperExperienceTask | None:
    try:
        return PaperExperienceTask.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def enqueue_paper(
    *,
    memory_dir: str | Path,
    project_id: str,
    url: str,
    paper_id: str = "",
    title: str = "",
) -> tuple[PaperExperienceTask, bool]:
    """Enqueue once across every queue state."""
    task_id = _task_id(project_id, paper_id, url)
    for status in ("pending", "running", "completed"):
        existing = _read_task(
            _status_dir(memory_dir, project_id, status) / f"{task_id}.json"
        )
        if existing is not None:
            return existing, False
    failed_path = _status_dir(memory_dir, project_id, "failed") / f"{task_id}.json"
    failed = _read_task(failed_path)
    if failed is not None:
        failed.status = "pending"
        failed.url = url.strip()
        failed.paper_id = paper_id.strip()
        failed.title = title.strip()
        failed.updated_at = _now()
        failed.error = None
        pending_path = _status_dir(memory_dir, project_id, "pending") / failed_path.name
        _write_task(failed_path, failed)
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(failed_path, pending_path)
        return failed, True
    task = PaperExperienceTask(
        task_id=task_id,
        project_id=project_id,
        paper_id=paper_id.strip(),
        url=url.strip(),
        title=title.strip(),
        enqueued_at=_now(),
    )
    _write_task(
        _status_dir(memory_dir, project_id, "pending") / f"{task_id}.json", task
    )
    return task, True


def list_tasks(
    *, memory_dir: str | Path, project_id: str, status: QueueStatus
) -> list[PaperExperienceTask]:
    directory = _status_dir(memory_dir, project_id, status)
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return []
    return [task for path in paths if (task := _read_task(path)) is not None]


def requeue_running_tasks(*, memory_dir: str | Path, project_id: str) -> int:
    """Recover tasks left running after an interrupted worker."""
    recovered = 0
    running_dir = _status_dir(memory_dir, project_id, "running")
    try:
        paths = sorted(running_dir.glob("*.json"))
    except OSError:
        return 0
    for source in paths:
        task = _read_task(source)
        if task is None:
            continue
        task.status = "pending"
        task.updated_at = _now()
        target = _status_dir(memory_dir, project_id, "pending") / source.name
        _write_task(source, task)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        recovered += 1
    return recovered


def queue_worker_lock_path(memory_dir: str | Path, project_id: str) -> Path:
    """Return the per-project consumer lock path."""
    return _status_dir(memory_dir, project_id, "pending").parent / ".worker.lock"


def claim_next_task(
    *, memory_dir: str | Path, project_id: str
) -> PaperExperienceTask | None:
    """Atomically move one pending task to running."""
    pending_dir = _status_dir(memory_dir, project_id, "pending")
    try:
        candidates = sorted(pending_dir.glob("*.json"))
    except OSError:
        return None
    for source in candidates:
        target = _status_dir(memory_dir, project_id, "running") / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(source, target)
        except OSError:
            continue
        task = _read_task(target)
        if task is None:
            continue
        task.status = "running"
        task.attempts += 1
        task.updated_at = _now()
        _write_task(target, task)
        return task
    return None


def _finish_task(
    *,
    memory_dir: str | Path,
    task: PaperExperienceTask,
    status: Literal["completed", "failed"],
    error: str | None,
) -> PaperExperienceTask:
    source = (
        _status_dir(memory_dir, task.project_id, "running") / f"{task.task_id}.json"
    )
    target = _status_dir(memory_dir, task.project_id, status) / source.name
    task.status = status
    task.updated_at = _now()
    task.error = error
    _write_task(source, task)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return task


def complete_task(
    *, memory_dir: str | Path, task: PaperExperienceTask
) -> PaperExperienceTask:
    return _finish_task(
        memory_dir=memory_dir, task=task, status="completed", error=None
    )


def fail_task(
    *, memory_dir: str | Path, task: PaperExperienceTask, error: str
) -> PaperExperienceTask:
    return _finish_task(memory_dir=memory_dir, task=task, status="failed", error=error)
