"""Background launch adapter for the paper experience queue."""

from __future__ import annotations

from pathlib import Path

from ..gateway.background_runs import (
    BackgroundRun,
    BackgroundRunPayload,
    BackgroundRunRequest,
    launch_background_run,
)

PAPER_EXPERIENCE_WORKER_GRAPH_ID = "evomemory-paper-experience-worker"


def paper_experience_worker_launch_request(
    project_id: str, workspace_dir: str | Path | None = None
) -> BackgroundRunRequest:
    """Build one drain request for a project's persisted queue.

    The server refuses a new run that names no registered workspace
    (``NoProjectError``), so the run goes to the workspace assistant of
    *workspace_dir*, like the other EvoMemory workers.
    """
    from ..paths import Workspace
    from .launch import _runs_create_kwargs

    workspace_key = Workspace(workspace_dir).key if workspace_dir else None
    metadata = {
        "run_kind": "evomemory_paper_experience_worker",
        "project_id": project_id,
        **({"workspace_dir": workspace_key} if workspace_key else {}),
    }

    def payload(thread_id: str) -> BackgroundRunPayload:
        body: BackgroundRunPayload = {
            "assistant_id": PAPER_EXPERIENCE_WORKER_GRAPH_ID,
            "input": {},
            "metadata": metadata,
            "config": {
                "configurable": {
                    "thread_id": thread_id,
                    "evomemory_project_id": project_id,
                }
            },
        }
        if workspace_key:
            body = _runs_create_kwargs(body, workspace_dir=workspace_key)
        return body

    return BackgroundRunRequest(
        graph_id=PAPER_EXPERIENCE_WORKER_GRAPH_ID,
        run_payload=payload,
        thread_metadata=metadata,
        name="EvoMemory paper experience worker",
    )


def launch_paper_experience_worker(
    project_id: str, workspace_dir: str | Path | None = None
) -> BackgroundRun | None:
    """Trigger a background queue drain when LangGraph dev is available."""
    return launch_background_run(
        paper_experience_worker_launch_request(project_id, workspace_dir)
    )
