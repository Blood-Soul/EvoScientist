"""Background launch adapter for the paper experience queue."""

from __future__ import annotations

from ..gateway.background_runs import (
    BackgroundRun,
    BackgroundRunPayload,
    BackgroundRunRequest,
    launch_background_run,
)

PAPER_EXPERIENCE_WORKER_GRAPH_ID = "evomemory-paper-experience-worker"


def paper_experience_worker_launch_request(project_id: str) -> BackgroundRunRequest:
    """Build one drain request for a project's persisted queue."""
    metadata = {
        "run_kind": "evomemory_paper_experience_worker",
        "project_id": project_id,
    }

    def payload(thread_id: str) -> BackgroundRunPayload:
        return {
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

    return BackgroundRunRequest(
        graph_id=PAPER_EXPERIENCE_WORKER_GRAPH_ID,
        run_payload=payload,
        thread_metadata=metadata,
        name="EvoMemory paper experience worker",
    )


def launch_paper_experience_worker(project_id: str) -> BackgroundRun | None:
    """Trigger a background queue drain when LangGraph dev is available."""
    return launch_background_run(paper_experience_worker_launch_request(project_id))
