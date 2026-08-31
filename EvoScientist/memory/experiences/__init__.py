"""Project-scoped paper experience persistence and extraction."""

from .extraction import (
    ExperienceOutputError,
    download_paper_text,
    load_experience_prompts,
    parse_experience_json,
    run_experience_extraction,
)
from .queue import (
    PaperExperienceTask,
    claim_next_task,
    complete_task,
    enqueue_paper,
    fail_task,
    list_tasks,
    queue_worker_lock_path,
    requeue_running_tasks,
)
from .retrieval import (
    browse_experience_facets,
    experience_library_stats,
    read_memory_file,
    search_experience_records,
)
from .store import (
    EXPERIENCE_CATALOG_FILENAME,
    experience_catalog_path,
    list_experience_documents,
    load_paper_experiences,
    read_experience_file,
    refresh_all_experience_catalogs,
    refresh_experience_catalog,
    store_paper_experiences,
)
from .taxonomy import (
    DISCIPLINES,
    FALLBACK_DISCIPLINE,
    normalize_discipline,
    resolve_discipline,
)

__all__ = [
    "DISCIPLINES",
    "EXPERIENCE_CATALOG_FILENAME",
    "FALLBACK_DISCIPLINE",
    "ExperienceOutputError",
    "PaperExperienceTask",
    "browse_experience_facets",
    "claim_next_task",
    "complete_task",
    "download_paper_text",
    "enqueue_paper",
    "experience_catalog_path",
    "experience_library_stats",
    "fail_task",
    "list_experience_documents",
    "list_tasks",
    "load_experience_prompts",
    "load_paper_experiences",
    "normalize_discipline",
    "parse_experience_json",
    "queue_worker_lock_path",
    "read_experience_file",
    "read_memory_file",
    "refresh_all_experience_catalogs",
    "refresh_experience_catalog",
    "requeue_running_tasks",
    "resolve_discipline",
    "run_experience_extraction",
    "search_experience_records",
    "store_paper_experiences",
]
