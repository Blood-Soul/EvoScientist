"""Project identity helpers for file-backed memory."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _short_hash(text: str, *, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _main_checkout_root(cwd: Path) -> str | None:
    """Root of the main checkout that owns ``cwd``, seen through linked worktrees.

    ``--show-toplevel`` gives a linked worktree its own root, so a project that
    keeps one worktree per branch would get one memory partition per branch.
    ``--git-common-dir`` is shared by the main checkout and all its worktrees;
    its parent is the main checkout. Falls back to ``--show-toplevel`` for
    layouts where the common dir is not a plain ``<root>/.git`` (submodules,
    bare repos, ``GIT_DIR``), which keeps those ids unchanged.
    """
    common = _run_git(["rev-parse", "--git-common-dir"], cwd)
    if common:
        common_path = Path(common)
        if not common_path.is_absolute():
            common_path = cwd / common_path
        common_path = common_path.resolve()
        if common_path.name == ".git":
            return str(common_path.parent)
    return _run_git(["rev-parse", "--show-toplevel"], cwd)


def resolve_project_id(workspace: str | Path) -> str:
    """Return the stable id used for this workspace's project memory."""
    root = Path(workspace).expanduser().resolve()
    git_root = _main_checkout_root(root)
    if git_root:
        git_root_path = Path(git_root).expanduser().resolve()
        remote = _run_git(["remote", "get-url", "origin"], git_root_path)
        source = f"git-remote:{remote}" if remote else f"git-root:{git_root_path}"
        return f"P-{_short_hash(source)}"
    return f"P-{_short_hash(f'path:{root}')}"
