"""resolve_project_id: linked git worktrees share their main checkout's id."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from EvoScientist.memory.project import resolve_project_id

_GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]


def _git(cwd: Path, *args: str) -> None:
    subprocess.run([*_GIT, *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "commit", "-q", "--allow-empty", "-m", "root")
    return root.resolve()


def _expected(source: str) -> str:
    return f"P-{hashlib.sha256(source.encode()).hexdigest()[:16]}"


def test_plain_repo_id_is_unchanged(repo: Path) -> None:
    # Same derivation as before the worktree change: git-root:<toplevel path>.
    assert resolve_project_id(repo) == _expected(f"git-root:{repo}")


def test_subdirectory_resolves_to_repo_id(repo: Path) -> None:
    (repo / "a" / "b").mkdir(parents=True)
    assert resolve_project_id(repo / "a" / "b") == resolve_project_id(repo)


def test_linked_worktree_shares_main_checkout_id(repo: Path) -> None:
    _git(repo, "worktree", "add", "-q", "nodes/n1", "-b", "node/n1")
    node = repo / "nodes" / "n1"
    (node / "sub").mkdir()
    main_id = resolve_project_id(repo)
    assert resolve_project_id(node) == main_id
    assert resolve_project_id(node / "sub") == main_id


def test_remote_still_wins_over_path(repo: Path) -> None:
    _git(repo, "remote", "add", "origin", "https://example.org/x.git")
    _git(repo, "worktree", "add", "-q", "nodes/n1", "-b", "node/n1")
    expected = _expected("git-remote:https://example.org/x.git")
    assert resolve_project_id(repo) == expected
    assert resolve_project_id(repo / "nodes" / "n1") == expected


def test_non_git_directory_falls_back_to_path(tmp_path: Path) -> None:
    plain = (tmp_path / "plain").resolve()
    plain.mkdir()
    assert resolve_project_id(plain) == _expected(f"path:{plain}")
