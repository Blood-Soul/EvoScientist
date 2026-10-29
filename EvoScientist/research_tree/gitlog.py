"""The research trajectory, kept as git history.

One node = one research decision = one *event commit* on branch `node/<id>`,
checked out as a worktree at `nodes/<id>/` while it is being built. The agent
may commit as often as it likes inside the node; when the node ends, all of
that is squashed into one commit whose parents are the parent nodes' event
commits (two or more for a merge), so the commit DAG is the research tree and
each commit is a decision, not a line edit. The agent's own commits are kept
under `refs/raw/<id>` for audit and stay out of the trajectory.

Whether a decision was kept is often only known later (a child beats it, or it
leads nowhere), so the outcome lives in a git note (`refs/notes/outcome`) on
the event commit and is rewritten as the tree grows: the event itself never
changes, its verdict can.

The message has a narrative part (written by the agent when it closes the node, component-level)
and a final paragraph of trailers that only the research-tree tools write, so
facts such as the score cannot come from the agent.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GIT_IDENT = [
    "-c",
    "user.name=research-loop",
    "-c",
    "user.email=research-loop@localhost",
]
ROOT_BRANCH = "research/root"
NODE_PREFIX = "node/"
NODES_DIR = "nodes"
NOTES_DIR = "research_notes"
OUTCOME_REF = "outcome"
FINAL_TAG = "research/final"
# Patterns anchored with "/" apply at the root of every worktree. Data, weights
# and predictions never enter an event commit: they would make branches heavy
# and every diff unreadable. Predictions and the node report are copied into
# `.research/artifacts/<id>/` instead. The task's own file names (prediction
# file, deliverable) are added by `Repo.init(extra_ignore=...)`.
STATE_DIR = ".research"
IGNORE_PATTERNS = [
    f"/{NODES_DIR}/",
    f"/{NOTES_DIR}/",
    f"/{STATE_DIR}/",
    "/data/",
    "/models/",
    "/NODE_REPORT.md",
    "/TRAJECTORY.md",
    "outputs/",
    "__pycache__/",
    "*.pt",
    "*.pth",
    "*.bin",
    "*.safetensors",
    "*.ckpt",
    "checkpoint-*/",
    "*.log",
    "*.arrow",
    "wandb/",
    "mlruns/",
]
NARRATIVE_KEYS = ("Hypothesis", "Source", "Change", "Observation", "Next")
_TRAILER_RE = re.compile(r"^([A-Z][A-Za-z-]*): (.*)$")


class GitError(RuntimeError):
    pass


@dataclass
class Event:
    sha: str
    node: str
    subject: str
    narrative: dict[str, str]
    trailers: dict[str, str]
    parents: list[str] = field(
        default_factory=list
    )  # parent node ids (empty: from the root)
    outcome: str = ""

    @property
    def score(self) -> float | None:
        try:
            return float(self.trailers.get("Score", ""))
        except ValueError:
            return None

    @property
    def status(self) -> str:
        return self.trailers.get("Status", "")


def node_number(nid: str) -> int:
    return int(nid[1:]) if nid[1:].isdigit() else 0


def flatten(text: str | None, limit: int = 1200) -> str:
    """One line, so a narrative field can never be mistaken for a trailer block."""
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def compose_message(
    category: str, subject: str, narrative: dict[str, str], trailers: dict[str, str]
) -> str:
    lines = [f"[{category}] {flatten(subject, 120)}", ""]
    lines += [
        f"{k}: {flatten(narrative.get(k))}" for k in NARRATIVE_KEYS if narrative.get(k)
    ]
    lines += [""] + [
        f"{k}: {flatten(v, 400)}" for k, v in trailers.items() if v not in (None, "")
    ]
    return "\n".join(lines) + "\n"


def parse_message(message: str) -> tuple[str, dict[str, str], dict[str, str]]:
    paras = [p.strip("\n") for p in message.strip().split("\n\n")]
    subject = paras[0].splitlines()[0] if paras and paras[0] else ""
    trailers: dict[str, str] = {}
    if len(paras) > 1:
        for line in paras[-1].splitlines():
            m = _TRAILER_RE.match(line)
            if m:
                trailers[m.group(1)] = m.group(2)
    narrative: dict[str, str] = {}
    for para in paras[1:-1]:
        for line in para.splitlines():
            m = _TRAILER_RE.match(line)
            if m and m.group(1) in NARRATIVE_KEYS:
                narrative[m.group(1)] = m.group(2)
    return subject, narrative, trailers


def link_tree(src: Path, dst: Path) -> None:
    """Hardlink `src` into `dst` file by file (copy across filesystems).

    Not a symlink: a link that resolves outside the agent's workdir is
    refused by sandboxed file tools, a hardlink resolves inside it.
    """
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    for f in src.rglob("*"):
        target = dst / f.relative_to(src)
        if f.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif f.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(f, target)
            except OSError:
                shutil.copy2(f, target)


class Repo:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def git(
        self,
        *args: str,
        cwd: Path | None = None,
        check: bool = True,
        input: str | None = None,
    ) -> str:
        p = subprocess.run(
            ["git", *GIT_IDENT, *args],
            cwd=cwd or self.root,
            input=input,
            capture_output=True,
            text=True,
        )
        if check and p.returncode:
            raise GitError(f"git {' '.join(args)}: {(p.stderr or p.stdout).strip()}")
        return p.stdout.strip()

    def node_path(self, nid: str) -> Path:
        return self.root / NODES_DIR / nid

    # ------------------------------------------------------------------ setup

    def init(self, extra_ignore: list[str] | None = None) -> str:
        if not (self.root / ".git").exists():
            self.git("init", "--quiet")
        common = Path(self.git("rev-parse", "--git-common-dir"))
        exclude = (
            (common if common.is_absolute() else self.root / common)
            / "info"
            / "exclude"
        )
        exclude.parent.mkdir(parents=True, exist_ok=True)
        have = set(exclude.read_text().splitlines()) if exclude.exists() else set()
        add = [p for p in [*IGNORE_PATTERNS, *(extra_ignore or [])] if p not in have]
        if add:
            with open(exclude, "a") as fh:
                fh.write("\n".join(add) + "\n")
        if self.git("rev-parse", "--verify", "--quiet", "HEAD", check=False) == "":
            self.git("add", "-A")
            self.git("commit", "--quiet", "--allow-empty", "-m", "research root")
        self.git("branch", "-f", ROOT_BRANCH, "HEAD")
        (self.root / NODES_DIR).mkdir(exist_ok=True)
        (self.root / NOTES_DIR).mkdir(exist_ok=True)
        return self.git("rev-parse", ROOT_BRANCH)

    # ------------------------------------------------------------------ nodes

    def event_sha(self, nid: str) -> str | None:
        out = self.git(
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/heads/{NODE_PREFIX}{nid}",
            check=False,
        )
        return out or None

    def new_node(
        self, nid: str, parents: list[str], shared: list[str]
    ) -> tuple[Path, list[str]]:
        """Create `nodes/<nid>` on `node/<nid>`, starting from the first parent.

        Further parents are merged in; conflicting files are left with their
        markers for the node's agent to resolve. Returns the path and the
        conflicting files.
        """
        base = f"{NODE_PREFIX}{parents[0]}" if parents else ROOT_BRANCH
        path = self.node_path(nid)
        self.git(
            "worktree", "add", "--quiet", "-b", f"{NODE_PREFIX}{nid}", str(path), base
        )
        conflicts: list[str] = []
        for p in parents[1:]:
            merged = subprocess.run(
                [
                    "git",
                    *GIT_IDENT,
                    "merge",
                    "--no-ff",
                    "--no-edit",
                    f"{NODE_PREFIX}{p}",
                ],
                cwd=path,
                capture_output=True,
                text=True,
            )
            if merged.returncode:
                conflicts += self.git(
                    "diff", "--name-only", "--diff-filter=U", cwd=path
                ).split()
                break
        for name in shared:
            src = self.root / name
            if src.exists() and not (path / name).exists():
                link_tree(src, path / name)
        # Trained weights are not versioned, but a refinement often needs its
        # parent's (a new decision rule on the same model). Hardlinked, so it
        # costs no disk; the node is told to write new files, not overwrite.
        if parents and (self.node_path(parents[0]) / "outputs").is_dir():
            link_tree(self.node_path(parents[0]) / "outputs", path / "outputs")
        return path, sorted(set(conflicts))

    def parent_shas(self, parents: list[str]) -> list[str]:
        return [self.git("rev-parse", f"{NODE_PREFIX}{p}") for p in parents] or [
            self.git("rev-parse", ROOT_BRANCH)
        ]

    def stage(
        self, nid: str, parents: list[str], excerpt_chars: int = 6000
    ) -> tuple[list[str], str, str]:
        """Stage everything the node changed; return (unresolved files, diffstat, diff excerpt) vs its first parent."""
        path = self.node_path(nid)
        unresolved = self.git(
            "diff", "--name-only", "--diff-filter=U", cwd=path
        ).split()
        self.git("add", "-A", cwd=path)
        base = self.parent_shas(parents)[0]
        stat = self.git("diff", "--cached", "--shortstat", base, cwd=path)
        text = (
            self.git("diff", "--cached", "--stat", base, cwd=path)
            + "\n\n"
            + self.git("diff", "--cached", "-U2", base, cwd=path, check=False)
        )
        if len(text) > excerpt_chars:
            text = text[:excerpt_chars] + "\n… (diff truncated)"
        return unresolved, stat, text

    def commit(self, nid: str, parents: list[str], message: str) -> str:
        """Squash the staged node into one event commit on `node/<nid>`; the agent's commits go to refs/raw/<nid>."""
        path = self.node_path(nid)
        tree = self.git("write-tree", cwd=path)
        self.git(
            "update-ref", f"refs/raw/{nid}", self.git("rev-parse", "HEAD", cwd=path)
        )
        args = ["commit-tree", tree]
        for s in self.parent_shas(parents):
            args += ["-p", s]
        sha = self.git(*args, "-F", "-", input=message)
        # An unfinished merge would otherwise survive the reset as MERGE_HEAD.
        self.git("merge", "--quit", cwd=path, check=False)
        self.git("reset", "--soft", sha, cwd=path)
        return sha

    # --------------------------------------------------------------- outcomes

    def set_outcome(self, sha: str, text: str) -> None:
        self.git("notes", f"--ref={OUTCOME_REF}", "add", "-f", "-m", text, sha)

    def outcome(self, sha: str) -> str:
        return self.git("notes", f"--ref={OUTCOME_REF}", "show", sha, check=False)

    def tag_final(self, sha: str, message: str) -> None:
        self.git("tag", "-a", "-f", FINAL_TAG, sha, "-m", message)

    # ------------------------------------------------------------------ reads

    def events(self) -> list[Event]:
        """Every finished node, in the order the decisions were made."""
        refs = self.git(
            "for-each-ref",
            "--format=%(refname:short) %(objectname)",
            f"refs/heads/{NODE_PREFIX}",
        )
        out: list[Event] = []
        for line in refs.splitlines():
            ref, sha = line.split()
            nid = ref[len(NODE_PREFIX) :]
            message = self.git("log", "-1", "--format=%B", sha)
            subject, narrative, trailers = parse_message(message)
            if trailers.get("Event-Id") != nid:
                continue  # a node still being built: its branch tip is not an event yet
            parents = [
                p
                for p in trailers.get("Parents", "").split(",")
                if p.strip() and p.strip() != "root"
            ]
            out.append(
                Event(
                    sha=sha,
                    node=nid,
                    subject=subject,
                    narrative=narrative,
                    trailers=trailers,
                    parents=[p.strip() for p in parents],
                    outcome=self.outcome(sha),
                )
            )
        return sorted(out, key=lambda e: node_number(e.node))


def ancestry(events: dict[str, Event], nid: str | None) -> set[str]:
    seen: set[str] = set()
    stack = [nid] if nid else []
    while stack:
        cur = stack.pop()
        if cur in seen or cur not in events:
            continue
        seen.add(cur)
        stack += events[cur].parents
    return seen


def render_event(e: Event, *, compact: bool = False) -> str:
    t = e.trailers
    head = f"### {e.node} {e.subject}"
    facts = [
        f"kind {t.get('Kind', '?')}",
        f"from {t.get('Parents', 'root')}",
        f"status {t.get('Status', '?')}",
    ]
    if t.get("Score"):
        facts.append(f"holdout {t['Score']}")
    if t.get("Delta"):
        facts.append(f"Δ {t['Delta']}")
    if t.get("Minutes"):
        facts.append(f"{t['Minutes']} min")
    if t.get("Cluster"):
        facts.append(f"family {t['Cluster']}")
    if t.get("Based-On"):
        facts.append(f"based on {t['Based-On']}")
    lines = [head, "- " + "; ".join(facts), f"- outcome: {e.outcome or 'pending'}"]
    keys = (
        ("Hypothesis", "Change", "Observation", "Next") if compact else NARRATIVE_KEYS
    )
    for k in keys:
        if e.narrative.get(k):
            lines.append(f"- {k.lower()}: {e.narrative[k]}")
    if compact:
        for k in ("Slices", "Error"):
            if t.get(k):
                lines.append(f"- {k.lower()}: {t[k]}")
    else:
        for k in ("Slices", "Diffstat", "Error", "Unresolved-Conflicts"):
            if t.get(k):
                lines.append(f"- {k.lower()}: {t[k]}")
        lines.append(f"- commit: {e.sha[:12]}")
    return "\n".join(lines)


def render_trajectory(
    events: list[Event],
    *,
    champion: str | None = None,
    insights: list[dict] | None = None,
    compact: bool = False,
    title: str = "Research trajectory",
) -> str:
    parts = [f"# {title}", ""]
    if champion:
        parts += [f"Final: {champion}", ""]
    if not events:
        parts.append("(no finished steps yet)")
    for e in events:
        parts += [render_event(e, compact=compact), ""]
    if insights:
        parts += ["## Lessons", ""]
        for i in insights:
            parts.append(
                f"- {i['text']} (evidence: {', '.join(i.get('evidence', []))})"
            )
    return "\n".join(parts).rstrip() + "\n"
