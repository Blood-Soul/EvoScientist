"""Coordination between the main agent and the node agents of a skeleton run: one GPU, questions, notes.

In a run with `research_tree_skeleton`, the main agent writes each step's skeleton and hands the node to a
`node-agent`, which works out the details, writes the code and runs the experiment. Several node agents work at
the same time; there is one GPU. This module is the state they share, kept in the agent process:

- **The GPU lease.** A node agent asks for the GPU with a reason. A free GPU with nobody waiting goes to it at
  once (the main agent is told and can take it back); otherwise the request waits in a queue and the main agent
  decides who is next. A free GPU with requests waiting for `auto_grant_seconds` goes to the earliest request.
  Releasing or losing the lease stops the node's processes: they are the ones that could still use the card.
- **Questions.** A node agent may ask the main agent and waits up to `answer_seconds` for the answer; then it
  continues with the skeleton's choice.
- **Notes.** Every change is an event addressed to the main agent, to one node, or to every open node. Each agent
  is shown the events addressed to it once, as a message after its last one, so nothing earlier in its context
  changes and its prompt cache stays valid.

Shell commands of an agent that does not hold the GPU run with `CUDA_VISIBLE_DEVICES` empty (:func:`shell_env`).
That keeps a node from using the card by mistake; it is not a security boundary.

The node an agent works for is a context variable (:data:`CURRENT_NODE`), set by the background job that runs the
node agent and inherited by everything the agent does, its tools and shell commands included.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .gitlog import NODES_DIR, STATE_DIR
from .monitor import gpu_in_use, kill_processes_under

AUTO_GRANT_MINUTES = 10.0
ANSWER_MINUTES = 20.0
WAIT_MINUTES = 30.0  # one request_gpu call waits at most this long; the request stays queued after it
POLL_SECONDS = 0.5
# A holder with no GPU process in its node directory is probably doing CPU work. Warn once, then
# release the card (without killing its CPU processes) so the queue can move.
GPU_IDLE_WARN_MINUTES = 5.0
GPU_IDLE_RELEASE_MINUTES = 10.0
GPU_CHECK_SECONDS = 60.0

MAIN = "main"
NODES = "nodes"  # every open node

CURRENT_NODE: contextvars.ContextVar[str] = contextvars.ContextVar("evosci_tree_node", default="")

# Setting the variable, not reading it: `CUDA_VISIBLE_DEVICES=0 cmd`, `export CUDA_VISIBLE_DEVICES`, `env -u ...`,
# or `os.environ["CUDA_VISIBLE_DEVICES"] = ...` in a one-liner. A plain assignment is only refused in
# command position (start of a command segment): a heredoc or an echo that merely *writes* the line into a
# script is not setting anything, and the agents write scripts on the CPU before they hold the GPU.
_CUDA_SEGMENT = r"(?:^|[;&|(\n])\s*"
_CUDA_VAR = re.compile(
    _CUDA_SEGMENT + r"CUDA_VISIBLE_DEVICES\s*="
    r"|" + _CUDA_SEGMENT + r"(?:export|unset)\s+(?:\S+\s+)*CUDA_VISIBLE_DEVICES\b"
    r"|" + _CUDA_SEGMENT + r"env\s+(?:-\S+\s+)*(?:\S+\s+)*CUDA_VISIBLE_DEVICES\b"
    r"|CUDA_VISIBLE_DEVICES['\"]\s*\]\s*="
    r"|(?:pop|del)\W+(?:\w+\W+)*CUDA_VISIBLE_DEVICES",
    re.MULTILINE,
)
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _without_heredoc_bodies(command: str) -> str:
    """The command with heredoc bodies blanked: writing a line is not running it."""
    lines = command.splitlines(keepends=True)
    if not any(_HEREDOC.search(line) for line in lines):
        return command
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        opener = _HEREDOC.search(line)
        if opener is not None:
            tag = opener.group(2)
            i += 1
            while i < len(lines) and lines[i].strip() != tag:
                out.append("\n" if lines[i].endswith("\n") else "")
                i += 1
            if i < len(lines):
                out.append(lines[i])  # the terminator line
        i += 1
    return "".join(out)


def _clock(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t))


@dataclass
class Hub:
    root: Path
    now: Callable[[], float] = time.time
    auto_grant_seconds: float = AUTO_GRANT_MINUTES * 60
    answer_seconds: float = ANSWER_MINUTES * 60
    wait_seconds: float = WAIT_MINUTES * 60
    kill: Callable[[Path], list[int]] = kill_processes_under
    # Production sets this from the supervisor (`gpu_in_use`); tests leave it None so a fake clock
    # never launches nvidia-smi. None means "cannot tell", never "idle".
    gpu_check: Callable[[Path], bool | None] | None = None
    gpu_idle_warn_seconds: float = GPU_IDLE_WARN_MINUTES * 60
    gpu_idle_release_seconds: float = GPU_IDLE_RELEASE_MINUTES * 60
    last_gpu_active: float = 0.0
    gpu_idle_warned: bool = False
    _last_gpu_check: float = 0.0
    nodes: dict[str, dict] = field(default_factory=dict)  # open node -> {"job": id or "", "cursor": seq}
    holder: str = ""
    held_since: float = 0.0
    hold_reason: str = ""
    free_since: float = 0.0
    queue: list[dict] = field(default_factory=list)  # {"node", "reason", "t"}, oldest first
    questions: dict[str, dict] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    main_cursor: int = 0
    _n_questions: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.free_since = self.now()

    # ------------------------------------------------------------------ events

    def _emit(self, to: str, text: str) -> None:
        with self._lock:  # take_notes moves a cursor to len(events); an unlocked append could fall past it
            ev = {"seq": len(self.events) + 1, "t": self.now(), "to": to, "text": text}
            self.events.append(ev)
            try:
                path = self.root / STATE_DIR / "coord.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except OSError:
                pass  # the in-memory event is what the agents read; the file is for audit

    def _broadcast_gpu(self, skip: str = "") -> None:
        """Tell every open node where the GPU is now: the short notice that lets them plan their CPU work."""
        if self.holder:
            text = (f"The GPU is now held by {self.holder} ({self.hold_reason}). Until you hold it your commands "
                    "see no GPU: do the work that needs none, and call request_gpu(reason) when you need it.")
        else:
            waiting = ", ".join(q["node"] for q in self.queue)
            text = "The GPU is free" + (f"; waiting for it: {waiting}" if waiting else "") + "."
        for nid in self.nodes:
            if nid != skip:
                self._emit(nid, text)

    def peek(self, audience: str) -> bool:
        """True when there are events for `audience` it has not been shown."""
        with self._lock:
            self.tick()
            return bool(self._unseen(audience))

    def _unseen(self, audience: str) -> list[dict]:
        cursor = self.main_cursor if audience == MAIN else self.nodes.get(audience, {}).get("cursor", 0)
        want = {MAIN} if audience == MAIN else {audience, NODES}
        return [e for e in self.events[cursor:] if e["to"] in want]

    def take_notes(self, audience: str) -> list[str]:
        """The events for `audience` since it was last shown any, and mark them shown."""
        with self._lock:
            self.tick()
            notes = [f"[{_clock(e['t'])}] {e['text']}" for e in self._unseen(audience)]
            if audience == MAIN:
                self.main_cursor = len(self.events)
            elif audience in self.nodes:
                self.nodes[audience]["cursor"] = len(self.events)
            return notes

    # ------------------------------------------------------------------ nodes and jobs

    def open_node(self, nid: str) -> None:
        with self._lock:
            self.nodes.setdefault(nid, {"job": "", "cursor": len(self.events)})

    def is_open(self, nid: str) -> bool:
        with self._lock:
            return nid in self.nodes

    def job_of(self, nid: str) -> str:
        with self._lock:
            return self.nodes.get(nid, {}).get("job", "")

    def attach_job(self, nid: str, job: str) -> None:
        with self._lock:
            if nid in self.nodes:
                self.nodes[nid]["job"] = job

    def job_ended(self, nid: str, job: str) -> None:
        """The node's agent returned. Its waiting request and questions go; a lease it holds stays for the main
        agent to end (tree_finish or gpu_revoke), because only the main agent decides who uses the GPU."""
        with self._lock:
            node = self.nodes.get(nid)
            if node is None or node.get("job") != job:
                return
            node["job"] = ""
            self.queue = [q for q in self.queue if q["node"] != nid]
            for q in self.questions.values():
                if q["node"] == nid and q["status"] == "open":
                    q["status"] = "dropped"
            if self.holder == nid:
                self._emit(MAIN, f"{nid}'s agent finished while holding the GPU. Close the node with tree_finish "
                                 f"(which frees the GPU) or take the GPU back with gpu_revoke({nid}).")

    def close_node(self, nid: str) -> list[int]:
        """The node is closed: drop its request and questions, free the GPU if it holds it."""
        with self._lock:
            killed: list[int] = []
            if self.holder == nid:
                killed = self._free(f"{nid} was closed")
            self.queue = [q for q in self.queue if q["node"] != nid]
            for q in self.questions.values():
                if q["node"] == nid and q["status"] == "open":
                    q["status"] = "dropped"
            self.nodes.pop(nid, None)
            return killed

    # ------------------------------------------------------------------ the GPU

    def _node_dir(self, nid: str) -> Path:
        return self.root / NODES_DIR / nid

    def _give(self, nid: str, reason: str) -> None:
        self.holder, self.held_since, self.hold_reason = nid, self.now(), reason
        self.last_gpu_active = self.now()
        self.gpu_idle_warned = False
        self._last_gpu_check = 0.0
        self.queue = [q for q in self.queue if q["node"] != nid]

    def _free(self, why: str, *, kill: bool = True) -> list[int]:
        nid = self.holder
        killed = self.kill(self._node_dir(nid)) if kill and self._node_dir(nid).is_dir() else []
        self.holder, self.hold_reason, self.free_since = "", "", self.now()
        self.last_gpu_active, self.gpu_idle_warned, self._last_gpu_check = 0.0, False, 0.0
        held = (self.now() - self.held_since) / 60
        waiting = "; ".join(f"{q['node']} ({q['reason']}, waiting {(self.now() - q['t']) / 60:.0f} min)"
                            for q in self.queue)
        self._emit(MAIN, f"The GPU is free: {why} after {held:.0f} min."
                   + (f" Waiting for it: {waiting}. Give it with gpu_grant(node); without a decision within "
                      f"{self.auto_grant_seconds / 60:.0f} min it goes to the earliest request." if waiting else ""))
        self._broadcast_gpu()
        return killed

    def request(self, nid: str, reason: str) -> str:
        """`granted` when the GPU was free with nobody waiting (or is already this node's), else `queued`."""
        with self._lock:
            if self.holder == nid:
                return "granted"
            if not self.holder and not self.queue:
                self._give(nid, reason)
                self._emit(MAIN, f"{nid} took the free GPU: {reason}. Take it back with gpu_revoke if another "
                                 "node needs it more.")
                self._broadcast_gpu(skip=nid)
                return "granted"
            if not any(q["node"] == nid for q in self.queue):
                self.queue.append({"node": nid, "reason": reason, "t": self.now()})
                where = f"held by {self.holder} for {(self.now() - self.held_since) / 60:.0f} min ({self.hold_reason})" \
                    if self.holder else "free, with earlier requests waiting"
                self._emit(MAIN, f"{nid} asks for the GPU: {reason}. The GPU is {where}. Decide with "
                                 f"gpu_grant({nid}) when it is free.")
            return "queued"

    def lease(self, nid: str) -> str:
        """Where the node stands: `granted`, `queued`, `closed` (no longer open), or `none`."""
        with self._lock:
            self.tick()
            if nid not in self.nodes:
                return "closed"
            if self.holder == nid:
                return "granted"
            return "queued" if any(q["node"] == nid for q in self.queue) else "none"

    def grant(self, nid: str) -> str:
        with self._lock:
            if nid not in self.nodes:
                raise ValueError(f"{nid} is not an open node; open: {', '.join(self.nodes) or 'none'}")
            if self.holder == nid:
                return f"{nid} already holds the GPU."
            if self.holder:
                raise ValueError(f"{self.holder} holds the GPU ({self.hold_reason}); take it back with "
                                 f"gpu_revoke({self.holder}, reason) first, or wait for it to be released")
            asked = next((q for q in self.queue if q["node"] == nid), None)
            reason = asked["reason"] if asked else "given by the main agent"
            self._give(nid, reason)
            # A node whose request waited (and may have timed out of its own wait) must hear that
            # the GPU is now its own; skipping the holder in _broadcast_gpu is not enough.
            if asked:
                self._emit(nid, f"The GPU is yours ({reason}). Start GPU work now (a process started before "
                                "cannot see it); call release_gpu when it is done.")
            else:
                self._emit(nid, "The main agent gave you the GPU. Start GPU work now (a process started before "
                                "cannot see it); call release_gpu when it is done.")
            self._broadcast_gpu(skip=nid)
            return f"{nid} holds the GPU now ({reason})."

    def release(self, nid: str) -> list[int]:
        with self._lock:
            if self.holder != nid:
                raise ValueError(f"{nid} does not hold the GPU")
            return self._free(f"{nid} released it")

    def revoke(self, nid: str, reason: str) -> list[int]:
        with self._lock:
            if self.holder != nid:
                raise ValueError(f"{nid} does not hold the GPU; it is "
                                 + (f"held by {self.holder}" if self.holder else "free"))
            self._emit(nid, f"The main agent took the GPU back: {reason}. Your processes in the node directory "
                            "were stopped. Call request_gpu(reason) again when you need it.")
            return self._free(f"taken back from {nid} ({reason})")

    def _check_holder_idle(self) -> None:
        """Warn once, then free a lease that has no GPU process, without killing the holder's CPU work."""
        if not self.holder or self.gpu_check is None:
            return
        now = self.now()
        if now - self._last_gpu_check < GPU_CHECK_SECONDS:
            return
        self._last_gpu_check = now
        try:
            active = self.gpu_check(self._node_dir(self.holder))
        except Exception:
            return
        if active is None:
            return  # nvidia-smi could not answer; unknown is not idle
        if active:
            self.last_gpu_active = now
            self.gpu_idle_warned = False
            return
        idle = now - self.last_gpu_active
        if idle < self.gpu_idle_warn_seconds:
            return
        if not self.gpu_idle_warned:
            self.gpu_idle_warned = True
            self._emit(MAIN, f"{self.holder} has held the GPU for {(now - self.held_since) / 60:.0f} min "
                             f"with no GPU process for {idle / 60:.0f} min. If it is doing CPU work, take the GPU "
                             f"back with gpu_revoke({self.holder}, reason); otherwise it is released automatically "
                             f"in {(self.gpu_idle_release_seconds - idle) / 60:.0f} min.")
        if idle < self.gpu_idle_release_seconds:
            return
        node = self.holder
        self._free(f"{node} held the GPU without a GPU process", kill=False)
        if self.queue:
            first = self.queue[0]
            self._give(first["node"], first["reason"])
            self._emit(first["node"], f"The GPU is yours ({first['reason']}, released by an idle lease). Start GPU "
                                      "work now; call release_gpu when it is done.")
            self._broadcast_gpu(skip=first["node"])

    def tick(self) -> None:
        """Give a GPU that stayed free with requests waiting to the earliest request."""
        with self._lock:
            self._check_holder_idle()
            if self.holder or not self.queue:
                return
            first = self.queue[0]
            if self.now() - max(self.free_since, first["t"]) < self.auto_grant_seconds:
                return
            self._give(first["node"], first["reason"])
            self._emit(MAIN, f"No decision on the GPU within {self.auto_grant_seconds / 60:.0f} min: it went to "
                             f"the earliest request, {first['node']} ({first['reason']}).")
            # The holder is skipped by _broadcast_gpu, so tell it directly (its wait may have timed out).
            self._emit(first["node"], f"The GPU is yours ({first['reason']}, granted automatically). Start GPU "
                                      "work now; call release_gpu when it is done.")
            self._broadcast_gpu(skip=first["node"])

    # ------------------------------------------------------------------ questions

    def ask(self, nid: str, text: str) -> str:
        with self._lock:
            self._n_questions += 1
            qid = f"Q{self._n_questions}"
            self.questions[qid] = {"id": qid, "node": nid, "text": text, "t": self.now(), "answer": "",
                                   "status": "open"}
            self._emit(MAIN, f"Question {qid} from {nid}: {text}\nAnswer with answer_node('{qid}', answer). "
                             f"Without an answer within {self.answer_seconds / 60:.0f} min the node continues "
                             "with the skeleton's choice.")
            return qid

    def answer_of(self, qid: str) -> dict:
        with self._lock:
            return dict(self.questions[qid])

    def expire(self, qid: str) -> None:
        with self._lock:
            if self.questions[qid]["status"] == "open":
                self.questions[qid]["status"] = "timed_out"

    def answer(self, qid: str, text: str) -> str:
        with self._lock:
            q = self.questions.get(qid)
            if q is None:
                raise ValueError(f"no question {qid}")
            if q["status"] == "answered":
                raise ValueError(f"{qid} was answered already")
            if q["status"] == "dropped" or q["node"] not in self.nodes:
                raise ValueError(f"{qid} no longer has anyone waiting: {q['node']}'s agent has ended")
            late = q["status"] == "timed_out"
            q.update(status="answered", answer=text)
            if late:
                self._emit(q["node"], f"A late answer to your question {qid} ({q['text']}): {text}")
                return (f"{qid} had timed out, so {q['node']} went on with the skeleton's choice; the answer is "
                        "passed on as a note.")
            return f"{qid} answered; {q['node']} continues."

    # ------------------------------------------------------------------ status

    def status_lines(self) -> list[str]:
        with self._lock:
            self.tick()
            now = self.now()
            holder_line = (
                f"GPU: held by {self.holder} for {(now - self.held_since) / 60:.0f} min ({self.hold_reason})"
                if self.holder else f"GPU: free for {(now - self.free_since) / 60:.0f} min"
            )
            if self.holder and self.gpu_check is not None and self.last_gpu_active:
                idle = (now - self.last_gpu_active) / 60
                if idle >= 1:
                    holder_line += f"; no GPU process for {idle:.0f} min"
            lines = [holder_line]
            lines += [f"  waiting: {q['node']} for {(now - q['t']) / 60:.0f} min ({q['reason']})" for q in self.queue]
            lines += [f"  question {q['id']} from {q['node']}, open {(now - q['t']) / 60:.0f} min: {q['text']}"
                      for q in self.questions.values() if q["status"] == "open"]
            return lines

    def node_line(self, nid: str) -> str:
        with self._lock:
            if self.holder == nid:
                return f"holds the GPU ({self.hold_reason})"
            if any(q["node"] == nid for q in self.queue):
                return "waiting for the GPU"
            return "working without the GPU" if self.nodes.get(nid, {}).get("job") else "no agent running"


# ---------------------------------------------------------------------- the hub of this process

_hubs: dict[Path, Hub] = {}
_active: Hub | None = None
_registry_lock = threading.Lock()


def hub_for(root: str | Path, **kwargs) -> Hub:
    """The hub of the research tree at `root`, created on first use. It becomes the process's active hub, which
    is what the shell consults (:func:`shell_env`)."""
    global _active
    key = Path(root).resolve()
    with _registry_lock:
        hub = _hubs.get(key)
        if hub is None:
            hub = _hubs[key] = Hub(key, **kwargs)
        _active = hub
        return hub


def active_hub() -> Hub | None:
    return _active


def deactivate() -> None:
    """Forget every hub (tests, or a process that leaves skeleton mode)."""
    global _active
    with _registry_lock:
        _hubs.clear()
        _active = None


def _holds_gpu() -> bool:
    node = CURRENT_NODE.get()
    return bool(_active is not None and node and _active.holder == node)


def shell_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """The environment for one shell command: unchanged outside a skeleton run or for the GPU's holder; for
    everyone else the same with `CUDA_VISIBLE_DEVICES` empty."""
    if _active is None or _holds_gpu():
        return env
    out = dict(os.environ if env is None else env)
    out["CUDA_VISIBLE_DEVICES"] = ""
    return out


def refuse_command(command: str) -> str | None:
    """An error for a command that would set `CUDA_VISIBLE_DEVICES` without holding the GPU, else None."""
    if _active is None or _holds_gpu():
        return None
    if not _CUDA_VAR.search(_without_heredoc_bodies(command or "")):
        return None
    return ("Refused: this command sets CUDA_VISIBLE_DEVICES. The GPU is leased: call request_gpu(reason), and run "
            "GPU work once it is granted.")
