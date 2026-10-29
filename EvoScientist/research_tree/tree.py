"""The research tree: facts written by tools, decisions made by the agent.

A run is a tree of *nodes*. A node is one hypothesis: it has its own git
branch and worktree under `nodes/<id>/`, is built by a sub-agent, and ends as
one *event commit* (see :mod:`.gitlog`). The commit DAG is the research tree
and `git log` is the trajectory.

Who writes what:

- The agent chooses what to try, which history to read, and writes the
  narrative of each event (hypothesis, change, observation, next step).
- This module writes every fact: node ids and parents, the clock, scores,
  deltas against the parent, the champion, the hash of the judge. Scores come
  from the frozen judge (:mod:`.judge`), never from the agent's own report.

Pilot nodes (`kind="pilot"`) are small, time-boxed experiments whose only job
is to produce evidence for a decision the history cannot settle yet. They are
not scored and never become the champion, but they are events like any other
and can be cited by later nodes.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from .gitlog import (
    STATE_DIR,
    Event,
    Repo,
    ancestry,
    compose_message,
    flatten,
    node_number,
    render_event,
    render_trajectory,
)
from .judge import Judge, judge_from_env
from .monitor import (
    kill_processes_under,
    log_event,
    snapshot_node,
    start_gpu_log,
    start_ticker,
    stop_gpu_log,
    stop_ticker,
)
from .stats import Delta, complementarity, paired_delta, slice_table

try:  # POSIX only; on other platforms calls are simply not serialised
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

DEADLINE_ENV = "EVOSCIENTIST_TREE_DEADLINE_EPOCH"
BUDGET_ENV = "EVOSCIENTIST_TREE_BUDGET_MINUTES"
KINDS = ("explore", "refine", "merge", "pilot")
CATEGORIES = ("method", "experiment", "ablation")
INSIGHT_STATUS = ("open", "supported", "contradicted")
PILOT_MINUTES = (3.0, 20.0)
NODE_MAX_MINUTES = 180.0
INSIGHT_EVERY = 3  # closed nodes after which the tools ask for an insight
OVERRUN_GRACE_MINUTES = (
    2.0  # past its budget a node keeps this long before its processes are stopped
)
WATCH_SECONDS = 30.0
RELATED_NODES = 2  # other nodes (besides the parents) put into a sub-agent's brief


class TreeError(RuntimeError):
    """A request the tree refuses; the message says what to do instead."""


def _minutes(seconds: float) -> float:
    return round(seconds / 60, 1)


class ResearchTree:
    def __init__(
        self,
        root: str | Path,
        judge: Judge | None = None,
        now: Callable[[], float] = time.time,
    ):
        self.root = Path(root).resolve()
        self.repo = Repo(self.root)
        self._judge = judge
        self.now = now
        self.dir = self.root / STATE_DIR
        self.state_path = self.dir / "state.json"

    @property
    def judge(self) -> Judge | None:
        return self._judge if self._judge is not None else judge_from_env()

    # ------------------------------------------------------------- state io

    @contextlib.contextmanager
    def _edit(self, *, create: bool = False) -> Iterator[dict]:
        """Lock, load the state, yield it, save it if the block succeeds."""
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "lock", "w") as lock:
            if fcntl:
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if self.state_path.exists():
                    if create:
                        raise TreeError(
                            "the research tree is already initialised; call tree_status"
                        )
                    state = json.loads(self.state_path.read_text())
                elif create:
                    state = {}
                else:
                    raise TreeError(
                        "there is no research tree yet; call tree_init first"
                    )
                yield state
                tmp = self.state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=1))
                tmp.replace(self.state_path)
            finally:
                if fcntl:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _scores_path(self, nid: str) -> Path:
        return self.dir / "scores" / f"{nid}.json"

    def _scores(self, nid: str) -> list[float] | None:
        p = self._scores_path(nid)
        return json.loads(p.read_text())["per_sample"] if p.is_file() else None

    def _remaining(self, st: dict) -> float:
        return _minutes(st["deadline"] - self.now())

    def _events(self) -> dict[str, Event]:
        return {e.node: e for e in self.repo.events()}

    def _log(self, st: dict, event: str, **data) -> None:
        log_event(self.dir / "events.jsonl", st["started"], self.now(), event, **data)

    def resume_monitor(self) -> None:
        """Restart the GPU log after the agent process was restarted mid-run."""
        try:
            st = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        if not st.get("finalized"):
            self._start_monitors()

    def _start_monitors(self) -> None:
        start_gpu_log(self.dir / "gpu.jsonl")
        start_ticker(str(self.dir), self._tick, WATCH_SECONDS)

    def _tick(self) -> bool:
        if not self.state_path.exists():
            return False  # the project is gone: end the ticker
        try:
            self.enforce_budgets()
        except (TreeError, OSError, ValueError):
            pass
        return True

    def enforce_budgets(self) -> list[str]:
        """Stop the processes of nodes that ran past their budget plus a short grace.

        A node's budget is a hard limit. The agent cannot be pre-empted, but
        what it started in the node's directory (training, servers) can: with
        those gone the sub-agent sees its commands fail and wraps up. The node
        stays open until the agent closes it with tree_finish.
        """
        if not self.state_path.exists():
            return []
        st = json.loads(self.state_path.read_text())
        if st.get("finalized"):
            return []
        late = [
            nid
            for nid, n in st["nodes"].items()
            if n["status"] == "running"
            and not n.get("timed_out")
            and _minutes(self.now() - n["started"])
            > n["budget_minutes"] + OVERRUN_GRACE_MINUTES
        ]
        killed = {
            nid: kill_processes_under(self.repo.node_path(nid))
            if self.repo.node_path(nid).is_dir()
            else []
            for nid in late
        }
        if late:
            with self._edit() as st:
                for nid in late:
                    node = st["nodes"][nid]
                    node["timed_out"] = True
                    self._log(
                        st,
                        "node-timeout",
                        node=nid,
                        used=_minutes(self.now() - node["started"]),
                        budget=node["budget_minutes"],
                        killed=len(killed[nid]),
                    )
        return late

    @staticmethod
    def _closed(st: dict) -> int:
        """Nodes that ended as a candidate (pilots are evidence, not candidates)."""
        return sum(
            1
            for n in st["nodes"].values()
            if n["kind"] != "pilot" and n["status"] != "running"
        )

    def _insight_nudge(self, st: dict) -> str:
        since = self._closed(st) - st.get("insight_mark", 0)
        if since < INSIGHT_EVERY:
            return ""
        return (
            f"{since} nodes have closed since the last insight. Record what the sequence shows with "
            "tree_insight: add a lesson, or confirm or contradict an existing one."
        )

    # ----------------------------------------------------------------- init

    def init(
        self,
        *,
        budget_minutes: float | None = None,
        metric: str = "score",
        direction: str = "max",
        prediction_file: str = "holdout_predictions.csv",
        deliverable: str = "submission.csv",
        min_node_minutes: float = 20.0,
        reserve_minutes: float = 15.0,
        shared: tuple[str, ...] = ("data", "models"),
    ) -> str:
        if direction not in ("max", "min"):
            raise TreeError("direction must be 'max' or 'min'")
        with self._edit(create=True) as st:
            if os.environ.get(DEADLINE_ENV):
                deadline = float(os.environ[DEADLINE_ENV])
            else:
                budget = budget_minutes or float(os.environ.get(BUDGET_ENV) or 0)
                if budget <= 0:
                    raise TreeError(
                        "budget_minutes is required: the total minutes left for the whole task"
                    )
                deadline = self.now() + budget * 60
            self.repo.init(extra_ignore=[f"/{prediction_file}", f"/{deliverable}"])
            st.update(
                version=1,
                started=self.now(),
                deadline=deadline,
                metric=metric,
                direction=direction,
                prediction_file=prediction_file,
                deliverable=deliverable,
                min_node_minutes=float(min_node_minutes),
                reserve_minutes=float(reserve_minutes),
                shared=list(shared),
                next=1,
                champion=None,
                eval_sha="",
                nodes={},
                insights=[],
                survey=[],
                insight_mark=0,
                finalized=False,
            )
            self._log(st, "init", deadline=deadline, shared=list(shared))
            self._start_monitors()
            return "Research tree initialised.\n\n" + self._status(st)

    # --------------------------------------------------------------- survey

    def survey(self, clusters: list[dict]) -> str:
        """Record the mechanism families found by the literature survey.

        Calling it again with a known name replaces that family; new names are
        added. Nodes name the family they belong to (`cluster`), so the status
        can show which families were tried and which are still open.
        """
        if not clusters:
            raise TreeError(
                "clusters is empty: list 4-6 mechanism families, each a kind of approach that differs in what "
                "it changes, not in a hyperparameter"
            )
        with self._edit() as st:
            by = {c["name"]: c for c in st.get("survey", [])}
            for c in clusters:
                name, mechanism = (
                    flatten(c.get("name"), 60),
                    flatten(c.get("mechanism"), 400),
                )
                if not name or not mechanism:
                    raise TreeError("every cluster needs a name and a mechanism")
                sources = [
                    flatten(
                        f"{x.get('id', '')} {x.get('title', '')}"
                        if isinstance(x, dict)
                        else x,
                        160,
                    )
                    for x in (c.get("sources") or [])
                ]
                by[name] = {
                    "name": name,
                    "mechanism": mechanism,
                    "sources": [x for x in sources if x],
                    "fit": flatten(c.get("fit"), 240),
                    "risk": flatten(c.get("risk"), 240),
                }
            st["survey"] = list(by.values())
            self._log(st, "survey", clusters=list(by))
            return "Survey recorded.\n" + self._survey_text(st)

    def _survey_text(self, st: dict) -> str:
        used: dict[str, list[str]] = {}
        for nid, n in sorted(st["nodes"].items(), key=lambda kv: node_number(kv[0])):
            if n.get("cluster"):
                used.setdefault(n["cluster"], []).append(nid)
        return "\n".join(
            f"  {c['name']}: {c['mechanism']}"
            + (f" (fit: {c['fit']})" if c.get("fit") else "")
            + (f" (risk: {c['risk']})" if c.get("risk") else "")
            + (f" [sources: {'; '.join(c['sources'])}]" if c.get("sources") else "")
            + f" -- tried in: {', '.join(used.get(c['name'], [])) or 'nothing yet'}"
            for c in st.get("survey", [])
        )

    # --------------------------------------------------------------- status

    def status(self) -> str:
        with self._edit() as st:
            return self._status(st)

    def _status(self, st: dict) -> str:
        events = self._events()
        nodes = st["nodes"]
        remaining = self._remaining(st)
        usable = remaining - st["reserve_minutes"]
        lines = [
            f"Clock: {_minutes(self.now() - st['started'])} min used, {remaining} min left. The last "
            f"{st['reserve_minutes']:.0f} min are reserved for the deliverable"
            + (
                ""
                if usable >= st["min_node_minutes"]
                else "; there is no room for a new node: wrap up"
            ),
            f"Metric: {st['metric']} ({st['direction']}). Judge: "
            + (
                "configured"
                if self.judge
                else "NOT configured (nodes cannot be scored)"
            ),
        ]
        champ = st["champion"]
        lines.append(
            f"Champion: {champ} (score {nodes[champ]['score']:.4f}); its {st['deliverable']} is at the "
            "project root"
            if champ
            else "Champion: none yet"
        )
        for n in nodes.values():
            if n["status"] == "running":
                used = _minutes(self.now() - n["started"])
                lines.append(
                    f"Running: {n['id']} ({n['kind']}) {used}/{n['budget_minutes']:.0f} min"
                    + (
                        " OVER BUDGET"
                        + (
                            "; its processes were stopped, close it with tree_finish"
                            if n.get("timed_out")
                            else ""
                        )
                        if used > n["budget_minutes"]
                        else ""
                    )
                )
        lines.append("Nodes:" if nodes else "Nodes: none yet")
        for n in sorted(nodes.values(), key=lambda x: node_number(x["id"])):
            e = events.get(n["id"])
            score = f"{n['score']:.4f}" if n.get("score") is not None else "-"
            lines.append(
                f"  {n['id']} {n['kind']:<7} {n['status']:<9} {score:<7} "
                f"{(e.outcome if e else 'in progress') or 'pending'} | from "
                f"{','.join(n['parents']) or 'root'} | {n['title']}"
            )
        if st.get("survey"):
            lines += ["Mechanism families from the survey:", self._survey_text(st)]
        pairs = self._merge_candidates(st)
        if pairs:
            lines.append(
                "Merge candidates (rows where two nodes disagree; a larger gain suits a merge): "
                + "; ".join(f"{a}+{b} {g:+.4f}" for a, b, g in pairs)
            )
        if st["insights"]:
            lines.append("Insights:")
            lines += [
                f"  {i['id']} [{i['status']}] {i['text']} (evidence: {', '.join(i['evidence']) or '-'})"
                for i in st["insights"]
            ]
        nudge = self._insight_nudge(st)
        if nudge:
            lines.append(nudge)
        return "\n".join(lines)

    def _merge_candidates(self, st: dict, top: int = 3) -> list[tuple[str, str, float]]:
        merged = {
            frozenset(n["parents"])
            for n in st["nodes"].values()
            if n["kind"] == "merge"
        }
        scored = {
            k: s
            for k, n in st["nodes"].items()
            if n["status"] == "done"
            and n.get("score") is not None
            and (s := self._scores(k))
        }
        ids = sorted(scored)
        pairs = []
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if frozenset((a, b)) not in merged and len(scored[a]) == len(scored[b]):
                    pairs.append(
                        (a, b, complementarity(scored[a], scored[b], st["direction"]))
                    )
        return [p for p in sorted(pairs, key=lambda p: -p[2])[:top] if p[2] > 0]

    # -------------------------------------------------------------- history

    def history(self, nodes: list[str] | None = None, *, full: bool = False) -> str:
        with self._edit():
            events = self._events()
            if nodes:
                missing = [n for n in nodes if n not in events]
                if missing:
                    raise TreeError(
                        f"no finished node {', '.join(missing)}; finished nodes: "
                        f"{', '.join(sorted(events, key=node_number)) or 'none'}"
                    )
                chosen = [events[n] for n in nodes]
            else:
                chosen = sorted(events.values(), key=lambda e: node_number(e.node))
            if not chosen:
                return "(no finished node yet)"
            body = "\n\n".join(render_event(e, compact=not full) for e in chosen)
            hint = (
                ""
                if full
                else (
                    "\n\n(Brief view. Pass the node ids with full=true for the complete record, or "
                    "run `git log` / `git show node/<id>` in the project root.)"
                )
            )
            return body + hint

    # ------------------------------------------------------------------ new

    def new_node(
        self,
        *,
        kind: str,
        title: str,
        hypothesis: str,
        category: str = "method",
        parents: list[str] | tuple[str, ...] = (),
        based_on: list[str] | tuple[str, ...] = (),
        minutes: float = 60.0,
        gpu: bool = True,
        cluster: str = "",
        component: str = "",
        plan: str = "",
        reuse: str = "",
        avoid: str = "",
        stop_condition: str = "",
    ) -> str:
        if kind not in KINDS:
            raise TreeError(f"kind must be one of {', '.join(KINDS)}")
        if category not in CATEGORIES:
            raise TreeError(f"category must be one of {', '.join(CATEGORIES)}")
        if not title.strip() or not hypothesis.strip():
            raise TreeError(
                "title and hypothesis are required: state what you expect and what would refute it"
            )
        parents = list(dict.fromkeys(parents))
        with self._edit() as st:
            nodes = st["nodes"]
            for p in parents:
                if (
                    p not in nodes
                    or nodes[p]["status"] != "done"
                    or nodes[p].get("score") is None
                ):
                    raise TreeError(f"parent {p} must be a finished, scored node")
            arity = {
                "explore": (0, 0),
                "refine": (1, 1),
                "merge": (2, 99),
                "pilot": (0, 1),
            }[kind]
            if not arity[0] <= len(parents) <= arity[1]:
                need = {
                    "explore": "no parents",
                    "refine": "exactly one parent",
                    "merge": "two or more parents",
                    "pilot": "at most one parent",
                }[kind]
                raise TreeError(f"a {kind} node takes {need}")
            known = {k for k, n in nodes.items() if n["status"] != "running"} | {
                i["id"] for i in st["insights"]
            }
            if known and not based_on and not parents:
                raise TreeError(
                    "based_on is required once there is history: cite the finished nodes or insights "
                    "(ids from tree_status) that this step builds on. If nothing in the history bears "
                    "on it, run a pilot first and cite that."
                )
            families = {c["name"] for c in st.get("survey", [])}
            if cluster and cluster not in families:
                raise TreeError(
                    f"cluster {cluster!r} is not a surveyed family; known: "
                    f"{', '.join(sorted(families)) or 'none (record the survey with tree_survey first)'}"
                )
            unknown = [b for b in based_on if b not in known]
            if unknown:
                raise TreeError(
                    f"based_on cites unknown or unfinished ids: {', '.join(unknown)}"
                )
            if gpu and any(
                n.get("gpu") for n in nodes.values() if n["status"] == "running"
            ):
                busy = [
                    k
                    for k, n in nodes.items()
                    if n["status"] == "running" and n.get("gpu")
                ]
                raise TreeError(
                    f"{', '.join(busy)} is still running and holds the GPU; finish it with tree_finish "
                    "first (or pass gpu=false for work that needs no GPU)"
                )
            usable = self._remaining(st) - st["reserve_minutes"]
            lo, hi = (
                PILOT_MINUTES
                if kind == "pilot"
                else (st["min_node_minutes"], NODE_MAX_MINUTES)
            )
            if usable < lo:
                raise TreeError(
                    f"only {max(usable, 0):.0f} min of usable time left, a {kind} node needs at least "
                    f"{lo:.0f}. Close the tree: tree_finalize."
                )
            budget = min(max(float(minutes), lo), hi, usable)
            nid = f"n{st['next']}"
            st["next"] += 1
            path, conflicts = self.repo.new_node(nid, parents, st["shared"])
            nodes[nid] = {
                "id": nid,
                "kind": kind,
                "parents": parents,
                "title": title.strip(),
                "category": category,
                "hypothesis": hypothesis.strip(),
                "based_on": sorted(set(based_on) | set(parents)),
                "budget_minutes": budget,
                "gpu": bool(gpu),
                "status": "running",
                "started": self.now(),
                "score": None,
                "conflicts": conflicts,
                "cluster": cluster or None,
                "component": component.strip(),
                "plan": plan.strip(),
                "reuse": reuse.strip(),
                "avoid": avoid.strip(),
                "stop_condition": stop_condition.strip(),
            }
            self._log(
                st, "node-start", node=nid, kind=kind, parents=parents, minutes=budget
            )
            deliver = (
                "This is a pilot: it is not scored. Write what you observed in NODE_REPORT.md."
                if kind == "pilot"
                else f"Before the node ends it must contain `{st['prediction_file']}` (the held-out predictions "
                f"that are scored), `{st['deliverable']}` (the deliverable) and `NODE_REPORT.md` "
                "(what was done, what the errors look like, what you would try next)."
            )
            return "\n".join(
                [
                    f"Created {nid} ({kind}) at {path.relative_to(self.root)}/ with a budget of {budget:.0f} min.",
                    f"Work only inside that directory. Give the sub-agent that path, the hypothesis and the budget. {deliver}",
                    *(
                        [
                            "Merge conflicts to resolve in the node: "
                            + ", ".join(conflicts)
                        ]
                        if conflicts
                        else []
                    ),
                    f"When it is done, call tree_finish {nid}.",
                    "",
                    self._save_brief(nid, self._brief(st, nid, path, deliver)),
                ]
            )

    def _related(
        self, nodes: dict, target: dict, events: dict[str, Event]
    ) -> list[Event]:
        """The other finished nodes closest in wording to the new one, oldest first."""

        def words(t: str) -> set[str]:
            return set(re.findall(r"[\w-]{3,}", t.lower()))

        want = words(
            " ".join(
                str(target.get(x) or "")
                for x in ("title", "hypothesis", "component", "plan")
            )
        )
        skip = set(target["parents"])

        def sim(e: Event) -> float:
            w = words(" ".join([e.subject, *e.narrative.values()]))
            return len(w & want) / (len(w | want) or 1)

        pool = [e for e in events.values() if e.node not in skip and e.node in nodes]
        ranked = [
            e for e in sorted(pool, key=sim, reverse=True)[:RELATED_NODES] if sim(e) > 0
        ]
        return sorted(ranked, key=lambda e: node_number(e.node))

    def _save_brief(self, nid: str, text: str) -> str:
        """Keep the brief a sub-agent was handed, so the run can be replayed."""
        try:
            (self.dir / "briefs").mkdir(parents=True, exist_ok=True)
            (self.dir / "briefs" / f"{nid}.md").write_text(text)
        except OSError:
            pass
        return text

    def _brief(self, st: dict, nid: str, path: Path, deliver: str) -> str:
        """Everything a sub-agent needs for one node, ready to hand over as it is."""
        node = st["nodes"][nid]
        events = self._events()
        parents = node["parents"]
        related = self._related(st["nodes"], node, events)
        shown = [events[p] for p in parents if p in events] + related
        cited = set(node["based_on"]) | {e.node for e in shown}
        insights = [
            i
            for i in st["insights"]
            if i["id"] in node["based_on"] or cited & set(i["evidence"])
        ]
        origin = (
            "It starts from the code of "
            + ", ".join(parents)
            + (" merged together." if len(parents) > 1 else ".")
            if parents
            else "It starts empty: build this step from scratch."
        )
        outputs_note = (
            f" `outputs/` already holds {parents[0]}'s files (shared on disk): read and reuse them, but write "
            "new files under new names instead of overwriting them."
            if parents and (self.repo.node_path(parents[0]) / "outputs").is_dir()
            else ""
        )
        left = node["budget_minutes"]
        until = time.strftime("%H:%M", time.localtime(self.now() + left * 60))
        blueprint = [
            f"- {k}: {node[key]}"
            for k, key in (
                ("title", "title"),
                ("kind", "kind"),
                ("family", "cluster"),
                ("component changed", "component"),
                ("hypothesis", "hypothesis"),
                ("what to build", "plan"),
                ("reuse", "reuse"),
                ("dead ends to avoid", "avoid"),
                ("stop early when", "stop_condition"),
            )
            if node.get(key)
        ]
        found = (
            "\n\n".join(render_event(e, compact=True) for e in shown)
            or "(nothing yet: this is the first step)"
        )
        if insights:
            found += "\n\nInsights that bear on it:\n" + "\n".join(
                f"- {i['id']} [{i['status']}] {i['text']}" for i in insights
            )
        if node["conflicts"]:
            found += (
                "\n\nMerging the parents' code left conflicts in: "
                + ", ".join(node["conflicts"])
                + ". Resolve them first (keep what served each parent's result)."
            )
        rules = [
            f"You have {left:.0f} minutes from now (until about {until}); after that anything still running in this "
            "directory is stopped and the node is closed. Test on a "
            "small slice first, start long training with `nohup <command> > outputs/train.log 2>&1 &` and check "
            "its log with separate short commands. Write predictions as soon as you have a model, then "
            "overwrite them if you improve.",
            "Work only inside this directory. Do not modify `data/`. Put checkpoints and logs under "
            f"`outputs/`.{outputs_note}",
            "Test the hypothesis of this step; do not drift into a different idea. If the stop condition is "
            "met, stop, write what you found, and finish.",
            "The full record of the steps above is in git: `git log --all` and `git show node/<id>` work from "
            "this directory. Read an ancestor's complete entry when the summary leaves a question open.",
            "Before you finish, make sure no training process you started is still running (the node's "
            "leftover processes are stopped when it is closed).",
        ]
        return "\n".join(
            [
                f"## Brief for the sub-agent (give it verbatim; its working directory is {path.relative_to(self.root)}/)",
                "",
                f"This directory is your copy of the project for one step of a longer research effort. {origin}",
                "",
                "### The step",
                *blueprint,
                "",
                "### What earlier steps found",
                found,
                "",
                "### Rules",
                *[f"- {r}" for r in rules],
                "",
                "### What you must leave in this directory",
                deliver,
                "Your final reply is a short summary of the step.",
            ]
        )

    # --------------------------------------------------------------- finish

    def finish(
        self,
        node: str,
        *,
        change: str,
        observation: str,
        next_step: str = "",
        source: str = "",
        abandon_reason: str = "",
    ) -> str:
        with self._edit() as st:
            return self._finish(
                st,
                node,
                change=change,
                observation=observation,
                next_step=next_step,
                source=source,
                abandon_reason=abandon_reason,
            )

    def _evaluate(self, st: dict, art: Path) -> dict:
        pred, deliv = art / st["prediction_file"], art / st["deliverable"]
        if not pred.is_file():
            return {"error": f"no {st['prediction_file']} in the node"}
        judge = self.judge
        if judge is None:
            return {
                "error": "no judge configured (EVOSCIENTIST_TREE_JUDGE_URL is not set); the node cannot be scored"
            }
        res = judge.score(pred, deliv if deliv.is_file() else None)
        if res.get("error"):
            return res
        if not deliv.is_file():
            return {
                "error": f"held-out predictions scored, but there is no {st['deliverable']}"
            }
        sha = res.get("eval_sha") or ""
        if st["eval_sha"] and sha and sha != st["eval_sha"]:
            return {"error": "the judge changed during the run; refusing to score"}
        if sha and not st["eval_sha"]:
            st["eval_sha"] = sha
        return res

    def _finish(
        self,
        st: dict,
        nid: str,
        *,
        change: str,
        observation: str,
        next_step: str,
        source: str,
        abandon_reason: str,
    ) -> str:
        node = st["nodes"].get(nid)
        if node is None or node["status"] != "running":
            running = [k for k, n in st["nodes"].items() if n["status"] == "running"]
            raise TreeError(
                f"{nid} is not a running node; running: {', '.join(running) or 'none'}"
            )
        direction = st["direction"]
        path = self.repo.node_path(nid)
        # A job the sub-agent detached would keep the GPU after the node is closed.
        killed = kill_processes_under(path) if path.is_dir() else []
        art = self.dir / "artifacts" / nid
        art.mkdir(parents=True, exist_ok=True)
        for name in (st["prediction_file"], st["deliverable"], "NODE_REPORT.md"):
            if (path / name).is_file():
                shutil.copy2(path / name, art / name)
        if path.is_dir():
            snap = snapshot_node(path, art, skip=tuple(st["shared"]))
            self._log(st, "node-snapshot", node=nid, **snap)
        pilot = node["kind"] == "pilot"
        if abandon_reason:
            res: dict = {"error": f"abandoned: {abandon_reason}"}
        elif pilot:
            res = {}
        else:
            res = self._evaluate(st, art)
        scored = "score" in res and not res.get("error")
        status = (
            "abandoned"
            if abandon_reason
            else ("done" if (pilot or scored) else "failed")
        )
        minutes = _minutes(self.now() - node["started"])

        deltas: dict[str, Delta] = {}
        if scored:
            mine_scores = res["per_sample"]
            others = [*node["parents"], *([st["champion"]] if st["champion"] else [])]
            for o in dict.fromkeys(others):
                theirs = self._scores(o)
                if theirs is not None and len(theirs) == len(mine_scores):
                    deltas[o] = paired_delta(mine_scores, theirs, direction=direction)
            self._scores_path(nid).parent.mkdir(parents=True, exist_ok=True)
            self._scores_path(nid).write_text(
                json.dumps(
                    {"per_sample": mine_scores, "slices": res.get("slices") or []}
                )
            )

        events = self._events()
        updates: dict[str, str] = {}
        parent_d = {p: deltas[p] for p in node["parents"] if p in deltas}
        if status in ("failed", "abandoned"):
            outcome = f"abandoned: {res.get('error') or 'no result'}"
        elif pilot:
            outcome = "evidence: pilot"
        elif parent_d and all(d.better for d in parent_d.values()):
            outcome = "retained: better than " + ", ".join(
                f"{p} ({d.text()})" for p, d in parent_d.items()
            )
        elif any(d.worse for d in parent_d.values()):
            outcome = "abandoned: worse than " + ", ".join(
                f"{p} ({d.text()})" for p, d in parent_d.items() if d.worse
            )
        elif parent_d:
            outcome = "pending: not resolved against " + ", ".join(
                f"{p} ({d.text()})" for p, d in parent_d.items()
            )
        else:
            outcome = "retained: new line"
        for p, d in parent_d.items():
            if scored and d.better and p in events:
                updates[p] = f"superseded by {nid}: {d.text()}"

        champ = st["champion"]
        became_best = False
        if scored and (champ is None or (champ in deltas and deltas[champ].better)):
            st["champion"] = nid
            shutil.copy2(art / st["deliverable"], self.root / st["deliverable"])
            became_best = True

        slices = (
            slice_table(res.get("slices") or [], res.get("per_sample") or [])
            if scored
            else {}
        )
        unresolved, diffstat, _ = self.repo.stage(nid, node["parents"])
        trailers = {
            "Event-Id": nid,
            "Kind": node["kind"],
            "Category": node["category"],
            "Parents": ",".join(node["parents"]) or "root",
            "Status": status,
            "Score": f"{res['score']:.4f}" if scored else "",
            "Delta": "; ".join(f"vs {k} {d.text()}" for k, d in deltas.items()),
            "Slices": ", ".join(f"{k} {v['share']:.3f}" for k, v in slices.items()),
            "Source": source,
            "Cluster": node.get("cluster") or "",
            "Based-On": ",".join(node["based_on"]),
            "Minutes": f"{minutes}/{node['budget_minutes']:.0f}",
            "Diffstat": diffstat,
            "Error": res.get("error") or "",
            "Unresolved-Conflicts": ",".join(unresolved),
            "Eval-Sha": (st["eval_sha"] or "")[:16],
        }
        narrative = {
            "Hypothesis": node["hypothesis"],
            "Source": source
            or (f"survey family {node['cluster']}" if node.get("cluster") else ""),
            "Change": change,
            "Observation": observation,
            "Next": next_step,
        }
        sha = self.repo.commit(
            nid,
            node["parents"],
            compose_message(node["category"], node["title"], narrative, trailers),
        )
        self.repo.set_outcome(sha, outcome)
        for p, text in updates.items():
            self.repo.set_outcome(events[p].sha, text)

        node.update(
            status=status,
            score=res.get("score") if scored else None,
            error=res.get("error"),
            minutes=minutes,
            ended=self.now(),
            sha=sha,
            slices={k: v["share"] for k, v in slices.items()},
            deltas={k: [d.mean, d.low, d.high] for k, d in deltas.items()},
        )
        out = [
            f"{nid} closed as {status}"
            + (f", score {res['score']:.4f}" if scored else "")
            + f" after {minutes} min (budget {node['budget_minutes']:.0f})."
        ]
        if deltas:
            out.append(
                "Against "
                + "; ".join(f"{k} {d.text()}" for k, d in deltas.items())
                + "."
            )
        out.append(f"Outcome: {outcome}.")
        if became_best:
            out.append(
                f"{nid} is the new champion; its {st['deliverable']} is now at the project root."
            )
        elif scored:
            out.append(
                "Not better than the champion with confidence; the root deliverable is unchanged."
            )
        if res.get("error") and status == "failed":
            out.append(f"Error: {res['error']}")
        if unresolved:
            out.append(
                "Unresolved merge conflicts were committed as they were: "
                + ", ".join(unresolved)
            )
        if killed:
            out.append(
                f"Stopped {len(killed)} process(es) the node left running in its directory."
            )
        self._log(
            st,
            "node-end",
            node=nid,
            status=status,
            score=node["score"],
            outcome=outcome,
            champion=st["champion"],
            killed=len(killed),
        )
        nudge = self._insight_nudge(st)
        if nudge:
            out.append(nudge)
        out.append(f"{self._remaining(st)} min left.")
        return " ".join(out)

    # ------------------------------------------------------------- insights

    def insight(
        self,
        *,
        text: str = "",
        evidence: list[str] | tuple[str, ...] = (),
        status: str = "open",
        insight_id: str = "",
    ) -> str:
        if status not in INSIGHT_STATUS:
            raise TreeError(f"status must be one of {', '.join(INSIGHT_STATUS)}")
        with self._edit() as st:
            finished = {k for k, n in st["nodes"].items() if n["status"] != "running"}
            bad = [e for e in evidence if e not in finished]
            if bad:
                raise TreeError(
                    f"evidence cites unknown or unfinished nodes: {', '.join(bad)}"
                )
            if insight_id:
                found = next((i for i in st["insights"] if i["id"] == insight_id), None)
                if found is None:
                    raise TreeError(f"no insight {insight_id}")
                if text.strip():
                    found["text"] = flatten(text, 400)
                found["status"] = status
                found["evidence"] = sorted(set(found["evidence"]) | set(evidence))
                st["insight_mark"] = self._closed(st)
                self._log(st, "insight", id=insight_id, status=status)
                return f"{insight_id} updated: [{found['status']}] {found['text']}"
            if not text.strip():
                raise TreeError(
                    "text is required for a new insight: when the situation looks like X, the step that "
                    "worked or failed was Y"
                )
            new = {
                "id": f"I{len(st['insights']) + 1}",
                "text": flatten(text, 400),
                "status": status,
                "evidence": sorted(set(evidence)),
            }
            st["insights"].append(new)
            st["insight_mark"] = self._closed(st)
            self._log(st, "insight", id=new["id"], status=status)
            return f"{new['id']} recorded: [{new['status']}] {new['text']}"

    # ------------------------------------------------------------- finalize

    def _export(self, st: dict, events: dict[str, Event]) -> None:
        """Write `.research/trajectory.json`: the whole record in one machine-readable file."""
        nodes = []
        for nid, n in sorted(st["nodes"].items(), key=lambda kv: node_number(kv[0])):
            e = events.get(nid)
            nodes.append(
                {
                    **n,
                    "event": None
                    if e is None
                    else {
                        "sha": e.sha,
                        "subject": e.subject,
                        "narrative": e.narrative,
                        "trailers": e.trailers,
                        "parents": e.parents,
                        "outcome": e.outcome,
                    },
                    "brief": f"briefs/{nid}.md",
                    "artifacts": f"artifacts/{nid}",
                }
            )
        doc = {
            "version": 1,
            "root": str(self.root),
            "started": st["started"],
            "deadline": st["deadline"],
            "metric": st["metric"],
            "direction": st["direction"],
            "champion": st["champion"],
            "eval_sha": st["eval_sha"],
            "survey": st.get("survey", []),
            "insights": st["insights"],
            "nodes": nodes,
            "files": {
                "state": "state.json",
                "events": "events.jsonl",
                "calls": "calls.jsonl",
                "gpu": "gpu.jsonl",
                "summary": "summary.json",
            },
        }
        (self.dir / "trajectory.json").write_text(
            json.dumps(doc, indent=1, ensure_ascii=False, default=str)
        )

    def finalize(self) -> str:
        with self._edit() as st:
            for nid in [k for k, n in st["nodes"].items() if n["status"] == "running"]:
                self._finish(
                    st,
                    nid,
                    change="(not recorded)",
                    observation="The run ended before this node was closed.",
                    next_step="",
                    source="",
                    abandon_reason="the run ended before the node was closed",
                )
            events = self._events()
            champ = st["champion"]
            keep = ancestry(events, champ)
            for nid, e in events.items():
                o = e.outcome
                if o.startswith(("superseded", "abandoned", "evidence")):
                    continue
                if nid in keep:
                    final = "retained: part of the final solution" + (
                        f" ({champ})" if nid != champ else ""
                    )
                elif o.startswith("pending"):
                    final = (
                        "abandoned: never resolved better than its parent; "
                        + o.split(":", 1)[1].strip()
                    )
                else:
                    final = f"abandoned: not carried into the final solution ({champ or 'none'})"
                self.repo.set_outcome(e.sha, final)
            events = self._events()
            if champ:
                src = self.dir / "artifacts" / champ / st["deliverable"]
                if src.is_file():
                    shutil.copy2(src, self.root / st["deliverable"])
                self.repo.tag_final(
                    events[champ].sha,
                    f"final solution: {champ} {events[champ].subject} "
                    f"({st['metric']} {st['nodes'][champ]['score']:.4f})",
                )
            text = render_trajectory(
                sorted(events.values(), key=lambda e: node_number(e.node)),
                champion=champ,
                insights=st["insights"],
            )
            (self.root / "TRAJECTORY.md").write_text(text)
            summary = {
                "champion": champ,
                "score": st["nodes"][champ]["score"] if champ else None,
                "nodes": len(events),
                "eval_sha": st["eval_sha"],
                "minutes": _minutes(self.now() - st["started"]),
            }
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=1))
            self._export(st, events)
            st["finalized"] = True
            self._log(st, "finalize", champion=champ, nodes=len(events))
            stop_gpu_log(self.dir / "gpu.jsonl")
            stop_ticker(str(self.dir))
            return text
