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
            )
            return "Research tree initialised.\n\n" + self._status(st)

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
                    + (" OVER BUDGET" if used > n["budget_minutes"] else "")
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
            }
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
        art = self.dir / "artifacts" / nid
        art.mkdir(parents=True, exist_ok=True)
        for name in (st["prediction_file"], st["deliverable"], "NODE_REPORT.md"):
            if (path / name).is_file():
                shutil.copy2(path / name, art / name)
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
            "Based-On": ",".join(node["based_on"]),
            "Minutes": f"{minutes}/{node['budget_minutes']:.0f}",
            "Diffstat": diffstat,
            "Error": res.get("error") or "",
            "Unresolved-Conflicts": ",".join(unresolved),
            "Eval-Sha": (st["eval_sha"] or "")[:16],
        }
        narrative = {
            "Hypothesis": node["hypothesis"],
            "Source": source,
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
            return f"{new['id']} recorded: [{new['status']}] {new['text']}"

    # ------------------------------------------------------------- finalize

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
            return text
