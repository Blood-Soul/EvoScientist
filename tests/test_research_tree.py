"""Research tree: facts come from tools, the record is git history."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from EvoScientist.research_tree.tree import ResearchTree, TreeError

ROWS = 400


class FakeJudge:
    """Scores a prediction file of one 0/1 per line; the 'frozen' hash is fixed."""

    def __init__(self, sha: str = "frozen-judge-0001"):
        self.sha = sha
        self.calls = 0

    def score(self, prediction: Path, deliverable: Path | None) -> dict:
        self.calls += 1
        rows = [float(x) for x in prediction.read_text().split()]
        return {
            "score": sum(rows) / len(rows),
            "per_sample": rows,
            "slices": ["right" if r else "wrong" for r in rows],
            "eval_sha": self.sha,
        }


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, minutes: float) -> None:
        self.t += minutes * 60


def accuracy_rows(n_right: int, shift: int = 0) -> str:
    """ROWS rows with n_right ones, the ones starting at `shift` (wrapping)."""
    rows = [0] * ROWS
    for i in range(n_right):
        rows[(i + shift) % ROWS] = 1
    return "\n".join(map(str, rows)) + "\n"


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "data").mkdir()
    (root / "data" / "train.txt").write_text("rows")
    clock, judge = Clock(), FakeJudge()
    tree = ResearchTree(root, judge=judge, now=clock)
    return tree, root, clock, judge


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def do_node(
    tree, root, clock, *, minutes_used, n_right, shift=0, deliverable="d", **new
):
    out = tree.new_node(**new)
    nid = out.split()[1]
    path = root / "nodes" / nid
    (path / "holdout_predictions.csv").write_text(accuracy_rows(n_right, shift))
    (path / "submission.csv").write_text(deliverable)
    (path / "model.py").write_text(f"# {nid}\n")
    clock.advance(minutes_used)
    return nid


def test_init_needs_budget_and_is_once(env):
    tree, *_ = env
    with pytest.raises(TreeError, match="budget_minutes"):
        tree.init()
    tree.init(budget_minutes=360)
    with pytest.raises(TreeError, match="already initialised"):
        tree.init(budget_minutes=360)


def test_full_run_records_facts_from_the_judge(env):
    tree, root, clock, _judge = env
    tree.init(budget_minutes=360)
    n1 = do_node(
        tree,
        root,
        clock,
        minutes_used=30,
        n_right=200,
        kind="explore",
        title="baseline",
        hypothesis="a plain model sets a floor",
        minutes=60,
        deliverable="v1",
    )
    msg = tree.finish(
        n1, change="plain fine-tune", observation="agent claims 0.99", next_step="try X"
    )
    assert "champion" in msg
    assert "score 0.5000" in msg
    assert (root / "submission.csv").read_text() == "v1"

    # the score in the record is the judge's, whatever the agent wrote
    log = git(root, "log", "-1", "--format=%B", "node/n1")
    assert "Score: 0.5000" in log
    assert "0.99" in log
    assert "Eval-Sha: frozen-judge-00" in log
    assert (
        git(root, "notes", "--ref=outcome", "show", "node/n1") == "retained: new line"
    )

    # a clearly better refinement supersedes its parent and becomes champion
    n2 = do_node(
        tree,
        root,
        clock,
        minutes_used=40,
        n_right=320,
        kind="refine",
        title="better head",
        hypothesis="a task head lifts accuracy",
        parents=[n1],
        based_on=[n1],
        minutes=60,
        deliverable="v2",
    )
    msg = tree.finish(n2, change="new head", observation="improved", next_step="")
    assert "new champion" in msg
    assert (root / "submission.csv").read_text() == "v2"
    assert git(root, "notes", "--ref=outcome", "show", "node/n1").startswith(
        "superseded by n2"
    )
    assert git(root, "notes", "--ref=outcome", "show", "node/n2").startswith(
        "retained: better than n1"
    )

    # the DAG is the tree: n2's commit has n1's event commit as parent
    assert git(root, "rev-parse", "node/n2^") == git(root, "rev-parse", "node/n1")

    # an unresolved change does not take the champion
    n3 = do_node(
        tree,
        root,
        clock,
        minutes_used=30,
        n_right=322,
        shift=0,
        kind="refine",
        title="tweak",
        hypothesis="small tweak",
        parents=[n2],
        based_on=[n2, n1],
        minutes=60,
        deliverable="v3",
    )
    msg = tree.finish(n3, change="tweak", observation="same", next_step="")
    assert "Not better" in msg
    assert (root / "submission.csv").read_text() == "v2"
    assert git(root, "notes", "--ref=outcome", "show", "node/n3").startswith("pending")

    text = tree.finalize()
    assert "research/final" in git(root, "tag")
    assert git(root, "notes", "--ref=outcome", "show", "node/n3").startswith(
        "abandoned: never resolved"
    )
    assert git(root, "notes", "--ref=outcome", "show", "node/n1").startswith(
        "superseded"
    )
    assert "### n2" in text
    assert (root / "TRAJECTORY.md").is_file()
    assert (root / "submission.csv").read_text() == "v2"


def test_history_is_required_to_be_cited(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    n1 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=200,
        kind="explore",
        title="a",
        hypothesis="h",
        minutes=30,
    )
    tree.finish(n1, change="c", observation="o")
    with pytest.raises(TreeError, match="based_on is required"):
        tree.new_node(kind="explore", title="b", hypothesis="h")
    with pytest.raises(TreeError, match="unknown or unfinished"):
        tree.new_node(kind="explore", title="b", hypothesis="h", based_on=["n9"])
    with pytest.raises(TreeError, match="unknown or unfinished"):
        tree.new_node(kind="explore", title="b", hypothesis="h", based_on=["I1"])
    tree.new_node(kind="explore", title="b", hypothesis="h", based_on=[n1])


def test_one_gpu_node_at_a_time_and_kind_rules(env):
    tree, _root, _clock, _ = env
    tree.init(budget_minutes=360)
    tree.new_node(kind="explore", title="a", hypothesis="h")
    with pytest.raises(TreeError, match="holds the GPU"):
        tree.new_node(kind="explore", title="b", hypothesis="h")
    tree.new_node(kind="explore", title="b", hypothesis="h", gpu=False)
    with pytest.raises(TreeError, match="exactly one parent"):
        tree.new_node(kind="refine", title="c", hypothesis="h", gpu=False)
    with pytest.raises(TreeError, match="finished, scored"):
        tree.new_node(
            kind="refine", title="c", hypothesis="h", parents=["n1"], gpu=False
        )


def test_pilot_is_evidence_not_a_candidate(env):
    tree, root, clock, judge = env
    tree.init(budget_minutes=360)
    n1 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=200,
        kind="explore",
        title="a",
        hypothesis="h",
        minutes=30,
    )
    tree.finish(n1, change="c", observation="o")
    created = tree.new_node(
        kind="pilot",
        title="does X matter",
        hypothesis="X changes the error mix",
        parents=[n1],
        minutes=100,
    )
    pilot = created.split()[1]
    assert "budget of 20 min" in created  # a pilot is capped however much is asked for
    clock.advance(8)
    calls = judge.calls
    msg = tree.finish(pilot, change="3-minute probe", observation="X moves 12% of rows")
    assert judge.calls == calls  # a pilot is not scored
    assert "closed as done" in msg
    assert (
        git(root, "notes", "--ref=outcome", "show", f"node/{pilot}")
        == "evidence: pilot"
    )
    with pytest.raises(TreeError, match="finished, scored"):
        tree.new_node(
            kind="refine", title="r", hypothesis="h", parents=[pilot], gpu=False
        )
    # but later nodes can cite it
    tree.new_node(kind="explore", title="b", hypothesis="h", based_on=[pilot])
    # a pilot cannot become champion; finalize keeps its outcome
    tree.finalize()
    assert (
        git(root, "notes", "--ref=outcome", "show", f"node/{pilot}")
        == "evidence: pilot"
    )


def test_missing_prediction_fails_the_node_and_keeps_the_record(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    nid = tree.new_node(kind="explore", title="a", hypothesis="h").split()[1]
    (root / "nodes" / nid / "model.py").write_text("x")
    clock.advance(20)
    msg = tree.finish(nid, change="tried", observation="training crashed")
    assert "closed as failed" in msg
    assert "Status: failed" in git(root, "log", "-1", "--format=%B", f"node/{nid}")
    assert git(root, "notes", "--ref=outcome", "show", f"node/{nid}").startswith(
        "abandoned"
    )
    assert "Champion: none yet" in tree.status()


def test_no_judge_means_no_score(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    clock = Clock()
    tree = ResearchTree(root, judge=None, now=clock)
    tree.init(budget_minutes=360)
    nid = tree.new_node(kind="explore", title="a", hypothesis="h").split()[1]
    (root / "nodes" / nid / "holdout_predictions.csv").write_text(accuracy_rows(300))
    (root / "nodes" / nid / "submission.csv").write_text("d")
    msg = tree.finish(nid, change="c", observation="o")
    assert "closed as failed" in msg
    assert "no judge configured" in msg
    assert not (root / "submission.csv").exists()


def test_judge_change_is_refused(env):
    tree, root, clock, judge = env
    tree.init(budget_minutes=360)
    n1 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=200,
        kind="explore",
        title="a",
        hypothesis="h",
    )
    tree.finish(n1, change="c", observation="o")
    judge.sha = "someone-edited-the-judge"
    n2 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=390,
        kind="explore",
        title="b",
        hypothesis="h",
        based_on=[n1],
    )
    msg = tree.finish(n2, change="c", observation="o")
    assert "judge changed" in msg
    assert "failed" in msg


def test_clock_limits_new_nodes(env):
    tree, _root, clock, _ = env
    tree.init(budget_minutes=120)
    clock.advance(90)  # 30 min left, 15 reserved: 15 usable < 20
    with pytest.raises(TreeError, match="usable time"):
        tree.new_node(kind="explore", title="a", hypothesis="h")
    out = tree.new_node(kind="pilot", title="p", hypothesis="h")
    assert "budget of 15 min" in out  # clipped to the usable time
    assert "wrap up" in tree.status()


def test_deadline_from_the_harness_wins(env, monkeypatch):
    tree, _root, clock, _ = env
    monkeypatch.setenv("EVOSCIENTIST_TREE_DEADLINE_EPOCH", str(clock.t + 3600))
    tree.init(budget_minutes=9999)
    assert "60.0 min left" in tree.status()


def test_insights_cite_real_nodes_and_update(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    n1 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=200,
        kind="explore",
        title="a",
        hypothesis="h",
    )
    tree.finish(n1, change="c", observation="o")
    with pytest.raises(TreeError, match="unknown or unfinished"):
        tree.insight(text="x", evidence=["n7"])
    assert tree.insight(text="plain models stall at 0.5", evidence=[n1]).startswith(
        "I1 recorded"
    )
    assert "contradicted" in tree.insight(
        insight_id="I1", status="contradicted", evidence=[n1]
    )
    tree.new_node(kind="explore", title="b", hypothesis="h", based_on=["I1"])
    assert "I1 [contradicted]" in tree.status()


def test_merge_candidates_show_complementary_nodes(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=600)
    n1 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=200,
        shift=0,
        kind="explore",
        title="a",
        hypothesis="h",
    )
    tree.finish(n1, change="c", observation="o")
    n2 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=200,
        shift=200,
        kind="explore",
        title="b",
        hypothesis="h",
        based_on=[n1],
    )
    tree.finish(n2, change="c", observation="o")
    assert "Merge candidates" in tree.status()
    assert f"{n1}+{n2}" in tree.status()


def test_history_brief_and_full(env):
    tree, root, clock, _ = env
    assert tree.init(budget_minutes=360)
    assert tree.history() == "(no finished node yet)"
    n1 = do_node(
        tree,
        root,
        clock,
        minutes_used=10,
        n_right=200,
        kind="explore",
        title="a",
        hypothesis="h1",
    )
    tree.finish(n1, change="c1", observation="o1", next_step="next-step-text")
    brief, full = tree.history(), tree.history([n1], full=True)
    assert "hypothesis: h1" in brief
    # the brief view keeps the next step too, so the planner sees where each node pointed
    assert "next: next-step-text" in brief
    assert "next-step-text" in full
    assert "commit:" in full
    with pytest.raises(TreeError, match="no finished node n9"):
        tree.history(["n9"])


# ---- capabilities carried over from the external loop -----------------------


def close(tree, root, clock, n_right, **new):
    nid = do_node(tree, root, clock, minutes_used=10, n_right=n_right, **new)
    return nid, tree.finish(nid, change="c", observation="o", next_step="try X")


def test_survey_families_are_recorded_validated_and_tracked(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    with pytest.raises(TreeError, match="record the survey with tree_survey first"):
        tree.new_node(kind="explore", title="a", hypothesis="h", cluster="abstain")
    out = tree.survey(
        [
            {
                "name": "abstain",
                "mechanism": "calibrate a no-answer threshold",
                "sources": ["2101.1 A paper"],
            },
            {"name": "long-ctx", "mechanism": "window the long plots", "risk": "slow"},
        ]
    )
    assert "abstain" in out
    assert "tried in: nothing yet" in out
    with pytest.raises(
        TreeError, match="not a surveyed family; known: abstain, long-ctx"
    ):
        tree.new_node(kind="explore", title="a", hypothesis="h", cluster="nope")
    with pytest.raises(TreeError, match="name and a mechanism"):
        tree.survey([{"name": "x", "mechanism": ""}])
    n1, _ = close(
        tree,
        root,
        clock,
        200,
        kind="explore",
        title="baseline",
        hypothesis="floor",
        cluster="abstain",
        plan="plain model",
    )
    status = tree.status()
    assert "abstain: calibrate a no-answer threshold" in status
    assert f"tried in: {n1}" in status
    assert "long-ctx" in status
    assert "tried in: nothing yet" in status
    log = git(root, "log", "-1", "--format=%B", f"node/{n1}")
    assert "Cluster: abstain" in log
    assert "Source: survey family abstain" in log
    # a family can be corrected by naming it again
    tree.survey([{"name": "abstain", "mechanism": "a revised mechanism"}])
    assert "a revised mechanism" in tree.status()


def test_new_node_returns_a_brief_with_the_relevant_record(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    n1, _ = close(
        tree,
        root,
        clock,
        200,
        kind="explore",
        title="span extraction baseline",
        hypothesis="span extraction gives a floor",
        plan="plain",
    )
    n2, _ = close(
        tree,
        root,
        clock,
        260,
        kind="explore",
        title="retrieval of sentences",
        hypothesis="retrieval helps long plots",
        based_on=[n1],
        plan="retrieve",
    )
    tree.insight(text="plain span models stall on long plots", evidence=[n1])
    out = tree.new_node(
        kind="refine",
        title="span extraction with a no-answer head",
        parents=[n2],
        based_on=[n2, "I1"],
        hypothesis="a no-answer head raises span extraction accuracy",
        plan="add the head and calibrate it",
        component="decision rule",
        avoid="plain windowing already failed",
        stop_condition="no gain after one run",
    )
    assert "## Brief for the sub-agent" in out
    assert "what to build: add the head and calibrate it" in out
    assert "component changed: decision rule" in out
    assert "dead ends to avoid: plain windowing already failed" in out
    assert "### n2 [method] retrieval of sentences" in out  # the parent's record
    assert "next: try X" in out  # ... including where it pointed
    assert (
        "### n1 [method] span extraction baseline" in out
    )  # the closest other node, by wording
    assert "I1 [open] plain span models stall" in out  # an insight that bears on it
    assert "git log --all" in out
    assert "holdout_predictions.csv" in out
    assert "starts from the code of n2" in out


def test_insight_reminder_after_three_closed_nodes(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    n1, _ = close(tree, root, clock, 100, kind="explore", title="a", hypothesis="h")
    n2, msg2 = close(
        tree, root, clock, 120, kind="explore", title="b", hypothesis="h", based_on=[n1]
    )
    assert "tree_insight" not in msg2
    n3, msg3 = close(
        tree, root, clock, 140, kind="explore", title="c", hypothesis="h", based_on=[n1]
    )
    assert "3 nodes have closed since the last insight" in msg3
    assert "3 nodes have closed" in tree.status()
    tree.insight(text="b and c beat a", evidence=[n1, n2, n3])
    assert "since the last insight" not in tree.status()


def test_pilots_do_not_count_towards_the_insight_reminder(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    n1, _ = close(tree, root, clock, 100, kind="explore", title="a", hypothesis="h")
    for i in range(3):
        nid = tree.new_node(
            kind="pilot", title=f"p{i}", hypothesis="h", based_on=[n1], minutes=5
        ).split()[1]
        (root / "nodes" / nid / "NODE_REPORT.md").write_text("saw it")
        clock.advance(5)
        tree.finish(nid, change="c", observation="o")
    assert "since the last insight" not in tree.status()


@pytest.mark.skipif(not Path("/proc/self/cwd").exists(), reason="needs /proc")
def test_closing_a_node_stops_what_it_left_running(env):
    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    nid = tree.new_node(kind="explore", title="a", hypothesis="h").split()[1]
    node_dir = root / "nodes" / nid
    (node_dir / "holdout_predictions.csv").write_text(accuracy_rows(200))
    (node_dir / "submission.csv").write_text("d")
    leftover = subprocess.Popen(["sleep", "300"], cwd=node_dir)
    outside = subprocess.Popen(["sleep", "300"], cwd=root)
    try:
        clock.advance(10)
        msg = tree.finish(nid, change="c", observation="o")
        assert "Stopped 1 process(es)" in msg
        assert leftover.wait(timeout=15) != 0
        assert outside.poll() is None  # only the node's own directory is touched
    finally:
        outside.kill()
        leftover.kill()


def _command_judge(tmp_path, body_extra=""):
    script = tmp_path / "score.py"
    script.write_text(
        "import json, sys\n"
        "rows = [float(x) for x in open(sys.argv[1]).read().split()]\n"
        "print(json.dumps({'score': sum(rows) / len(rows), 'per_sample': rows, "
        "'slices': ['hit' if r else 'miss' for r in rows]}))\n" + body_extra
    )
    return script


def test_a_local_scoring_command_is_a_judge_and_edits_to_it_are_caught(tmp_path):
    import sys

    from EvoScientist.research_tree.judge import CommandJudge

    script = _command_judge(tmp_path)
    root = tmp_path / "ws"
    root.mkdir()
    clock = Clock()
    tree = ResearchTree(
        root, judge=CommandJudge([sys.executable, str(script)]), now=clock
    )
    tree.init(budget_minutes=360)
    n1, msg = close(tree, root, clock, 200, kind="explore", title="a", hypothesis="h")
    assert "score 0.5000" in msg
    assert "Eval-Sha: " in git(root, "log", "-1", "--format=%B", f"node/{n1}")
    script.write_text(
        script.read_text() + "# loosened\n"
    )  # the scorer is edited mid-run
    _, msg2 = close(
        tree, root, clock, 390, kind="explore", title="b", hypothesis="h", based_on=[n1]
    )
    assert "closed as failed" in msg2
    assert "judge changed" in msg2


def test_command_judge_reports_failures_as_errors_not_scores(tmp_path):
    import sys

    from EvoScientist.research_tree.judge import CommandJudge

    pred = tmp_path / "p.csv"
    pred.write_text("1\n0\n")
    bad = tmp_path / "bad.py"
    bad.write_text("print('not json')\n")
    assert "error" in CommandJudge([sys.executable, str(bad)]).score(pred, None)
    assert "error" in CommandJudge(["/no/such/scorer"]).score(pred, None)


def test_judge_from_env_prefers_url_then_command(monkeypatch):
    from EvoScientist.research_tree.judge import CommandJudge, HttpJudge, judge_from_env

    monkeypatch.delenv("EVOSCIENTIST_TREE_JUDGE_URL", raising=False)
    monkeypatch.delenv("EVOSCIENTIST_TREE_JUDGE_COMMAND", raising=False)
    assert judge_from_env() is None
    monkeypatch.setenv("EVOSCIENTIST_TREE_JUDGE_COMMAND", "python /x/score.py --flag")
    judge = judge_from_env()
    assert isinstance(judge, CommandJudge)
    assert judge.command == [
        "python",
        "/x/score.py",
        "--flag",
    ]
    monkeypatch.setenv("EVOSCIENTIST_TREE_JUDGE_URL", "http://127.0.0.1:1/")
    assert isinstance(judge_from_env(), HttpJudge)


def test_every_tree_action_is_written_to_the_event_log(env):
    import json

    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    n1, _ = close(tree, root, clock, 200, kind="explore", title="a", hypothesis="h")
    tree.insight(text="a stalls", evidence=[n1])
    tree.finalize()
    events = [
        json.loads(x)
        for x in (root / ".research" / "events.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in events] == [
        "init",
        "node-start",
        "node-snapshot",
        "node-end",
        "insight",
        "finalize",
    ]
    assert events[3]["node"] == n1
    assert events[3]["status"] == "done"
    assert (
        json.loads((root / ".research" / "state.json").read_text())["finalized"] is True
    )


@pytest.mark.skipif(not Path("/proc/self/cwd").exists(), reason="needs /proc")
def test_a_node_past_its_budget_loses_its_processes_but_stays_open(env):
    import json

    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    nid = tree.new_node(kind="explore", title="a", hypothesis="h", minutes=30).split()[
        1
    ]
    job = subprocess.Popen(["sleep", "300"], cwd=root / "nodes" / nid)
    try:
        clock.advance(31)  # past the budget but inside the grace period
        assert tree.enforce_budgets() == []
        assert job.poll() is None
        clock.advance(2)  # now beyond budget plus grace
        assert tree.enforce_budgets() == [nid]
        assert job.wait(timeout=15) != 0
        assert tree.enforce_budgets() == []  # once per node
        assert "its processes were stopped" in tree.status()
        events = [
            json.loads(x)
            for x in (root / ".research" / "events.jsonl").read_text().splitlines()
        ]
        timeout = next(e for e in events if e["event"] == "node-timeout")
        assert timeout["node"] == nid
        assert timeout["killed"] == 1
        # the node is still open: the agent closes it with what it has
        (root / "nodes" / nid / "holdout_predictions.csv").write_text(
            accuracy_rows(200)
        )
        (root / "nodes" / nid / "submission.csv").write_text("d")
        assert "closed as done" in tree.finish(nid, change="c", observation="o")
    finally:
        job.kill()


# ---- the whole run can be replayed from what is saved ------------------------


def test_the_brief_handed_to_the_sub_agent_is_kept(env):
    tree, root, _clock, _ = env
    tree.init(budget_minutes=360)
    out = tree.new_node(kind="explore", title="a", hypothesis="h", plan="build it")
    saved = (root / ".research" / "briefs" / "n1.md").read_text()
    assert saved.startswith("## Brief for the sub-agent")
    assert saved in out


def test_a_closed_node_keeps_its_logs_and_a_manifest_of_what_it_left(env):
    import json

    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    nid = tree.new_node(kind="explore", title="a", hypothesis="h").split()[1]
    node_dir = root / "nodes" / nid
    (node_dir / "holdout_predictions.csv").write_text(accuracy_rows(200))
    (node_dir / "submission.csv").write_text("d")
    (node_dir / "outputs").mkdir()
    (node_dir / "outputs" / "train.log").write_text(
        "epoch 1 loss 0.9\nepoch 2 loss 0.5\n"
    )
    (node_dir / "outputs" / "model.bin").write_bytes(b"w" * 5000)
    clock.advance(10)
    tree.finish(nid, change="c", observation="o")
    art = root / ".research" / "artifacts" / nid
    assert (art / "traces" / "outputs" / "train.log").read_text().endswith("loss 0.5\n")
    assert not (
        art / "traces" / "outputs" / "model.bin"
    ).exists()  # weights are listed, not copied
    files = {
        f["path"]: f for f in json.loads((art / "files.json").read_text())["files"]
    }
    assert files["outputs/train.log"]["kept"] == "full"
    assert files["outputs/model.bin"]["bytes"] == 5000
    assert len(files["outputs/model.bin"]["sha256"]) == 64
    assert not any(
        p.startswith("data/") for p in files
    )  # the shared data link is not walked


def test_every_event_carries_an_absolute_timestamp(env):
    import json

    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    close(tree, root, clock, 200, kind="explore", title="a", hypothesis="h")
    events = [
        json.loads(x)
        for x in (root / ".research" / "events.jsonl").read_text().splitlines()
    ]
    assert all(isinstance(e["ts"], float) for e in events)
    stamps = [e["ts"] for e in events]
    assert stamps == sorted(stamps)


def test_finalize_exports_the_record_as_one_json_file(env):
    import json

    tree, root, clock, _ = env
    tree.init(budget_minutes=360)
    tree.survey([{"name": "abstain", "mechanism": "calibrate a threshold"}])
    n1, _ = close(
        tree,
        root,
        clock,
        200,
        kind="explore",
        title="a",
        hypothesis="h",
        cluster="abstain",
    )
    tree.insight(text="a is a floor", evidence=[n1])
    tree.finalize()
    doc = json.loads((root / ".research" / "trajectory.json").read_text())
    assert doc["champion"] == n1
    assert doc["survey"][0]["name"] == "abstain"
    assert doc["insights"][0]["id"] == "I1"
    node = doc["nodes"][0]
    assert node["event"]["trailers"]["Score"] == "0.5000"
    assert node["event"]["narrative"]["Observation"] == "o"
    assert node["brief"] == f"briefs/{n1}.md"
    assert (root / ".research" / node["brief"]).is_file()
