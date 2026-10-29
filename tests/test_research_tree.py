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
    assert "next-step-text" not in brief
    assert "next-step-text" in full
    assert "commit:" in full
    with pytest.raises(TreeError, match="no finished node n9"):
        tree.history(["n9"])
