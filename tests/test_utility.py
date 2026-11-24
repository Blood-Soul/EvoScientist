"""Experience utility: S, A, U against a fixed reference, computed once per record and kept across library changes."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from EvoScientist.memory import embedding
from EvoScientist.memory.experiences import utility as U
from tests.test_embedding import FakeService


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class TestFormula:
    def _ref(self):
        # four tasks on four axes: every task pair has cosine 0
        return U.Reference(["t1", "t2", "t3", "t4"], np.eye(4), key="k")

    def test_specificity_is_the_share_of_task_pairs_below_the_best_cosine(self):
        ref = U.Reference(["a", "b", "c"], np.stack([_unit([1, 0]), _unit([1, 1]), _unit([0, 1])]), key="k")
        # task pairs: a-c 0, a-b 0.707, b-c 0.707
        out = ref.score(np.stack([_unit([1, 0.05]), _unit([1, -0.6])]))
        assert out["S"][0] == pytest.approx(1.0)          # c_max ~ 0.999 >= every pair
        # second record: cos to a 0.857, to b 0.243, to c -0.514 -> c_max 0.857 >= all three pairs
        assert out["S"][1] == pytest.approx(1.0)
        low = ref.score(np.stack([_unit([-1, -0.2])]))     # every cosine negative: c_max -0.196
        assert low["S"][0] == pytest.approx(0.0)           # below even the 0 of a-c
        mid = ref.score(np.stack([_unit([1, -2.5])]))      # cos to a 0.371: above a-c (0), below the 0.707 pairs
        assert mid["S"][0] == pytest.approx(1 / 3)
        assert out["nearest"][0] == 0

    def test_abstractness_is_the_effective_share_of_tasks(self):
        ref = self._ref()
        one = ref.score(np.stack([_unit([1, 0, 0, 0])]))
        even = ref.score(np.stack([_unit([1, 1, 1, 1])]))
        assert even["A"][0] == pytest.approx(1.0)         # equal cosines: every task counts
        assert one["A"][0] == pytest.approx(0.25, abs=1e-3)  # all mass on one task: the floor 1/N
        p = np.exp(np.array([1, 0, 0, 0]) / 0.1)
        p /= p.sum()
        assert one["A"][0] == pytest.approx(math.exp(-(p * np.log(p)).sum()) / 4)
        assert even["U"][0] == pytest.approx(even["S"][0] * even["A"][0])

    def test_the_reference_key_follows_the_tasks_and_the_model(self):
        tasks = [("a", "x"), ("b", "y")]
        assert U.reference_key(tasks, "m") == U.reference_key(list(tasks), "m")
        assert U.reference_key(tasks, "m") != U.reference_key(tasks, "m2")
        assert U.reference_key(tasks, "m") != U.reference_key([("a", "x"), ("b", "z")], "m")


@pytest.fixture
def service(monkeypatch):
    embedding._reset_for_tests()
    U._references.clear()
    svc = FakeService()
    monkeypatch.setenv(embedding.URL_ENV, svc.url)
    yield svc
    svc.close()
    embedding._reset_for_tests()
    U._references.clear()


TASKS = [("t-qa", "span extraction question answering"), ("t-abstain", "unanswerable questions abstain"),
         ("t-lr", "learning rate warmup schedules"), ("t-mol", "graph molecule property prediction")]


def _rec(problem):
    return {"problem": problem, "method": ["m"], "task": "t"}


class TestStore:
    def test_without_embeddings_nothing_is_scored(self, tmp_path, monkeypatch):
        monkeypatch.delenv(embedding.URL_ENV, raising=False)
        assert U.update_scores(tmp_path, "P", tasks=TASKS, records=[("E-1", "l1", _rec("span"))])["status"] == "no_embeddings"

    def test_each_record_is_scored_once_and_growth_does_not_rescore(self, service, tmp_path):
        first = [("E-1", "l1", _rec("span head")), ("E-2", "l2", _rec("warmup"))]
        out = U.update_scores(tmp_path, "P", tasks=TASKS, records=first)
        assert out["status"] == "ok" and out["scored"] == 2
        saved = U.load_scores(tmp_path, "P")
        assert set(saved["scores"]) == {"E-1", "E-2"} and saved["scores"]["E-1"]["nearest"] == "t-qa"
        assert {"S", "A", "U", "c_max", "nearest", "level"} <= set(saved["scores"]["E-2"])
        before = json.dumps(saved["scores"]["E-1"])
        calls = len(service.calls)
        grown = [*first, ("E-3", "l1", _rec("molecule graph"))]
        out = U.update_scores(tmp_path, "P", tasks=TASKS, records=grown)
        assert out["scored"] == 1 and out["total"] == 3
        assert json.dumps(U.load_scores(tmp_path, "P")["scores"]["E-1"]) == before   # untouched
        assert len(service.calls) == calls + 1                                         # only the new record sent

    def test_a_record_that_leaves_loses_its_score(self, service, tmp_path):
        U.update_scores(tmp_path, "P", tasks=TASKS, records=[("E-1", "l1", _rec("a")), ("E-2", "l1", _rec("b"))])
        out = U.update_scores(tmp_path, "P", tasks=TASKS, records=[("E-1", "l1", _rec("a"))])
        assert out["dropped"] == 1 and set(U.load_scores(tmp_path, "P")["scores"]) == {"E-1"}

    def test_a_new_reference_rescores_everything(self, service, tmp_path):
        recs = [("E-1", "l1", _rec("span"))]
        U.update_scores(tmp_path, "P", tasks=TASKS, records=recs)
        ref1 = U.load_scores(tmp_path, "P")["ref"]
        out = U.update_scores(tmp_path, "P", tasks=TASKS[:3], records=recs)
        assert out["scored"] == 1 and U.load_scores(tmp_path, "P")["ref"] != ref1

    def test_the_packaged_reference_loads(self):
        tasks = U.load_tasks()
        assert len(tasks) == 80 and all(tid and text for tid, text in tasks)
