"""Candidate scale, isolation, diversity and incremental invalidation."""

from EvoScientist.memory.experiences.candidates import nominate_candidates


def record(key, text, paper=None):
    return {
        "id": key,
        "paper_key": paper or key,
        "level": "l1",
        "content": {"action": text},
    }


def test_shared_topic_does_not_force_links():
    records = {
        "a": record("a", "solar forecasting model"),
        "b": record("b", "solar forecasting dataset"),
        "c": record("c", "protein sequence alignment"),
    }
    pairs, state, stats = nominate_candidates(records)
    assert pairs == {}
    assert stats["covered_nodes"] == 0
    assert all(not values for values in state["nominations"].values())


def test_linear_nomination_bound_and_stable_rebuild():
    records = {}
    for group in range(20):
        for index in range(5):
            key = f"{group}-{index}"
            records[key] = record(
                key, f"feature{group} validation{group} technique{group}"
            )
    pairs, state, stats = nominate_candidates(records)
    assert 0 < len(pairs) <= len(records) * 3
    assert stats["covered_nodes"] == 100
    assert all(len(values) <= 3 for values in state["nominations"].values())
    repeated, _, again = nominate_candidates(records, previous=state)
    assert repeated == pairs
    assert again["sources_searched"] == 0


def test_cross_paper_replacement_is_conditional_and_does_not_add_slots():
    records = {
        "a": record("a", "interval calibration reserve sizing", "one"),
        "b": record("b", "interval calibration reserve sizing", "one"),
        "c": record("c", "interval calibration reserve sizing", "two"),
    }
    _, state, _ = nominate_candidates(records, neighbors=1)
    assert state["nominations"]["a"] == ["c"]
    records["c"] = record(
        "c",
        "interval calibration protein sequences enzyme substrates kinetics structure",
        "two",
    )
    _, state, _ = nominate_candidates(records, neighbors=1)
    assert state["nominations"]["a"] == ["b"]


def test_changed_or_removed_neighbor_refills_but_new_nodes_do_not_rerank_every_source():
    records = {key: record(key, "interval calibration reserve sizing") for key in "abc"}
    _, state, _ = nominate_candidates(records, neighbors=1)
    records["d"] = record("d", "interval calibration reserve sizing")
    _, updated, stats = nominate_candidates(records, neighbors=1, previous=state)
    assert stats["sources_searched"] == 1
    assert updated["nominations"]["a"] == state["nominations"]["a"]
    del records["b"]
    pairs, final, stats = nominate_candidates(records, neighbors=1, previous=updated)
    assert all("b" not in pair for pair in pairs)
    assert stats["sources_searched"] >= 1
    assert all(len(values) <= 1 for values in final["nominations"].values())


def test_policy_change_invalidates_nomination_cache():
    records = {key: record(key, "interval calibration reserve sizing") for key in "abc"}
    _, state, _ = nominate_candidates(records)
    _, _, stats = nominate_candidates(records, min_similarity=0.3, previous=state)
    assert stats["sources_searched"] == 3
