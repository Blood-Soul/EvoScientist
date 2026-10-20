"""Field-role and facet invariants before candidate/graph integration."""

import copy
import json

from EvoScientist.memory.experiences.field_candidates import nominate_field_candidates
from EvoScientist.memory.experiences.routing import build_routing_index, strings


def record(key, **content):
    return {"id": key, "paper_key": key, "level": "l2", "content": content}


def test_views_preserve_polarity_and_absent_fields():
    records = {
        "a": record(
            "a",
            applicable_when=["calibrated intervals"],
            not_applicable_when=["uncalibrated intervals"],
            rationale=None,
        )
    }
    original = copy.deepcopy(records)
    entry = build_routing_index(records)["entries"]["a"]
    assert (
        entry["field_views"]["positive"]["sources"][0]["quote"]
        == "calibrated intervals"
    )
    assert (
        entry["field_views"]["negative"]["sources"][0]["quote"]
        == "uncalibrated intervals"
    )
    assert entry["field_views"]["transfer"]["sources"] == []
    assert entry["field_views"]["outcome"]["sources"] == []
    assert records == original


def test_facets_have_literal_basis_and_are_not_causal_claims():
    records = {
        "a": record(
            "a",
            task="model selection",
            action="Use forward-chaining validation to select a model",
            trigger_context="short history",
            discipline="CS",
            domain="Time Series-Forecasting",
        )
    }
    entry = build_routing_index(records)["entries"]["a"]
    assert entry["domain"] == "time_series_forecasting"
    for axis in entry["facets"].values():
        for facet in axis:
            assert facet["origin"] == "rule_mention"
            for basis in facet["basis"]:
                assert any(
                    basis["quote"] in text
                    for text in strings(records["a"]["content"][basis["field"]])
                )
    assert any(
        item["value"] == "short_history" for item in entry["facets"]["problem_patterns"]
    )


def test_index_reuses_only_unchanged_entries_and_does_not_invent_unknown_roles():
    records = {
        "a": record("a", action="opaque unfamiliar term"),
        "b": record("b", action="use latent factors"),
    }
    first = build_routing_index(records)
    assert first["entries"]["a"]["unknown_axes"]
    assert build_routing_index(records, first)["stats"]["entries_reused"] == 2
    records["b"]["content"]["action"] = "different action"
    assert build_routing_index(records, first)["stats"]["entries_built"] == 1


def test_feedback_view_does_not_turn_action_order_into_dependency():
    records = {
        "a": record(
            "a",
            practice_trace=[{"action": "fit model", "feedback": "calibration failed"}],
        )
    }
    entry = build_routing_index(records)["entries"]["a"]
    assert entry["field_views"]["outcome"]["sources"] == [
        {"field": "practice_trace", "quote": "calibration failed"}
    ]
    assert "depends_on" not in str(entry)


def test_validation_calibration_is_not_mistagged_as_probabilistic_calibration():
    records = {
        "a": record(
            "a",
            action="Generate rolling-origin calibration forecasts and select models using validation errors",
        ),
        "b": record(
            "b", action="Calibrate the predictive distribution and size reserves"
        ),
    }
    entries = build_routing_index(records)["entries"]
    assert "uncertainty_calibration" not in {
        facet["value"] for facet in entries["a"]["facets"]["method_patterns"]
    }
    assert "uncertainty_calibration" in {
        facet["value"] for facet in entries["b"]["facets"]["method_patterns"]
    }


def test_boundary_and_prerequisite_match_different_roles_with_literal_reasons():
    records = {
        "a": record(
            "a",
            action="Use calibrated predictive intervals",
            effect="calibrated predictive distribution produced",
        ),
        "b": record(
            "b",
            not_applicable_when=["calibrated predictive intervals unavailable"],
            applicable_when=["calibrated predictive distribution available"],
        ),
    }
    pairs, _, _ = nominate_field_candidates(
        records, enabled_routes={"boundary", "prerequisite"}
    )
    assert set(pairs[("a", "b")]["routes"]) == {"boundary", "prerequisite"}
    for match in pairs[("a", "b")]["matches"]:
        assert match["condition_state"] == "unknown"
        for source in match["field_matches"]:
            assert any(
                source["quote"] in text
                for text in strings(records[source["id"]]["content"][source["field"]])
            )
    assert pairs[("a", "b")]["status"] == "pending"


def test_cross_label_route_needs_structure_and_known_distinct_domains():
    records = {
        "a": record(
            "a",
            domain="energy",
            task="model selection",
            action="route difficult inputs",
            transferable_core="conditional escalation of complex examples",
        ),
        "b": record(
            "b",
            domain="medicine",
            task="model selection",
            action="routing difficult cases",
            transferable_core="triage cases requiring additional review",
        ),
        "c": record(
            "c",
            domain="materials",
            task="unrelated question",
            action="unfamiliar method",
        ),
    }
    pairs, _, _ = nominate_field_candidates(records, enabled_routes={"transfer"})
    assert set(pairs) == {("a", "b")}
    assert (
        pairs[("a", "b")]["matches"][0]["domain_difference"]["status"]
        == "label_difference_only"
    )
    records["b"]["content"]["domain"] = "energy"
    assert not nominate_field_candidates(records, enabled_routes={"transfer"})[0]
    records["b"]["content"].pop("domain")
    assert not nominate_field_candidates(records, enabled_routes={"transfer"})[0]


def test_all_routes_share_one_budget_and_roundtrip_cache_is_valid():
    records = {
        str(i): record(
            str(i),
            domain=f"domain{i % 3}",
            task="model selection",
            action="route difficult inputs",
            trigger_context="short history",
            transferable_core="selective conditional routing",
        )
        for i in range(30)
    }
    pairs, state, _ = nominate_field_candidates(records, neighbors=3)
    assert len(pairs) <= 30 * 3
    assert all(len(choices) <= 3 for choices in state["nominations"].values())
    repeated, _, stats = nominate_field_candidates(
        records, previous=json.loads(json.dumps(state))
    )
    assert repeated == pairs
    assert stats["sources_searched"] == 0
    records["0"]["content"]["action"] = "different action"
    _, _, updated = nominate_field_candidates(records, previous=state)
    assert updated["index"]["entries_built"] == 1
    assert updated["sources_searched"] == 30


def test_unknown_fields_do_not_force_a_nomination_and_routes_are_versioned():
    records = {key: record(key, action="opaque unfamiliar text") for key in "ab"}
    assert not nominate_field_candidates(
        records, enabled_routes={"boundary", "transfer"}
    )[0]
    _, state, _ = nominate_field_candidates(records, enabled_routes={"shared_method"})
    _, _, stats = nominate_field_candidates(
        records, previous=state, enabled_routes={"shared_problem"}
    )
    assert stats["sources_searched"] == 2
