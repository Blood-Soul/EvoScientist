"""Offline lexical/field candidate comparison and ablation; never calls an LLM."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from EvoScientist.memory._atomic import atomic_write_json  # noqa: E402
from EvoScientist.memory.experiences.candidates import nominate_candidates  # noqa: E402
from EvoScientist.memory.experiences.field_candidates import (  # noqa: E402
    HYPOTHESES,
    nominate_field_candidates,
)
from EvoScientist.memory.experiences.graph import _record_documents  # noqa: E402
from EvoScientist.memory.experiences.routing import strings  # noqa: E402


def check_literal_citations(records, pairs, state):
    """Audit index/route excerpts against record fields, not against PDFs."""
    checked = 0
    for evidence in pairs.values():
        for match in evidence.get("matches", []):
            for source in match.get("field_matches", []):
                if not source or not any(
                    source["quote"] in text
                    for text in strings(
                        records[source["id"]]["content"][source["field"]]
                    )
                ):
                    raise ValueError("candidate has an invalid field citation")
                checked += 1
    for key, entry in state.get("routing_index", {}).get("entries", {}).items():
        for facets in entry["facets"].values():
            for facet in facets:
                for basis in facet["basis"]:
                    if not any(
                        basis["quote"] in text
                        for text in strings(records[key]["content"][basis["field"]])
                    ):
                        raise ValueError("facet has an invalid field citation")
                    checked += 1
    return checked


def run(args):
    records = _record_documents(args.bank, args.project, None)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    known = []
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if reference.get("project_id") != args.project:
            raise ValueError("reference belongs to another project")
        known = [
            tuple(row["ids"])
            for row in reference["decisions"]
            if row.get("result", {}).get("decision") == "related"
        ]
    states = {}
    for k in args.neighbors:
        baseline, _, _ = nominate_candidates(records, neighbors=k)
        routes = set(HYPOTHESES)
        for name, enabled in (
            ("lexical", None),
            ("field", routes),
            ("no_boundary", routes - {"boundary"}),
            ("no_transfer", routes - {"transfer"}),
            ("no_prerequisite", routes - {"prerequisite"}),
        ):
            started = time.perf_counter()
            if name == "lexical":
                pairs, state, stats = nominate_candidates(records, neighbors=k)
            else:
                pairs, state, stats = nominate_field_candidates(
                    records, neighbors=k, enabled_routes=enabled
                )
            stats["elapsed_seconds"] = round(time.perf_counter() - started, 4)
            stats["literal_citations_checked"] = check_literal_citations(
                records, pairs, state
            )
            assert len(pairs) <= len(records) * k
            summary = {
                "arm": name,
                "neighbors": k,
                **stats,
                "added_vs_lexical": len(set(pairs) - set(baseline)),
                "removed_vs_lexical": len(set(baseline) - set(pairs)),
                "prior_positive_sample_retained": sum(pair in pairs for pair in known),
                "prior_positive_sample_total": len(known),
            }
            pair_budget = min(len(pairs), len(baseline))
            capped = set(
                sorted(pairs, key=lambda pair: (-pairs[pair]["similarity"], pair))[
                    :pair_budget
                ]
            )
            capped_baseline = set(
                sorted(
                    baseline, key=lambda pair: (-baseline[pair]["similarity"], pair)
                )[:pair_budget]
            )
            summary["equal_pair_budget"] = {
                "pairs_per_arm": pair_budget,
                "added_vs_lexical": len(capped - capped_baseline),
                "removed_vs_lexical": len(capped_baseline - capped),
                "prior_positive_sample_retained": sum(pair in capped for pair in known),
                "covered_nodes": len({key for pair in capped for key in pair}),
            }
            summaries.append(summary)
            atomic_write_json(
                args.output / f"{name}-k{k}.json",
                {
                    "project_id": args.project,
                    "summary": summary,
                    "state": state,
                    "candidates": [
                        {"ids": list(pair), **evidence}
                        for pair, evidence in sorted(pairs.items())
                    ],
                },
            )
            if name == "field":
                states[k] = state
    k = args.neighbors[0]
    _, _, repeated = nominate_field_candidates(
        records, neighbors=k, previous=json.loads(json.dumps(states[k]))
    )
    changed = copy.deepcopy(records)
    first_id = sorted(changed)[0]
    changed[first_id]["content"]["action"] = (
        changed[first_id]["content"].get("action", "")
        + " Additional calibration check."
    )
    _, _, modified = nominate_field_candidates(changed, neighbors=k, previous=states[k])
    assert modified["index"]["entries_built"] == 1
    assert repeated["sources_searched"] == 0
    facets = Counter(
        f"{axis}:{facet['value']}"
        for entry in states[k]["routing_index"]["entries"].values()
        for axis, values in entry["facets"].items()
        for facet in values
    )
    result = {
        "project_id": args.project,
        "records": len(records),
        "llm_calls": 0,
        "settings": {
            "neighbors": args.neighbors,
            "recall_pool": 20,
            "min_similarity": 0.12,
            "min_shared_terms": 2,
            "max_document_frequency": 0.6,
        },
        "arms": summaries,
        "identical_rebuild": repeated,
        "one_modified_record": modified,
        "rule_facet_counts": dict(facets),
    }
    atomic_write_json(args.output / "summary.json", result)
    print(
        json.dumps(
            {
                "records": len(records),
                "arms": [
                    {
                        key: row[key]
                        for key in (
                            "arm",
                            "neighbors",
                            "nominated_pairs",
                            "covered_nodes",
                            "added_vs_lexical",
                            "removed_vs_lexical",
                            "prior_positive_sample_retained",
                        )
                    }
                    for row in summaries
                ],
                "identical_rebuild": repeated,
                "one_modified_record": modified,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--neighbors", nargs="+", type=int, default=[3, 5, 8])
    run(parser.parse_args())
