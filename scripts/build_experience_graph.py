#!/usr/bin/env python3
"""Build or inspect a sidecar graph without modifying the experience bank."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Imports follow the repository path setup for direct script execution.
from EvoScientist.memory._atomic import atomic_write_json  # noqa: E402, I001
from EvoScientist.memory.experiences.graph import (  # noqa: E402
    build_experience_graph,
    experience_graph_path,
    expand_graph,
    graph_is_current,
)
from EvoScientist.memory.experiences.store import list_experience_documents  # noqa: E402
from EvoScientist.memory.search import search_documents  # noqa: E402
from EvoScientist.memory.types import ObservationSearchMode  # noqa: E402


class SuppliedResponses:
    """Feed decisions from the current assistant through the normal validator."""

    def __init__(self, path):
        from langchain_core.messages import AIMessage

        self._message_type = AIMessage
        rows = json.loads(path.read_text(encoding="utf-8"))
        self._responses = {tuple(sorted(row["ids"])): row["result"] for row in rows}
        if len(self._responses) != len(rows):
            raise ValueError("duplicate candidate pair in supplied responses")

    async def ainvoke(self, messages):
        records = json.loads(messages[-1].content)
        key = tuple(sorted(record["id"] for record in records))
        if key not in self._responses:
            raise ValueError(f"no supplied response for {key}")
        return self._message_type(
            content=json.dumps(self._responses[key], ensure_ascii=False)
        )


async def run(args):
    if args.query:
        graph = json.loads(args.output.read_text(encoding="utf-8"))
        if graph["project_id"] != args.project:
            raise SystemExit("graph project differs from requested project")
        if not graph_is_current(graph, memory_dir=args.bank):
            raise SystemExit(
                "experience records changed; rebuild this graph before searching"
            )
        documents = list_experience_documents(
            memory_dir=args.bank, project_id=args.project
        )
        hits = search_documents(
            documents=[d for d in documents if d.observation_id in graph["records"]],
            query=args.query,
            limit=args.limit,
            mode=ObservationSearchMode.RANKED,
        )
        ids = [hit["observation_id"] for hit in hits]
        internal = [e for e in graph["edges"] if e["from"] in ids and e["to"] in ids]
        print(
            json.dumps(
                {
                    "hits": hits,
                    "internal_edges": internal,
                    "expanded": expand_graph(
                        graph,
                        ids,
                        limit=args.related_limit,
                        relations=set(args.relation) if args.relation else None,
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    prompt = (ROOT / "prompt/experience_linker.md").read_text(encoding="utf-8")
    previous = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else None
    )
    model = None
    classifier_id = ""
    if args.responses:
        model = SuppliedResponses(args.responses)
        classifier_id = (
            "codex-supplied-decisions:"
            + hashlib.sha256(args.responses.read_bytes()).hexdigest()
        )
    elif args.classify:
        if args.model:
            from EvoScientist.config.settings import get_effective_config
            from EvoScientist.llm.models import get_chat_model

            get_effective_config()
            model = get_chat_model(args.model, provider=args.provider)
            classifier_id = f"{args.provider or 'default'}:{args.model}"
        else:
            from EvoScientist.EvoScientist import _ensure_auxiliary_chat_model

            model = _ensure_auxiliary_chat_model()
    graph = await build_experience_graph(
        memory_dir=args.bank,
        project_id=args.project,
        prompt=prompt,
        model=model,
        papers=set(args.paper) if args.paper else None,
        neighbors=args.neighbors,
        recall_pool=args.recall_pool,
        min_similarity=args.min_similarity,
        min_shared_terms=args.min_shared_terms,
        max_document_frequency=args.max_document_frequency,
        diversity_ratio=args.diversity_ratio,
        candidate_strategy=args.candidate_strategy,
        candidate_routes=set(args.candidate_route) if args.candidate_route else None,
        max_pairs=None if args.max_pairs == 0 else args.max_pairs,
        previous=previous,
        classifier_id=classifier_id,
    )
    atomic_write_json(args.output, graph)
    routing_index_output = None
    if args.candidate_strategy == "field":
        routing_index_output = args.output.with_name(
            args.output.stem + ".routing-index.json"
        )
        atomic_write_json(
            routing_index_output,
            {
                "project_id": args.project,
                **graph["candidate_state"]["routing_index"],
            },
        )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "routing_index_output": str(routing_index_output)
                if routing_index_output
                else None,
                "records": len(graph["records"]),
                "derived_edges": sum(
                    e["relation"] == "derived_from" for e in graph["edges"]
                ),
                "proposed_edges": sum(
                    e["status"] == "unreviewed" for e in graph["edges"]
                ),
                "pairs": len(graph["decisions"]),
                "candidate_stats": graph["candidate_stats"],
                "classified": sum("result" in d for d in graph["decisions"]),
                "errors": [d for d in graph["decisions"] if "error" in d],
                "warnings": graph["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--output", type=Path, help="Defaults to the project's standard graph location"
    )
    parser.add_argument(
        "--paper", action="append", help="Restrict to this paper ID; repeatable"
    )
    classifiers = parser.add_mutually_exclusive_group()
    classifiers.add_argument(
        "--classify",
        action="store_true",
        help="Ask the model to classify candidate pairs",
    )
    classifiers.add_argument(
        "--responses", type=Path, help="Validate supplied LLM decisions"
    )
    parser.add_argument("--model")
    parser.add_argument("--provider")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=3,
        help="Total nominated neighbors per record, across all levels and papers",
    )
    parser.add_argument("--recall-pool", type=int, default=20)
    parser.add_argument(
        "--candidate-strategy", choices=["lexical", "field"], default="lexical"
    )
    parser.add_argument(
        "--candidate-route",
        action="append",
        help="Enable only these field routes; repeatable for ablation",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.12,
        help="Minimum normalized lexical similarity; not a relation probability",
    )
    parser.add_argument("--min-shared-terms", type=int, default=2)
    parser.add_argument("--max-document-frequency", type=float, default=0.6)
    parser.add_argument(
        "--diversity-ratio",
        type=float,
        default=0.8,
        help="Minimum cross-paper score relative to the replaced nomination",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional total cap; default 0 relies on per-record nomination limits",
    )
    parser.add_argument("--query", help="Search the bank and inspect graph neighbors")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--relation", action="append", help="Expand this relation type; repeatable"
    )
    parser.add_argument("--related-limit", type=int, default=4)
    args = parser.parse_args()
    if args.output is None:
        args.output = experience_graph_path(args.bank, args.project)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
