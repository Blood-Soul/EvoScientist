"""Validate Codex-supplied extraction responses through the real local pipeline.

This is an offline response replay, NOT a call to Codex or another model API.
It preserves the historical bank and records input/prompt/response hashes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402

from EvoScientist.memory._atomic import atomic_write_json  # noqa: E402
from EvoScientist.memory.experiences.extraction import (  # noqa: E402
    INDUCTION_PROMPT_KEY,
    load_experience_prompts,
    run_experience_extraction,
)
from EvoScientist.memory.experiences.store import (  # noqa: E402
    load_paper_experiences,
    store_paper_experiences,
)
from EvoScientist.memory.experiences.task_context import TaskContext  # noqa: E402


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SuppliedExtractionResponses:
    def __init__(
        self, directory, prompts, *, expect_task_context=True, messages_dir=None
    ):
        self.directory = directory
        self.level_by_prompt = {prompt: level for level, prompt in prompts.items()}
        self.calls = []
        self.expect_task_context = expect_task_context
        self.messages_dir = messages_dir

    async def ainvoke(self, messages):
        level = self.level_by_prompt[messages[0].content]
        human = messages[-1].content
        paper = re.match(r"\[paper_id\] ([^\n]+)", human).group(1)
        path = self.directory / f"{paper}.{level}.json"
        response = path.read_text(encoding="utf-8")
        assert ("[task_context]" in human) == self.expect_task_context
        if self.messages_dir is not None:
            self.messages_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                self.messages_dir / f"{paper}.{level}.json",
                {
                    "system": messages[0].content,
                    "human": human,
                    "response_origin": "assistant_supplied_not_model_generated_by_replay",
                },
            )
        self.calls.append(
            {
                "paper_id": paper,
                "path": level,
                "human_sha256": sha(human),
                "system_sha256": sha(messages[0].content),
                "response_sha256": sha(response),
                "execution": "supplied_response_replay",
                "external_model_call": False,
            }
        )
        return AIMessage(content=response)


def quote_audit(payloads, text):
    counts = Counter()
    for payload in payloads.values():
        for item in payload["experiences"]:
            for row in item["evidence"]:
                quote = row["quote"]
                counts["quotes"] += 1
                counts["exact"] += quote in text
                counts["whitespace_normalized"] += " ".join(quote.split()) in " ".join(
                    text.split()
                )
    return dict(counts)


async def run(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    context = (
        TaskContext.from_json_file(args.task_context) if args.task_context else None
    )
    base_prompts = load_experience_prompts()
    assert INDUCTION_PROMPT_KEY in base_prompts
    overlay = args.overlay.read_text(encoding="utf-8") if args.overlay else ""
    prompts = {
        level: overlay + "\n\n" + prompt if overlay else prompt
        for level, prompt in base_prompts.items()
    }
    model = SuppliedExtractionResponses(
        args.responses,
        prompts,
        expect_task_context=context is not None,
        messages_dir=args.output / "actual-messages",
    )
    bank = args.output / ("bank-task" if context else "bank-neutral")
    snapshot = args.output / "bank-historical-snapshot"
    if args.baseline.resolve() in {bank.resolve(), snapshot.resolve()}:
        raise ValueError("output must not overwrite the baseline bank")
    baseline_root = args.baseline / "experiences/projects" / args.project
    args.output.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.output / "effective-prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for level, prompt in prompts.items():
        (prompt_dir / f"{level}.md").write_text(prompt, encoding="utf-8")
    rows = []
    for paper in manifest:
        paper_id = paper["paper_id"]
        text_path = Path(paper["text_path"])
        text = text_path.read_text(encoding="utf-8")
        candidates = [
            directory
            for directory in baseline_root.glob(paper_id + "-*")
            if directory.is_dir() and (directory / "metadata.json").is_file()
        ]
        if len(candidates) != 1:
            raise ValueError(f"ambiguous baseline paper: {paper_id}")
        source_dir = candidates[0]
        metadata = json.loads(
            (source_dir / "metadata.json").read_text(encoding="utf-8")
        )
        assert sha(text) == metadata["paper_sha256"] == paper["paper_sha256"]
        destination = snapshot / "experiences/projects" / args.project / source_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("l1.json", "l2.json", "metadata.json"):
            shutil.copy2(source_dir / filename, destination / filename)
        old = {
            level: json.loads(
                (source_dir / f"{level}.json").read_text(encoding="utf-8")
            )
            for level in ("l1", "l2")
        }
        payloads = await run_experience_extraction(
            paper_id=paper_id,
            paper_text=text,
            prompts=prompts,
            model=model,
            task_context=context,
        )
        expected_paths = {"l1", "l2"}
        if payloads["l1"]["experiences"]:
            expected_paths.add(INDUCTION_PROMPT_KEY)
        actual_paths = [
            call["path"] for call in model.calls if call["paper_id"] == paper_id
        ]
        assert set(actual_paths) == expected_paths
        assert len(actual_paths) == len(expected_paths)
        for payload in payloads.values():
            for item in payload["experiences"]:
                assert len(item["statement"].split()) >= 350
                assert len(item["trigger_context"].split()) <= 25
                assert len(item["transferable_core"].split()) <= 60
                assert all(len(row["quote"]) >= 150 for row in item["evidence"])
        audit = quote_audit(payloads, text)
        assert audit.get("quotes", 0) == audit.get("exact", 0)
        directory = store_paper_experiences(
            memory_dir=bank,
            project_id=args.project,
            paper_id=paper_id,
            url=metadata["url"],
            title=metadata["title"],
            paper_text=text,
            prompts={key: prompts[key] for key in ("l1", "l2")},
            payloads=payloads,
            domain_arxiv=metadata.get("domain_arxiv"),
        )
        assert (
            load_paper_experiences(
                memory_dir=bank,
                project_id=args.project,
                paper_id=paper_id,
                url=metadata["url"],
            )
            is not None
        )
        atomic_write_json(
            directory / "task-extraction-provenance.json",
            {
                "execution": "current_assistant_authored_then_replayed",
                "external_model_calls": 0,
                "task_context": context.to_dict() if context else None,
                "task_context_sha256": sha(context.render()) if context else None,
                "overlay_sha256": sha(overlay),
                "system_prompt_sha256": {
                    key: sha(value) for key, value in prompts.items()
                },
                "paper_sha256": sha(text),
                "responses": {
                    key: sha(
                        (args.responses / f"{paper_id}.{key}.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for key in prompts
                },
                "quote_audit": audit,
            },
        )
        rows.append(
            {
                "paper_id": paper_id,
                "source_hash_matches": True,
                "historical_counts": {
                    level: len(old[level]["experiences"]) for level in ("l1", "l2")
                },
                "task_counts": {
                    level: len(payloads[level]["experiences"]) for level in ("l1", "l2")
                },
                "historical_quote_audit": quote_audit(old, text),
                "task_quote_audit": audit,
            }
        )
    summary = {
        "execution": "supplied_response_replay",
        "external_model_calls": 0,
        "has_task_context": context is not None,
        "has_overlay": bool(overlay),
        "project_id": args.project,
        "papers": rows,
        "replay_calls": model.calls,
        "historical_records": sum(
            sum(row["historical_counts"].values()) for row in rows
        ),
        "task_records": sum(sum(row["task_counts"].values()) for row in rows),
        "caveat": "Historical versus current-assistant extraction; not a randomized prompt-only comparison.",
    }
    atomic_write_json(args.output / "validation.json", summary)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "replay_calls"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-context", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--project", default="P-solar")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
