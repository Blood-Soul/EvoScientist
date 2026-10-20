"""Freeze matched neutral/task extraction requests without generating responses.

No model API is used. The only first-stage input difference is the task block.
Historical responses are references, never substituted for a new neutral arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from EvoScientist.memory._atomic import atomic_write_json  # noqa: E402
from EvoScientist.memory.experiences.extraction import (  # noqa: E402
    INDUCTION_PROMPT_KEY,
    build_extraction_message,
    load_experience_prompts,
)
from EvoScientist.memory.experiences.task_context import TaskContext  # noqa: E402


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare(manifest_path, context_path, output):
    context = TaskContext.from_json_file(context_path)
    prompts = load_experience_prompts()
    assert set(prompts) == {"l1", "l2", INDUCTION_PROMPT_KEY}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, paper in enumerate(manifest):
        text = Path(paper["text_path"]).read_text(encoding="utf-8")
        assert digest(text) == paper["paper_sha256"]
        neutral = build_extraction_message(paper_id=paper["paper_id"], paper_text=text)
        task = build_extraction_message(
            paper_id=paper["paper_id"], paper_text=text, task_context=context
        )
        header = f"[paper_id] {paper['paper_id']}\n\n"
        assert task == header + context.render() + "\n\n" + text
        assert neutral == header + text
        # Alternate preparation order; this does not create isolated model sessions.
        order = ("neutral", "task") if index % 2 == 0 else ("task", "neutral")
        for arm in order:
            directory = output / arm
            directory.mkdir(parents=True, exist_ok=True)
            for level in ("l1", "l2"):
                atomic_write_json(
                    directory / f"{paper['paper_id']}.{level}.request.json",
                    {
                        "system": prompts[level],
                        "human": task if arm == "task" else neutral,
                        "paper_sha256": digest(text),
                        "arm": arm,
                        "execution_status": "prepared_not_extracted",
                    },
                )
            # Induction must use this arm's own L1, not a shared historical response.
            (directory / "l2_induced.system.md").write_text(
                prompts[INDUCTION_PROMPT_KEY], encoding="utf-8"
            )
        rows.append(
            {
                "paper_id": paper["paper_id"],
                "paper_sha256": digest(text),
                "first_stage_only_difference": "task_context_block",
                "preparation_order": order,
                "neutral_human_sha256": digest(neutral),
                "task_human_sha256": digest(task),
            }
        )
    audit = {
        "status": "inputs_prepared_no_extraction_results",
        "external_model_calls": 0,
        "paper_count": len(rows),
        "first_stage_request_count": len(rows) * 4,
        "same_system_prompts": True,
        "strict_overlay_used": False,
        "prompt_sha256": {level: digest(prompt) for level, prompt in prompts.items()},
        "task_context_sha256": digest(context.render()),
        "papers": rows,
        "induction_rule": "Use each arm's own L1; same system prompt and gates; skip when L1 is empty.",
        "generation_controls": "Same model revision, sampling, output budget and effort; no extra pruning or merging; no forced equal counts.",
        "historical_bank_role": "external historical reference, not the new neutral control",
        "current_session_limit": "A context-aware assistant is not a blinded neutral model execution.",
    }
    atomic_write_json(output / "input-audit.json", audit)
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k not in ("papers", "prompt_sha256")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.task_context, args.output)
