"""The frozen judge, reached over HTTP.

The judge scores a node's held-out predictions. It runs in its own process,
outside the agent's reach, and keeps the labels and the scoring code to
itself; the research-tree tools only hand it files and read back numbers.
That is what makes a score in the record a fact rather than a claim.

Protocol (JSON over HTTP, one endpoint):

    POST $EVOSCIENTIST_TREE_JUDGE_URL
    {"prediction": "<file text>", "deliverable": "<file text>" | null}

    200 {"score": float, "per_sample": [float, ...], "slices": [str, ...],
         "eval_sha": "<hash of the frozen judge>"}
    200 {"error": "<why the files cannot be scored>"}

`slices` is optional (a label per row, for error analysis). Any other
failure (connection refused, non-JSON) is reported as an error, never as a
score.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

JUDGE_URL_ENV = "EVOSCIENTIST_TREE_JUDGE_URL"
TIMEOUT_SECONDS = 600


class Judge(Protocol):
    def score(self, prediction: Path, deliverable: Path | None) -> dict:
        """Return {"score", "per_sample", "slices", "eval_sha"} or {"error"}."""


class HttpJudge:
    def __init__(self, url: str, timeout: float = TIMEOUT_SECONDS):
        self.url = url
        self.timeout = timeout

    def score(self, prediction: Path, deliverable: Path | None) -> dict:
        body = {
            "prediction": Path(prediction).read_text(
                encoding="utf-8", errors="replace"
            ),
            "deliverable": (
                Path(deliverable).read_text(encoding="utf-8", errors="replace")
                if deliverable and Path(deliverable).is_file()
                else None
            ),
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                out = json.loads(response.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as e:
            return {
                "error": f"judge unreachable or invalid reply: {type(e).__name__}: {e}"
            }
        if not isinstance(out, dict):
            return {"error": "judge reply is not an object"}
        if out.get("error"):
            return {"error": str(out["error"])}
        per_sample = out.get("per_sample")
        if (
            not isinstance(out.get("score"), (int, float))
            or not isinstance(per_sample, list)
            or not per_sample
        ):
            return {"error": "judge reply lacks score / per_sample"}
        return {
            "score": float(out["score"]),
            "per_sample": [float(x) for x in per_sample],
            "slices": [str(x) for x in out.get("slices") or []],
            "eval_sha": str(out.get("eval_sha") or ""),
        }


def judge_from_env() -> Judge | None:
    url = os.environ.get(JUDGE_URL_ENV, "").strip()
    return HttpJudge(url) if url else None
