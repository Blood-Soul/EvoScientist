"""The frozen judge, reached over HTTP.

The judge scores a node's held-out predictions. It runs in its own process,
outside the agent's reach, and keeps the labels and the scoring code to
itself; the research-tree tools only hand it files and read back numbers.
That is what makes a score in the record a fact rather than a claim.

Two ways to reach it, same reply format. A judge service over HTTP (set
`EVOSCIENTIST_TREE_JUDGE_URL`), or a scoring script run as a local command
(set `EVOSCIENTIST_TREE_JUDGE_COMMAND`, see :class:`CommandJudge`).

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

import hashlib
import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

JUDGE_URL_ENV = "EVOSCIENTIST_TREE_JUDGE_URL"
JUDGE_COMMAND_ENV = "EVOSCIENTIST_TREE_JUDGE_COMMAND"
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
        return parse_reply(out)


def parse_reply(out: object) -> dict:
    """Validate a judge reply (shared by every judge kind) and normalise it."""
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


class CommandJudge:
    """A scoring script run as a local command, for a user who has no judge service.

    `<command> <prediction file> [<deliverable file>]` prints the same JSON as
    the HTTP protocol on stdout. When the reply carries no `eval_sha`, the hash
    of the script files named in the command stands in for it, so editing the
    scorer mid-run is still caught. Keep the script and its labels where the
    agent's file tools cannot reach them; the hash only detects edits, it does
    not prevent them.
    """

    def __init__(self, command: list[str], timeout: float = TIMEOUT_SECONDS):
        self.command = command
        self.timeout = timeout

    def files_sha(self) -> str:
        h = hashlib.sha256()
        for part in self.command:
            p = Path(part)
            if p.is_file():
                h.update(part.encode())
                h.update(p.read_bytes())
        return h.hexdigest()

    def score(self, prediction: Path, deliverable: Path | None) -> dict:
        argv = [*self.command, str(prediction)]
        if deliverable and Path(deliverable).is_file():
            argv.append(str(deliverable))
        try:
            run = subprocess.run(
                argv, capture_output=True, text=True, timeout=self.timeout
            )
            out = json.loads(run.stdout.strip().splitlines()[-1])
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
            return {"error": f"judge command failed: {type(e).__name__}: {e}"}
        res = parse_reply(out)
        if "error" not in res and not res["eval_sha"]:
            res["eval_sha"] = self.files_sha()
        return res


def judge_from_env() -> Judge | None:
    url = os.environ.get(JUDGE_URL_ENV, "").strip()
    if url:
        return HttpJudge(url)
    command = shlex.split(os.environ.get(JUDGE_COMMAND_ENV, ""))
    return CommandJudge(command) if command else None
