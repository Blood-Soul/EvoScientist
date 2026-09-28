#!/usr/bin/env python3
"""Score experience records on survey Ch.7 intrinsic-quality dimensions, and
attribute each dimension's score to the fields it actually rested on.

Two things this is for:

1. Scoring records whose field sets differ. The corpus was extracted under
   several prompt versions, so a scorer that assumes a fixed schema cannot
   compare old records against new ones. Absent fields are classified before
   scoring: a field the record's own prompt version never asked for is *not
   evaluated* (it is not a defect of that record), while a field that version
   did ask for and the model failed to produce is scored as missing.

2. Measuring what each field contributes, so field changes are driven by
   evidence rather than intuition. Two methods, deliberately both:

   - `--method ablation` blanks one field at a time and re-scores. The drop is
     that field's contribution. Costs one call per field per record, and is the
     only method that sees cross-field dependencies (blank `evidence` and
     grounding collapses regardless of how good `statement` is).
   - `--method attribution` asks the scorer, in a single call, to name the
     fields each verdict rested on. Cheap enough for the full corpus, but it is
     the model reporting on itself.

   Run ablation on 20-30 records first to get a trustworthy baseline, then check
   whether attribution reproduces it. If it does, attribution scales; if not,
   attribution is not trustworthy here and only ablation counts.

Dimensions implemented are the ones survey Ch.7 can audit from the library's own
text: execution grounding (per-claim supported/contradicted/insufficient) and
applicability (is the stated scope wider than the evidence). Correctness,
actionability and coverage need an external reference the project does not have
-- see `docs/v4-explanatory/经验抽取迭代优化方案.md` section 2.2.1.

Scores are reported per dimension and never summed into one number: survey 7.1
is explicit that combining them hides which defect a figure describes. The
"total" used for ablation deltas is a diagnostic for field attribution only, not
a quality score for a record.

Usage:
    scripts/score_experiences.py --bank <dir> --project <id> --limit 8
    scripts/score_experiences.py --bank <dir> --project <id> --limit 24 \\
        --method ablation --out /tmp/ablation.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from langchain_core.messages import HumanMessage, SystemMessage

    from EvoScientist.utils import format_message_content
except ImportError as _exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"cannot import the experience layer ({_exc}); run from the repo root "
        "with its virtualenv, e.g. .venv/bin/python scripts/score_experiences.py"
    ) from _exc


# Fields the scorer may reason about, and the prompt version that introduced
# each. A record is only penalized for a field whose introducing version is at
# or below the record's own version -- otherwise every pre-v5 record would be
# marked defective for lacking fields its prompt never mentioned.
_FIELD_INTRODUCED_IN = {
    "domain": 1,
    "task": 1,
    "statement": 1,
    "applicable_when": 1,
    "not_applicable_when": 1,
    "scope": 1,
    "action": 1,
    "effect": 1,
    "evidence": 1,
    "practice_trace": 1,  # L1 only
    "claim_type": 1,  # L2 only
    "rationale": 1,  # L2 only
    "rationale_depth": 1,  # L2 only
    "discipline": 4,
    "transferable_core": 4,
    "bindings": 4,
    "trigger_context": 5,
    "evidence_scope": 5,
}

# Fields that legitimately carry null/empty as a *value*, not as an omission.
# `rationale: null` means the paper stated no mechanism, which is information;
# scoring it as a missing field would punish honest extraction.
_NULLABLE_FIELDS = {"rationale", "rationale_depth"}

# Fields whose absence the ablation method can blank. Identity/provenance fields
# are excluded: blanking them tests nothing about extraction quality.
_ABLATABLE_FIELDS = (
    "trigger_context",
    "statement",
    "applicable_when",
    "not_applicable_when",
    "scope",
    "action",
    "effect",
    "evidence",
    "evidence_scope",
    "practice_trace",
    "rationale",
    "transferable_core",
    "bindings",
)

_RUNTIME_INJECTED = {
    "id",
    "layer",
    "paper_id",
    "domain_arxiv",
    "utility",
    "confidence",
    "created_at",
    "source_l1_ids",
}


SCORER_PROMPT = """\
You audit one experience record extracted from a research paper. You see only \
the record. You do not see the paper.

That is deliberate. The record's own evidence quotes are the only source of \
support you may credit. If a claim is plausible but its quotes do not support \
it, that claim is unsupported -- reading as reasonable is not evidence. Judging \
the record against the paper would let the same material both make and check \
the claim.

## Dimensions

Score each independently. Do not let a low score on one pull down another, and \
do not produce an overall score.

### 1. execution_grounding

Enumerate claims by the following rule and no other. The rule fixes the \
denominator so two scorings of the same record are comparable; deviating from \
it changes the score without any change in record quality.

Emit exactly one claim for each of:

- `action` -- one claim.
- `effect` -- one claim.
- `rationale` -- one claim. Omit entirely when it is null.
- each entry of `applicable_when` -- one claim per array entry, in order. Never \
merge two entries into one claim, never split one entry into two, and never \
skip an entry because it looks obvious or because no quote addresses it.

Emit no other claims. In particular do not emit a claim for `statement`. \
`statement` is a long self-contained narrative that restates the same practice \
together with its problem, environment and boundaries; judging it as one \
atomic claim would make a record score lower purely for describing its context \
at greater length, and the checkable assertions it contains are already \
present in the fields above. Whether it reaches beyond its quotes is a \
separate question, asked in `statement_coverage` below. Also emit no claims for \
`scope`, `not_applicable_when`, or `transferable_core`, even if they assert \
something checkable. The claim count must equal: 1 (action) + 1 (effect) + 1 \
if rationale is non-null + len(applicable_when). Set `expected_claim_count` to \
that number and make `claims` exactly that long.

Judge each claim against the `evidence` quotes. For each emit exactly one of:
- `supported` -- a quote states it, or states something it follows from directly
- `contradicted` -- a quote states something incompatible with it
- `insufficient` -- no quote bears on it either way

An observed outcome, its diagnosis, and guidance drawn from it are separate \
claims with separate evidential demands. A quote supporting the outcome does \
not thereby support the diagnosis. Numerous supported observations must not be \
allowed to conceal an unsupported inference.

Report `score` as supported / (supported + contradicted + insufficient).

### 1b. statement_coverage

Asked separately from grounding, and never folded into it: "does this record \
assert things its quotes do not cover" is a different question from "is each of \
its checkable claims supported", and a long narrative answers the first badly \
while answering the second no differently from a short one.

Read `statement` and count the distinct factual assertions in it -- named \
results, numbers, datasets, models, and comparisons. Report:

- `assertions_total`: how many you counted.
- `assertions_uncovered`: how many of them no `evidence` quote bears on.
- `uncovered_examples`: up to three short verbatim fragments of the uncovered \
ones.

Count context, motivation, and restatements of the same assertion once or not \
at all: a record is not worse for explaining the situation it came from. Count \
only assertions a reader could act on being wrong about.

### 2. applicability

Is the stated scope wider than the evidence warrants? Over-inclusion is the \
failure that matters: an over-broad record gets retrieved for tasks it cannot \
serve.

Answer these ten checks and no others. Each is a yes/no judgment, where `true` \
means the over-reach is present. Judge each on its own criterion: do not weigh \
them against each other, do not let one answer pull another along, and do not \
emit an overall impression. The score is computed from your ten answers.

Ten rather than five so that one uncertain judgment moves the score by 0.1 \
instead of 0.2.

Scope breadth against what was tested:

- `beyond_datasets`: the claim is phrased as holding for datasets or data \
regimes beyond those in `evidence_scope.datasets`. A claim naming no dataset but \
implying a whole domain counts as `true` when only one dataset was tested.
- `beyond_models`: likewise for `evidence_scope.models` -- the claim is phrased \
as a property of a model class when one model was tested.
- `beyond_scale`: the claim implies it holds at scales (parameter count, data \
volume, sequence length, compute budget) other than the one tested, without \
having tested more than one.
- `beyond_metrics`: count the distinct named metrics appearing in the quotes and \
in `evidence_scope.metrics` (e.g. Dice, accuracy, BLEU, latency). This check is \
`true` only when that count is exactly one AND the claim asserts improvement or \
degradation without naming that metric -- i.e. it says "performs better" or \
"improves quality" rather than "improves Dice". If the claim names the single \
metric it was measured on, this is `false`: stating the metric is not \
over-reach. If two or more metrics appear, `false`.

Inference outrunning evidence:

- `mechanism_without_ablation`: the record asserts a causal mechanism (in \
`statement` or `rationale`) while `evidence_scope.has_ablation` is `false`. \
Attributing an effect to a specific cause requires isolating that cause.
- `superiority_without_baseline`: the claim asserts something is better than an \
alternative while `evidence_scope.has_baseline_comparison` is `false` and no \
quote reports an external comparison. Comparing only against the paper's own \
variants does not establish this.
- `direction_from_single_point`: the claim asserts a trend, monotonicity, or \
"more X gives more Y" while the quotes report observations at only one or two \
settings of X.

Conditions and boundaries:

- `untested_conditions`: one or more `applicable_when` entries name a setting \
that appears nowhere in `evidence_scope` and in no quote. The condition was \
asserted rather than observed.
- `boundary_absent`: `not_applicable_when` is empty, or every entry is generic \
filler ("when data is insufficient") rather than a condition traceable to \
something this paper actually observed failing.
- `universal_phrasing`: the claim is stated without hedging, as a general rule, \
while `verification_strength` is `single-setting`.

When `evidence_scope` is absent from the record, judge the checks that \
reference it from the `evidence` quotes alone, and say so in the note.

Report `score` as 1 - (checks true / 10).

### 3. completeness

For fields listed in `fields_expected` only, report any that are present but \
substantively defective: empty, placeholder, self-contradictory, or restating \
another field verbatim. Do not report a field absent from `fields_expected`, \
and do not report a merely terse field as defective -- brevity is not a defect.

List them in `field_defects`. The score is computed from that list plus the \
fields you were told are missing, over the count of `fields_expected`; you do \
not need to compute it.

### 4. actionability_proxy

Answer these five checks. `true` means the record ANSWERS that question -- note \
this is the opposite polarity from the applicability checks above.

- `says_what_to_do`: some field states a concrete action a reader could carry \
out, not merely an observation that something is the case.
- `says_when_it_applies`: a precondition is stated specifically enough to test \
against a new task, rather than as a restatement of the domain.
- `says_what_to_expect`: an outcome is stated, with direction at minimum and a \
magnitude where the source allows.
- `says_when_not_to`: at least one genuine failure or exclusion condition is \
given (not filler such as "when data is insufficient").
- `says_what_to_rebind`: the source-specific values a reader must replace for \
their own setting are identifiable, whether from `bindings` or named plainly in \
the text.

This is a PROXY. Real actionability requires a fixed executor's decision list, \
which is not available here; these five stand in for "does this answer the \
decisions a reader faces". Judge only what the record says, never whether the \
advice is good.

## Output

Return only this JSON object, no prose and no Markdown fences:

{
  "execution_grounding": {
    "score": 0.0,
    "expected_claim_count": 0,
    "claims": [
      {"claim_field": "action", "claim_index": 0, "verdict": "supported", "note": "brief"}
    ],
    "counts": {"supported": 0, "contradicted": 0, "insufficient": 0}
  },
  "statement_coverage": {
    "assertions_total": 0,
    "assertions_uncovered": 0,
    "uncovered_examples": ["brief verbatim fragment"]
  },
  "applicability": {
    "score": 0.0,
    "checks": {
      "beyond_datasets": false,
      "beyond_models": false,
      "beyond_scale": false,
      "beyond_metrics": false,
      "mechanism_without_ablation": false,
      "superiority_without_baseline": false,
      "direction_from_single_point": false,
      "untested_conditions": false,
      "boundary_absent": false,
      "universal_phrasing": false
    },
    "note": "brief"
  },
  "field_defects": [{"field": "scope", "defect": "brief"}],
  "actionability_proxy": {
    "checks": {
      "says_what_to_do": false,
      "says_when_it_applies": false,
      "says_what_to_expect": false,
      "says_when_not_to": false,
      "says_what_to_rebind": false
    },
    "note": "brief"
  }
}
"""

ATTRIBUTION_SUFFIX = """\

## Additional requirement for this run

For each dimension also report which fields its verdict actually rested on. \
Name only fields you genuinely used: if you judged grounding by reading \
`evidence` quotes against `statement`, list those two, not every field present.

Add to each dimension object:

  "relied_on": ["evidence", "statement"]
"""


def _record_version(item: dict[str, Any]) -> int:
    """Infer which prompt version produced a record, from the fields it carries.

    Reading the version off the record is more reliable than trusting a stored
    version marker: the corpus predates any such marker, and a record's fields
    are what the scorer must actually reason about.
    """
    if "trigger_context" in item or "evidence_scope" in item:
        return 5
    if {"discipline", "transferable_core", "bindings"} & set(item):
        return 4
    return 1


def _expected_fields(item: dict[str, Any], *, level: str) -> list[str]:
    """Fields this record's own prompt version asked for.

    A field absent here is not evaluated at all -- it is not a defect of a
    record whose prompt never requested it. This is what lets records from
    different prompt versions be compared on the same scale.
    """
    version = _record_version(item)
    level_only = {
        "practice_trace": "l1",
        "claim_type": "l2",
        "rationale": "l2",
        "rationale_depth": "l2",
    }
    expected = []
    for field, introduced in _FIELD_INTRODUCED_IN.items():
        if introduced > version:
            continue
        if level_only.get(field, level) != level:
            continue
        expected.append(field)
    return sorted(expected)


def _missing_fields(item: dict[str, Any], expected: list[str]) -> list[str]:
    """Expected fields the model failed to produce.

    Distinct from fields outside `expected`: these were asked for and are gone,
    so they are a real extraction defect. Nullable fields holding null are
    present, not missing -- `rationale: null` means the paper stated no
    mechanism, which is information rather than an omission.
    """
    missing = []
    for field in expected:
        if field not in item:
            missing.append(field)
        elif field not in _NULLABLE_FIELDS and item[field] in ("", [], {}, None):
            missing.append(field)
    return missing


def _scorable_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k not in _RUNTIME_INJECTED}


async def _ainvoke_with_retry(
    model: Any, messages: list[Any], *, attempts: int = 4
) -> Any:
    """Call the model, retrying transient connection failures.

    A scoring call sends the whole record and receives twenty checks plus notes,
    so it holds a connection long enough that the endpoint drops some of them
    under even modest concurrency -- while a single call to the same endpoint
    always succeeds. Retrying with a growing gap recovers those without lowering
    concurrency to one, which would make a full-corpus run impractical.
    """
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            return await model.ainvoke(messages)
        except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
            if attempt == attempts:
                raise
            label = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            print(
                f"    retry {attempt}/{attempts - 1} after {label} "
                f"(sleeping {delay:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


async def _score_once(
    item: dict[str, Any],
    *,
    level: str,
    expected: list[str],
    model: Any,
    attribute: bool,
) -> dict[str, Any]:
    prompt = SCORER_PROMPT + (ATTRIBUTION_SUFFIX if attribute else "")
    payload = {
        "level": level.upper(),
        "fields_expected": expected,
        "fields_missing": _missing_fields(item, expected),
        "record": _scorable_payload(item),
    }
    response = await _ainvoke_with_retry(
        model,
        [
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    raw = format_message_content(response).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].removeprefix("json").strip()
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"scorer returned no JSON object: {raw[:200]}")
    scores = json.JSONDecoder().raw_decode(raw[start:])[0]

    # Recompute grounding over the denominator the enumeration rule mandates,
    # rather than the one the scorer happened to use. Without this the same
    # record scores differently run to run purely from how it grouped the
    # `applicable_when` entries, and every ablation delta measures that drift
    # instead of the field's contribution.
    drift = _grouping_drift(item, scores)
    regrounded = _regrounded_score(item, scores)
    if isinstance(scores.get("execution_grounding"), dict):
        if regrounded is not None:
            scores["execution_grounding"]["score_as_reported"] = scores[
                "execution_grounding"
            ].get("score")
            scores["execution_grounding"]["score"] = regrounded
        if drift:
            scores["execution_grounding"]["grouping_drift"] = drift

    recounted = _recount_applicability(scores)
    if recounted is not None:
        scores["applicability"]["score_as_reported"] = scores["applicability"].get(
            "score"
        )
        scores["applicability"]["score"] = recounted

    # Same principle for the two added dimensions: the scorer supplies the
    # judgments, the code owns every denominator.
    actionability = _score_actionability(scores)
    if actionability is not None:
        scores["actionability_proxy"]["score"] = actionability
    completeness = _score_completeness(
        scores, expected=expected, missing=payload["fields_missing"]
    )
    if completeness is not None:
        scores["completeness"] = {
            "score": completeness,
            "expected": len(expected),
            "missing": payload["fields_missing"],
            "defective": [
                row.get("field")
                for row in scores.get("field_defects") or []
                if isinstance(row, dict)
            ],
        }
    return scores


def _expected_claim_count(item: dict[str, Any]) -> int:
    """Claim count the enumeration rule mandates for this record.

    Computed here rather than trusted from the scorer: the denominator is what
    makes two scorings comparable, so a scorer that silently merges or drops
    `applicable_when` entries would shift the score with no change in record
    quality. Mismatches are reported as `grouping_drift`.

    `statement` is deliberately not a claim. It is a >=350-word narrative that
    restates the practice with its problem, environment and boundaries, so
    scoring it as one atomic claim made any record lose that whole claim over a
    single uncovered aside -- penalising length rather than quality, which the
    survey's granularity rule forbids ("do not let a version that describes the
    same command in more detail accumulate more countable items"; the inverse is
    the same defect). Its checkable content is already carried by `action`,
    `effect`, `rationale` and `applicable_when`; whether it reaches past its
    quotes is reported separately as `statement_coverage`.
    """
    count = 2  # action + effect
    if item.get("rationale") is not None:
        count += 1
    conditions = item.get("applicable_when")
    if isinstance(conditions, list):
        count += len(conditions)
    return count


def _grouping_drift(item: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any] | None:
    """Report a claim count that departs from the mandated denominator."""
    block = scores.get("execution_grounding")
    if not isinstance(block, dict):
        return None
    mandated = _expected_claim_count(item)
    claims = block.get("claims")
    actual = len(claims) if isinstance(claims, list) else None
    if actual is None or actual == mandated:
        return None
    return {"mandated": mandated, "actual": actual}


def _regrounded_score(item: dict[str, Any], scores: dict[str, Any]) -> float | None:
    """Recompute grounding over the mandated denominator.

    Any claim the scorer failed to emit counts as `insufficient`: an unexamined
    condition is not a supported one, and letting the scorer shrink the
    denominator would reward records whose conditions it declined to check.
    """
    block = scores.get("execution_grounding")
    if not isinstance(block, dict) or not isinstance(block.get("claims"), list):
        return None
    mandated = _expected_claim_count(item)
    if mandated <= 0:
        return None
    supported = sum(
        1
        for claim in block["claims"]
        if isinstance(claim, dict) and claim.get("verdict") == "supported"
    )
    return round(min(supported, mandated) / mandated, 4)


def _reference_set(item: dict[str, Any]) -> str:
    """Which corpus this record's own quotes are drawn from.

    A directly extracted record quotes the paper, so grounding it measures
    fidelity to the paper. An induced record is given no paper text at all --
    `l2_induced_from_l1.md` requires every quote to be copied out of a
    contributing L1 record -- so the same procedure measures fidelity to those
    L1 records instead. Two different reference sets: the survey's first audit
    rule requires declaring them and forbids ranking records audited against
    different ones against each other, so the scores are reported in separate
    groups rather than pooled into one mean.
    """
    return "l1_records" if item.get("source_l1_ids") else "paper"


def _normalise_quote(text: str) -> str:
    """Collapse whitespace so a copied quote still matches after reflowing."""
    return " ".join(str(text or "").split())


def _quote_provenance(
    item: dict[str, Any], paper_dir: Path, level: str
) -> dict[str, Any] | None:
    """For an induced record, check each quote really was copied from an L1.

    The prompt requires verbatim copying rather than paraphrase, which makes
    this checkable in code -- no model judgment, and stricter than asking one
    whether a claim "is supported". It also confirms `source_l1_ids` resolves
    to L1 records that exist, which nothing else verifies.
    """
    source_ids = item.get("source_l1_ids")
    if not isinstance(source_ids, list) or not source_ids:
        return None
    l1_path = paper_dir / "l1.json"
    try:
        payload = json.loads(l1_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": f"cannot read {l1_path.name}"}

    by_id = {
        str(rec.get("id")): rec
        for rec in payload.get("experiences", [])
        if isinstance(rec, dict)
    }
    unresolved = [sid for sid in source_ids if str(sid) not in by_id]

    haystack: list[str] = []
    for sid in source_ids:
        rec = by_id.get(str(sid))
        if not rec:
            continue
        for ev in rec.get("evidence") or []:
            if isinstance(ev, dict):
                haystack.append(_normalise_quote(ev.get("quote", "")))
        for step in rec.get("practice_trace") or []:
            if isinstance(step, dict):
                haystack.append(_normalise_quote(step.get("action", "")))
                haystack.append(_normalise_quote(step.get("feedback", "")))
    haystack = [h for h in haystack if h]

    quotes = [
        _normalise_quote(ev.get("quote", ""))
        for ev in item.get("evidence") or []
        if isinstance(ev, dict)
    ]
    not_copied = [q for q in quotes if q and not any(q in h for h in haystack)]
    return {
        "source_l1_ids": [str(s) for s in source_ids],
        "unresolved_source_ids": unresolved,
        "quotes_total": len(quotes),
        "quotes_not_copied": len(not_copied),
        "not_copied_examples": [q[:120] for q in not_copied[:3]],
    }


_ACTIONABILITY_CHECKS = (
    "says_what_to_do",
    "says_when_it_applies",
    "says_what_to_expect",
    "says_when_not_to",
    "says_what_to_rebind",
)


def _score_actionability(scores: dict[str, Any]) -> float | None:
    """Fraction of the five reader-decision questions the record answers.

    A proxy, not survey's actionability metric: that one needs a fixed executor's
    decision list as an external reference, which this project does not have. An
    omitted check counts as unanswered, for the same reason an unexamined claim
    counts as insufficient in grounding.
    """
    block = scores.get("actionability_proxy")
    if not isinstance(block, dict) or not isinstance(block.get("checks"), dict):
        return None
    answered = sum(
        1 for name in _ACTIONABILITY_CHECKS if block["checks"].get(name) is True
    )
    return round(answered / len(_ACTIONABILITY_CHECKS), 4)


def _score_completeness(
    scores: dict[str, Any], *, expected: list[str], missing: list[str]
) -> float | None:
    """Share of expected fields that are present and not defective.

    Both failure modes count against the same denominator: a field the prompt
    asked for and the model omitted, and one it produced as a placeholder, cost a
    reader the same thing. Fields outside `expected` are not in the denominator at
    all, which is what keeps records from different prompt versions comparable.
    """
    if not expected:
        return None
    defective = {
        row.get("field")
        for row in scores.get("field_defects") or []
        if isinstance(row, dict) and row.get("field") in expected
    }
    bad = len(defective | set(missing))
    return round(max(0, len(expected) - bad) / len(expected), 4)


_APPLICABILITY_CHECKS = (
    "beyond_datasets",
    "beyond_models",
    "beyond_scale",
    "beyond_metrics",
    "mechanism_without_ablation",
    "superiority_without_baseline",
    "direction_from_single_point",
    "untested_conditions",
    "boundary_absent",
    "universal_phrasing",
)


def _recount_applicability(scores: dict[str, Any]) -> float | None:
    """Recompute applicability from its over-reach checks.

    Same reason as grounding: the score must follow from a fixed denominator, not
    from the scorer's arithmetic or overall impression. A check the scorer omitted
    counts as `false` -- an over-reach it did not report is not one it found.
    """
    block = scores.get("applicability")
    if not isinstance(block, dict) or not isinstance(block.get("checks"), dict):
        return None
    flagged = sum(1 for name in _APPLICABILITY_CHECKS if block["checks"].get(name) is True)
    return round(1.0 - flagged / len(_APPLICABILITY_CHECKS), 4)


# Weights for the composite used in ablation deltas. NOT a quality score:
# survey 7.1 requires dimensions be reported separately, and every per-dimension
# score stays in the output untouched. The composite exists only so that blanking
# one field yields a single comparable number.
#
# Grounding carries more weight because it rests on a fixed per-claim denominator
# and measured zero variance across repeated trials, while applicability rests on
# binary checks whose criteria are coarser -- one flipped check moves it 0.1, and
# the checks reading `evidence_scope` degrade to a judgment call on records
# predating that field. Weighting them equally would let the less stable dimension
# dominate the deltas it is least able to resolve.
#
# Re-derive these if the noise floor shifts: the ratio should track each
# dimension's measured stability, not an opinion about which matters more.
# `completeness` is deterministic given the record (the code owns both its
# numerator and denominator), so it contributes no variance; `actionability_proxy`
# reads what the record states rather than weighing it against evidence, which is
# an easier and steadier judgment than applicability's over-reach calls.
_DIMENSION_WEIGHTS = {
    "execution_grounding": 0.45,
    "applicability": 0.2,
    "actionability_proxy": 0.2,
    "completeness": 0.15,
}


def _diagnostic_total(scores: dict[str, Any]) -> float:
    """Weighted composite of dimension scores, for ablation deltas only.

    Renormalizes over whichever dimensions scored, so a record missing one
    dimension is not implicitly penalized.
    """
    total = 0.0
    weight_seen = 0.0
    for dim, weight in _DIMENSION_WEIGHTS.items():
        block = scores.get(dim)
        if isinstance(block, dict) and isinstance(block.get("score"), (int, float)):
            total += weight * float(block["score"])
            weight_seen += weight
    return round(total / weight_seen, 4) if weight_seen else 0.0


async def _run_noise_floor(
    records: list[tuple[Path, str, dict[str, Any]]],
    *,
    model: Any,
    concurrency: int,
    repeats: int,
) -> list[dict[str, Any]]:
    """Score each record `repeats` times on identical input.

    Establishes how much the scorer varies on its own, which bounds what an
    ablation delta can mean: a field whose measured contribution is smaller than
    this spread has not been shown to contribute anything.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one_pass(
        item: dict[str, Any], level: str, expected: list[str], trial: int
    ) -> dict[str, Any] | None:
        async with sem:
            try:
                return await _score_once(
                    item, level=level, expected=expected, model=model, attribute=False
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    trial {trial} FAILED {item.get('id')}: {exc}", file=sys.stderr
                )
                return None

    out = []
    for paper_dir, level, item in records:
        expected = _expected_fields(item, level=level)
        passes = await asyncio.gather(
            *(one_pass(item, level, expected, i) for i in range(repeats))
        )
        trials = [p for p in passes if p]
        if len(trials) < 2:
            print(f"  {item.get('id')}: too few successful trials", file=sys.stderr)
            continue

        totals = [_diagnostic_total(t) for t in trials]
        # Per-dimension stdev, so a rise in the composite can be traced to the
        # dimension that caused it rather than re-litigated from scratch.
        per_dim: dict[str, Any] = {}
        for dim in _DIMENSION_WEIGHTS:
            values = [
                t[dim]["score"]
                for t in trials
                if isinstance(t.get(dim), dict)
                and isinstance(t[dim].get("score"), (int, float))
            ]
            if len(values) >= 2:
                per_dim[dim] = {
                    "scores": values,
                    "stdev": round(statistics.pstdev(values), 4),
                    "range": round(max(values) - min(values), 4),
                }
        grounding = [
            t["execution_grounding"]["score"]
            for t in trials
            if isinstance(t.get("execution_grounding"), dict)
            and isinstance(t["execution_grounding"].get("score"), (int, float))
        ]
        drifts = sum(
            1
            for t in trials
            if isinstance(t.get("execution_grounding"), dict)
            and t["execution_grounding"].get("grouping_drift")
        )
        spread = round(max(totals) - min(totals), 4)
        row = {
            "id": item.get("id"),
            "paper": paper_dir.name,
            "level": level,
            "mandated_claim_count": _expected_claim_count(item),
            "trials": len(trials),
            "totals": totals,
            "total_spread": spread,
            "total_stdev": round(statistics.pstdev(totals), 4),
            "grounding_scores": grounding,
            "grounding_spread": round(max(grounding) - min(grounding), 4)
            if grounding
            else None,
            "per_dimension": per_dim,
            "grouping_drift_trials": drifts,
        }
        out.append(row)
        applic = [
            t["applicability"]["score"]
            for t in trials
            if isinstance(t.get("applicability"), dict)
            and isinstance(t["applicability"].get("score"), (int, float))
        ]
        row["applicability_scores"] = applic
        row["applicability_spread"] = (
            round(max(applic) - min(applic), 4) if applic else None
        )
        # Which specific checks flip, not just how much the score moved. A check
        # that disagrees across trials is a check whose criterion is underspecified
        # for this record -- that is a fixable prompt defect, whereas a uniformly
        # shifting score tells you nothing about where to look.
        per_check: dict[str, list[bool]] = {name: [] for name in _APPLICABILITY_CHECKS}
        for trial in trials:
            checks = (trial.get("applicability") or {}).get("checks") or {}
            for name in _APPLICABILITY_CHECKS:
                per_check[name].append(checks.get(name) is True)
        row["applicability_check_votes"] = per_check
        row["applicability_unstable_checks"] = sorted(
            name for name, votes in per_check.items() if len(set(votes)) > 1
        )
        row["has_evidence_scope"] = "evidence_scope" in item
        # Per-dimension spreads localize the variance. A counted ratio and a
        # free-form 0-1 judgment do not vary for the same reasons, and averaging
        # them into one number hides which of the two is unstable.
        print(
            f"  {item.get('id')}  claims={row['mandated_claim_count']}  "
            f"drift={drifts}/{len(trials)}",
            flush=True,
        )
        for dim, stats in per_dim.items():
            print(f"    {dim:22s} sd={stats['stdev']:<7} {stats['scores']}")
        print(
            f"    unstable checks {row['applicability_unstable_checks']} "
            f"(evidence_scope present: {row['has_evidence_scope']})"
        )
        print(f"    total         {totals} spread={spread}")
    return out


async def _run_attribution(
    records: list[tuple[Path, str, dict[str, Any]]],
    *,
    model: Any,
    concurrency: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def one(
        paper_dir: Path, level: str, item: dict[str, Any]
    ) -> dict[str, Any] | None:
        expected = _expected_fields(item, level=level)
        missing = _missing_fields(item, expected)
        async with sem:
            try:
                scores = await _score_once(
                    item, level=level, expected=expected, model=model, attribute=True
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  SCORING FAILED {item.get('id')}: {exc}", file=sys.stderr)
                return None
        row = {
            "id": item.get("id"),
            "paper": paper_dir.name,
            "level": level,
            "version": _record_version(item),
            "reference_set": _reference_set(item),
            "fields_expected": expected,
            "fields_missing": missing,
            "scores": scores,
            "diagnostic_total": _diagnostic_total(scores),
        }
        provenance = _quote_provenance(item, paper_dir, level)
        if provenance is not None:
            row["quote_provenance"] = provenance
        print(
            f"  scored {item.get('id')}  total={row['diagnostic_total']}"
            f"  ref={row['reference_set']}",
            flush=True,
        )
        return row

    done = await asyncio.gather(*(one(*rec) for rec in records))
    return [row for row in done if row]


async def _run_ablation(
    records: list[tuple[Path, str, dict[str, Any]]],
    *,
    model: Any,
    concurrency: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    async def scored(
        item: dict[str, Any], level: str, expected: list[str]
    ) -> dict[str, Any]:
        async with sem:
            return await _score_once(
                item, level=level, expected=expected, model=model, attribute=False
            )

    async def one(
        paper_dir: Path, level: str, item: dict[str, Any]
    ) -> dict[str, Any] | None:
        expected = _expected_fields(item, level=level)
        missing = _missing_fields(item, expected)
        try:
            base = await scored(item, level, expected)
        except Exception as exc:  # noqa: BLE001
            print(f"  BASELINE FAILED {item.get('id')}: {exc}", file=sys.stderr)
            return None
        base_total = _diagnostic_total(base)
        print(f"  {item.get('id')} baseline={base_total}", flush=True)

        fields = [f for f in _ABLATABLE_FIELDS if f in item and f not in missing]

        async def ablate(field: str) -> tuple[str, float | None]:
            stripped = {k: v for k, v in item.items() if k != field}
            # The blanked field leaves `fields_expected`: the scorer should treat
            # it as out of scope, so the delta measures the field's contribution
            # rather than a penalty for its absence.
            try:
                ablated = await scored(
                    stripped, level, [f for f in expected if f != field]
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    ablate {field}: FAILED ({exc})", file=sys.stderr)
                return field, None
            delta = round(base_total - _diagnostic_total(ablated), 4)
            print(f"    {item.get('id')} ablate {field}: {delta}", flush=True)
            return field, delta

        pairs = await asyncio.gather(*(ablate(f) for f in fields))
        return {
            "id": item.get("id"),
            "paper": paper_dir.name,
            "level": level,
            "version": _record_version(item),
            "fields_expected": expected,
            "fields_missing": missing,
            "baseline": base,
            "diagnostic_total": base_total,
            "contributions": {f: d for f, d in pairs if d is not None},
        }

    done = await asyncio.gather(*(one(*rec) for rec in records))
    return [row for row in done if row]


def _aggregate(results: list[dict[str, Any]], *, method: str) -> dict[str, Any]:
    """Aggregate per-field contribution across records.

    Counts how many records each field was measurable on, so a field present in
    two records is not compared against one present in twenty.
    """
    per_field: dict[str, list[float]] = defaultdict(list)
    if method == "ablation":
        for row in results:
            for field, delta in row.get("contributions", {}).items():
                per_field[field].append(delta)
    else:
        # Weight a field's share of a dimension by how much of that dimension's
        # score it could account for, split across the fields cited. Counting a
        # citation as 1.0 would make every cited field look equally important
        # and give the ranking no resolution at all.
        for row in results:
            for dim in _DIMENSION_WEIGHTS:
                block = row.get("scores", {}).get(dim) or {}
                cited = [f for f in (block.get("relied_on") or []) if f]
                score = block.get("score")
                if not cited or not isinstance(score, (int, float)):
                    continue
                share = round(float(score) / len(cited), 4)
                for field in cited:
                    per_field[field].append(share)

    summary = {
        field: {
            "n": len(values),
            "mean": round(statistics.fmean(values), 4),
            "max": round(max(values), 4),
        }
        for field, values in sorted(
            per_field.items(), key=lambda kv: -statistics.fmean(kv[1])
        )
    }

    missing_counts: dict[str, int] = defaultdict(int)
    for row in results:
        for field in row.get("fields_missing", []):
            missing_counts[field] += 1

    def _dims(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for dim in _DIMENSION_WEIGHTS:
            values = []
            for row in rows:
                block = (row.get("baseline") or row.get("scores") or {}).get(dim) or {}
                if isinstance(block.get("score"), (int, float)):
                    values.append(block["score"])
            if values:
                out[dim] = {
                    "n": len(values),
                    "mean": round(statistics.fmean(values), 4),
                    "min": round(min(values), 4),
                }
        return out

    # Grouped, not pooled: an induced record's quotes come from L1 records and a
    # directly extracted one's from the paper, so their grounding scores answer
    # questions about different reference sets. A single mean over both would
    # invite exactly the comparison the survey's audit rules rule out.
    by_reference: dict[str, Any] = {}
    for ref in sorted({str(row.get("reference_set") or "paper") for row in results}):
        rows = [row for row in results if str(row.get("reference_set") or "paper") == ref]
        totals = [
            row["diagnostic_total"]
            for row in rows
            if isinstance(row.get("diagnostic_total"), (int, float))
        ]
        by_reference[ref] = {
            "records": len(rows),
            "dimension_means": _dims(rows),
            "diagnostic_total_mean": (
                round(statistics.fmean(totals), 4) if totals else None
            ),
        }

    provenance_defects = [
        {
            "id": row.get("id"),
            **{
                k: v
                for k, v in (row.get("quote_provenance") or {}).items()
                if k != "source_l1_ids"
            },
        }
        for row in results
        if (row.get("quote_provenance") or {}).get("quotes_not_copied")
        or (row.get("quote_provenance") or {}).get("unresolved_source_ids")
        or (row.get("quote_provenance") or {}).get("error")
    ]

    out = {
        "method": method,
        "records_scored": len(results),
        "by_reference_set": by_reference,
        "dimension_means": _dims(results),
        "field_contribution": summary,
        "extraction_misses": dict(missing_counts),
    }
    if provenance_defects:
        out["quote_provenance_defects"] = provenance_defects
    return out


def _unstable_check_tally(results: list[dict[str, Any]]) -> dict[str, int]:
    """How many records each applicability check disagreed with itself on.

    A check that flips across identical trials has a criterion too vague to
    decide this record, which is a fixable prompt defect rather than irreducible
    model variance -- so the tally names where to look.
    """
    tally: dict[str, int] = defaultdict(int)
    for row in results:
        for name in row.get("applicability_unstable_checks", []) or []:
            tally[name] += 1
    return dict(sorted(tally.items(), key=lambda kv: -kv[1]))


def _load_records(
    bank: Path,
    project: str,
    limit: int,
    levels: tuple[str, ...],
    only: tuple[str, ...] = (),
) -> list[tuple[Path, str, dict[str, Any]]]:
    """Records to score, optionally narrowed to specific ids.

    `only` exists so a run interrupted partway can be finished by scoring just
    the records that never produced a score, instead of paying for the whole
    batch again. Ids are not unique -- a paper's direct and induced L2 records
    are numbered from `_01` independently -- so an id may select more than one
    record; all matches are scored.
    """
    project_dir = bank / "experiences" / "projects" / project
    if not project_dir.is_dir():
        raise SystemExit(f"project dir not found: {project_dir}")
    out: list[tuple[Path, str, dict[str, Any]]] = []
    for paper_dir in sorted(p for p in project_dir.iterdir() if p.is_dir()):
        for level in levels:
            path = paper_dir / f"{level}.json"
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for item in payload.get("experiences", []):
                if only and str(item.get("id") or "") not in only:
                    continue
                out.append((paper_dir, level, item))
                if len(out) >= limit:
                    return out
    return out


async def _main_async(args: argparse.Namespace) -> None:
    if args.model or args.provider:
        from EvoScientist.llm.models import get_chat_model

        model = get_chat_model(args.model, provider=args.provider)
    else:
        from EvoScientist.EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()

    levels = ("l1", "l2") if args.level == "both" else (args.level,)
    records = _load_records(
        args.bank, args.project, args.limit, levels, tuple(args.only or ())
    )
    if not records:
        raise SystemExit("no records found")
    print(f"scoring {len(records)} record(s), method={args.method}")

    if args.method == "noise":
        results = await _run_noise_floor(
            records,
            model=model,
            concurrency=args.concurrency,
            repeats=args.repeats,
        )
        spreads = [r["total_spread"] for r in results]
        stdevs = [r["total_stdev"] for r in results]
        # Lead with stdev, not range. Range is an extreme-value statistic: it
        # grows with trial count, so a run with more repeats reports a larger
        # range from identical scorer behaviour and the two runs cannot be
        # compared. Measured directly here -- the same three records gave range
        # 0.10 over 4 trials and 0.16 over 8.
        pooled = round(statistics.fmean(stdevs), 4) if stdevs else None
        summary = {
            "method": "noise",
            "records": len(results),
            "repeats": args.repeats,
            "per_record_stdev": stdevs,
            "pooled_stdev": pooled,
            "worst_stdev": round(max(stdevs), 4) if stdevs else None,
            "single_score_band": round(2 * pooled, 4) if pooled else None,
            "mean_of_n_band": round(2 * pooled / max(args.repeats, 1) ** 0.5, 4)
            if pooled
            else None,
            "per_record_range": spreads,
            "range_note": "Range grows with --repeats; do not compare ranges "
            "across runs with different repeat counts.",
            "records_with_grouping_drift": sum(
                1 for r in results if r["grouping_drift_trials"]
            ),
            "unstable_checks_by_record_count": _unstable_check_tally(results),
            "note": "single_score_band bounds what one scoring of one record can "
            "mean. mean_of_n_band is the corresponding bound on a mean over n "
            "records, which is what an A/B comparison of two extraction schemes "
            "actually rests on.",
        }
    elif args.method == "ablation":
        results = await _run_ablation(
            records, model=model, concurrency=args.concurrency
        )
        summary = _aggregate(results, method=args.method)
    else:
        results = await _run_attribution(
            records, model=model, concurrency=args.concurrency
        )
        summary = _aggregate(results, method=args.method)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        args.out.write_text(
            json.dumps(
                {"summary": summary, "records": results}, indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True, help="memory_dir root")
    parser.add_argument("--project", required=True, help="project id")
    parser.add_argument("--limit", type=int, default=8, help="records to score")
    parser.add_argument(
        "--only",
        nargs="*",
        help="score only these record ids (to finish an interrupted run)",
    )
    parser.add_argument(
        "--level", choices=("l1", "l2", "both"), default="l2", help="which level"
    )
    parser.add_argument(
        "--method",
        choices=("noise", "attribution", "ablation"),
        default="attribution",
        help="noise: score the same records repeatedly to measure scorer "
        "variance -- run this first, it bounds what any delta can mean. "
        "attribution: 1 call/record, model self-reports. "
        "ablation: 1+N calls/record, measures deltas directly.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="trials per record for --method noise",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="parallel scoring calls. A single 350-word record takes minutes, so "
        "serial runs are impractical past a handful of records.",
    )
    parser.add_argument("--model", help="override model for this run only")
    parser.add_argument("--provider", help="override provider for this run only")
    parser.add_argument("--out", type=Path, help="write full results as JSON")
    asyncio.run(_main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
