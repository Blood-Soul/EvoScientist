"""Quality gate for extracted experience records: grade each record with the expbench rubric, keep the good ones.

Ported from EvoScientist-Eval `experiment/expbench/rubric` (commit feffe92). The schema and the four dimension prompts
in `gate_rubric/` are verbatim copies; the checks, the evidence gate and the status rules below follow `judge.py`
there, so a record is graded the same way here as in the benchmark. Keep the two in step.

What it does
------------
Every record is graded on up to 18 binary checks in four dimensions (faithfulness, validity, scope, inference).
Three of them are program checks against the paper text (quotes verbatim, numbers present, locator exists); the rest
are judge checks, one model call per dimension, each `pass` backed by a span copied from the paper. A span the program
cannot find in the paper downgrades the verdict to `fail` after one repair turn. Dimension scores and the final status
are computed in code from the binary checks; the model never produces the number.

Status is one of ACCEPT, PROVISIONAL, REVISE, REJECT, INSUFFICIENT. A record enters the library on ACCEPT or
PROVISIONAL. PROVISIONAL means the only failure is an ungrounded mechanism claim: the record is kept and marked
UNVERIFIED_MECHANISM. Actionability (A1-A3) never changes the status, so it is not graded.

Modes (`memory_experience_gate`): `off` (default) does nothing; `audit` grades and writes `gate.json` but stores every
record; `enforce` also drops the records that did not make it. Dropped records are kept in `gate.json`.

The judge is the auxiliary chat model, one sample per check. If it gives no usable reply for a dimension the gate
raises, so the task fails and is retried rather than letting an ungraded record in.
"""

from __future__ import annotations

import asyncio
import functools
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ...utils import format_message_content
from .extraction import _V2_MAX_PAPER_CHARS, _without_lone_surrogates
from .render import is_current, is_l2_current

RUBRIC_DIR = Path(__file__).resolve().parent / "gate_rubric"
GATED_DIMENSIONS = ("faithfulness", "validity", "scope", "inference")
MODES = ("off", "audit", "enforce")
ADMIT = frozenset({"ACCEPT", "PROVISIONAL"})
VERDICTS = ("pass", "fail", "na")
SEVERITY = {"pass": 0, "na": 1, "fail": 2}
L2_CHECKS = frozenset({"F1", "F4", "F5", "S3", "S5", "I1", "I2", "I3"})
UNVERIFIED_MECHANISM = "UNVERIFIED_MECHANISM"
# Judge calls in flight for one paper's records. A drain already extracts 16 papers at once.
GATE_CONCURRENCY = 4

LAYER_GUIDANCE = {
    "l1": (
        "L1 is a reusable fact pattern grounded in one paper. It is more abstract "
        "than a retrieved chunk, but it must preserve the conditions that decide "
        "whether the observation transfers. Numbers are optional unless the record "
        "uses them as evidence; do not require a complete reproduction of the paper's "
        "dataset or model. Do not require a mechanism in an L1 record."
    ),
    "l2": (
        "L2 is a synthesis, not a second L1 and not a paper chunk. It states a "
        "pattern, mechanism, or transfer rule supported by the paper's summary or "
        "by the record's source L1 records. Specific numbers are optional. Judge "
        "whether the synthesis follows from its evidence, preserves uncertainty, "
        "and keeps its scope at the level the evidence supports. Do not demand a "
        "table locator, numeric effect, or full experiment setting when the L2 "
        "contract does not claim one."
    ),
}

REPAIR = """Some evidence spans were not found in their named source:

{bad}

Re-grade only these checks. Keep `pass` only when the evidence span is copied
verbatim from PAPER TEXT or, where the check allows it, from the RECORD. If no
verbatim span supports the claim, return `fail`. Return the same JSON shape."""


class GateError(RuntimeError):
    """The gate could not grade; nothing may be stored on its account."""


# ------------------------------------------------------------------- rubric

@functools.cache
def _schema() -> dict[str, Any]:
    return json.loads((RUBRIC_DIR / "schema.json").read_text(encoding="utf-8"))


def _dimensions() -> dict[str, Any]:
    return {name: _schema()["dimensions"][name] for name in GATED_DIMENSIONS}


def layer_checks(layer: str) -> set[str]:
    if layer == "l2":
        return set(L2_CHECKS)
    return {check["id"] for dim in _dimensions().values() for check in dim["checks"]}


def gate_mode() -> str:
    from ...config.settings import get_effective_config

    mode = str(getattr(get_effective_config(), "memory_experience_gate", "off") or "off").strip().lower()
    if mode not in MODES:
        raise ValueError(f"memory_experience_gate must be one of {', '.join(MODES)}, not {mode!r}")
    return mode


# ----------------------------------------------------------- evidence helpers

def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _alnum(text: str) -> str:
    """Letters and digits only: PDF/HTML extraction inserts stray spaces and punctuation around inline math."""
    return re.sub(r"[^0-9a-z一-鿿]+", "", (text or "").lower())


def _numbers(text: str) -> list[str]:
    out = []
    for match in re.finditer(r"(?<!\w)(\d+(?:[.,]\d+)*)\s*(%|x|×)?", text or ""):  # noqa: RUF001, RUF003
        token = match.group(1).replace(",", "")
        suffix = match.group(2) or ""
        if len(token.replace(".", "")) < 2 and suffix != "%":   # one-digit integers are too common to be evidence
            continue
        out.append(token + ("%" if suffix == "%" else ""))
    return out


def _number_in(token: str, paper: str) -> bool:
    raw = token.rstrip("%")
    candidates = {raw, raw.replace(",", "."), raw.replace(".", ",")}
    plain = re.sub(r"[^0-9.]", "", raw)
    if plain:
        candidates.add(plain)
    if re.fullmatch(r"0\.\d+", raw) or re.fullmatch(r"\d+\.\d+", raw):
        try:
            candidates.add(str(float(raw) * 100))
            candidates.add(str(round(float(raw) * 100, 2)))
        except ValueError:
            pass
    flat = paper.replace(",", "").replace(" ", "")
    if any(candidate.replace(" ", "").replace(",", ".") in flat for candidate in candidates):
        return True
    digits = re.sub(r"[^0-9]", "", paper)
    return bool(plain) and re.sub(r"[^0-9]", "", plain) in digits and len(re.sub(r"[^0-9]", "", plain)) >= 4


def _locator_hit(locator: str, paper: str) -> bool:
    if not locator.strip():
        return False
    terms = re.findall(r"(?:Table|Section|Figure|Sec\.|Fig\.)\s*[0-9]+[A-Za-z]?", locator) or [locator.strip()]
    text = _norm(paper)
    return any(_norm(term) in text for term in terms)


def _number_grounded(token: str, paper: str) -> bool:
    if _number_in(token, paper):
        return True
    raw = token.rstrip("%")
    if "." not in raw:
        return False
    try:
        value = float(raw)
    except ValueError:
        return False
    tolerance = max(0.005, min(0.05, abs(value) * 0.001))
    for match in re.finditer(r"-?\d+\.\d+", paper):
        try:
            if abs(float(match.group()) - value) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def _quote_ok(quote: str, paper: str) -> bool:
    """Verbatim after undoing PDF conversion artefacts: same words in the same order, a changed word still fails."""
    if _norm(quote) in _norm(paper):
        return True
    loose = _alnum(quote)
    return len(loose) >= 12 and loose in _alnum(paper)


def collect(record: Mapping[str, Any], path: str) -> list[str]:
    """All string/number leaves at a schema path such as ``source_setting.params[].value``."""
    head, _, rest = path.partition(".")
    many = head.endswith("[]")
    value = record.get(head[:-2] if many else head)

    def leaves(item: Any) -> list[str]:
        if isinstance(item, str):
            return [item]
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return [str(item)]
        if isinstance(item, Mapping):
            return [leaf for inner in item.values() for leaf in leaves(inner)]
        if isinstance(item, list):
            return [leaf for inner in item for leaf in leaves(inner)]
        return []

    out: list[str] = []
    if many:
        for item in value or []:
            if rest:
                if isinstance(item, Mapping):
                    out.extend(collect(item, rest))
            else:
                out.extend(leaves(item))
    elif rest:
        if isinstance(value, Mapping):
            out.extend(collect(value, rest))
    else:
        out.extend(leaves(value))
    return out


def _result(check: Mapping[str, Any], verdict: str, *, span: str = "", note: str = "") -> dict[str, Any]:
    return {
        "id": check["id"], "name": check.get("name", ""), "dimension": check.get("_dimension", ""),
        "hard": bool(check.get("hard")), "mode": check.get("mode", ""), "evidence": check.get("evidence", "paper"),
        "verdict": verdict, "evidence_span": span, "note": note,
    }


# ------------------------------------------------------------- program checks

def _check_f1(check: Mapping[str, Any], record: Mapping[str, Any], paper: str) -> dict[str, Any]:
    quotes = collect(record, "limits_stated[].quote")
    if not quotes:
        return _result(check, "na", note="no limits_stated quotes")
    bad = [q for q in quotes if not _quote_ok(q, paper)]
    if bad:
        return _result(check, "fail", span=bad[0], note="quote not found verbatim in paper")
    return _result(check, "pass", span=quotes[0], note=f"{len(quotes)} quotes found")


def _check_f2(check: Mapping[str, Any], record: Mapping[str, Any], paper: str) -> dict[str, Any]:
    values: list[str] = []
    for path in ("effect.compared", "effect.benefit", "effect.cost",
                 "source_setting.params[].value", "source_setting.params[].from"):
        values.extend(collect(record, path))
    tokens = list(dict.fromkeys(token for value in values for token in _numbers(value)))
    if not tokens:
        return _result(check, "na", note="no checkable numbers in effect or params")
    missing = [token for token in tokens if not _number_grounded(token, paper)]
    if missing:
        return _result(check, "fail", span=missing[0], note=f"numbers not in paper: {missing[:6]}")
    return _result(check, "pass", note=f"{len(tokens)} numbers grounded")


_LOCATOR_TERM_RE = re.compile(
    r"(?i)\b(tables?|sections?|figures?|secs?\.?|figs?\.?|appendix|appendices)\s*([IVXLC]+|\d+(?:\.\d+)*)")
_NUMBERED_HEADING_RE = re.compile(r"(?m)^\s*(?:#{1,4}\s*)?\d+(?:\.\d+)*\s+\S")
_KIND = {"tables": "table", "sections": "section", "figures": "figure", "figs": "figure", "secs": "section",
         "appendices": "appendix"}


def _check_f3(check: Mapping[str, Any], record: Mapping[str, Any], paper: str) -> dict[str, Any]:
    locator = (collect(record, "locator") or [""])[0].strip()
    if not locator:
        return _result(check, "fail", note="locator is empty")
    # A paper text without a single locator string (tables dropped by the conversion) cannot support this check.
    if not re.search(r"(?i)\b(table|figure|fig\.|section|sec\.|appendix)\s*\d", paper):
        return _result(check, "na", note="paper text carries no locator strings")
    if _locator_hit(locator, paper):
        return _result(check, "pass", span=locator, note="locator found in paper")
    terms = []
    for kind, number in _LOCATOR_TERM_RE.findall(locator):
        kind = kind.lower().rstrip(".")
        terms.append((_KIND.get(kind, kind), number.lower()))
    text = _norm(paper)
    has_headings = bool(_NUMBERED_HEADING_RE.search(paper))
    hits = [f"{kind} {number}" for kind, number in terms
            if f"{kind} {number}" in text
            or (kind == "section" and has_headings and re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])", text))]
    if hits:
        return _result(check, "pass", span=hits[0], note="locator found in paper")
    if terms and all(kind == "section" for kind, _ in terms) and not has_headings:
        return _result(check, "na", note="paper text has no section structure")   # cannot verify is not does not exist
    return _result(check, "fail", span=locator, note="locator not found in paper")


PROGRAM = {"F1": _check_f1, "F2": _check_f2, "F3": _check_f3}


# --------------------------------------------------------------- judge checks

def _render_checks(checks: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"- **{c['id']}** ({'hard' if c.get('hard') else 'soft'}; evidence: {c.get('evidence', 'paper')}): {c['question']}"
        for c in checks)


def _system_prompt(name: str, checks: Sequence[Mapping[str, Any]]) -> str:
    template = (RUBRIC_DIR / _dimensions()[name].get("prompt", f"{name}.md")).read_text(encoding="utf-8")
    return template.replace("{{CHECKS}}", _render_checks(checks))


def _user_prompt(record: Mapping[str, Any], paper: str, *, paper_id: str, ids: Sequence[str], layer: str) -> str:
    return "\n\n".join([
        f"PAPER_ID: {paper_id or 'unknown'}", f"LAYER: {layer.upper()}",
        "--- LAYER CONTRACT ---", LAYER_GUIDANCE[layer],
        "--- PAPER TEXT ---", paper,
        "--- RECORD ---", json.dumps(record, ensure_ascii=False, indent=1),
        f"Grade exactly these checks: {', '.join(ids)}"])


def _json_object(text: str) -> dict[str, Any] | None:
    """The largest JSON object in a reply, fenced or bare."""
    best: tuple[int, dict[str, Any]] | None = None
    starts = [m.start() for m in re.finditer(r"\{", text or "")]
    for start in starts:
        try:
            obj, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (best is None or end > best[0]):
            best = (end, obj)
    return best[1] if best else None


def _span_ok(span: str, paper: str, record: Mapping[str, Any], *, evidence: str = "paper") -> bool:
    """A span is usable when it is copied verbatim from the named source.

    For a paper-evidence check it must be in the PAPER, not merely in the record: accepting the record's own wording
    would let a paraphrase of the record vouch for itself, which is the failure this gate exists to stop.
    """
    cleaned = " ".join((span or "").split())
    if len(cleaned) < 2:
        return False
    needle = _norm(cleaned)
    record_text = _norm(json.dumps(record, ensure_ascii=False))
    haystacks = {"record": (record_text,), "either": (_norm(paper), record_text)}.get(evidence, (_norm(paper),))
    if any(needle in hay for hay in haystacks):
        return True
    # A table citation joins the caption line and the data row, which are not adjacent once the grid is flattened.
    lines = [line.strip() for line in (span or "").splitlines() if line.strip()]
    if len(lines) >= 2 and any(line.startswith("|") for line in lines):
        if all(any(_norm(line) in hay for hay in haystacks) for line in lines):
            return True
    if len(cleaned) >= 12:
        loose = _alnum(cleaned)
        if len(loose) >= 10 and any(loose in _alnum(hay) for hay in haystacks):
            return True
    # "Table 4" may be a caption the text extraction lost; F3 checks separately that the locator exists.
    return bool(re.fullmatch(
        r"(?i)(table|figure|fig\.?|section|sec\.?|appendix|eq\.?)\s*\d+(?:[.\-]\d+)*", cleaned))


async def _ask_once(model: Any, sem: asyncio.Semaphore, system: str, user: str) -> dict[str, Any] | None:
    async with sem:
        response = await model.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    return _json_object(format_message_content(response))


async def _ask(model: Any, sem: asyncio.Semaphore, system: str, user: str, what: str) -> dict[str, Any]:
    for _attempt in range(2):
        obj = await _ask_once(model, sem, system, user)
        if isinstance(obj, dict) and isinstance(obj.get("checks"), list):
            return obj
    raise GateError(f"the judge gave no usable verdicts for {what}")


async def _judge_dimension(name: str, record: Mapping[str, Any], paper: str, *, paper_id: str, layer: str,
                           only: set[str], model: Any, sem: asyncio.Semaphore) -> list[dict[str, Any]]:
    # The prompt lists every judge check in the dimension even when only some are scored for this layer: a shorter
    # list changed the verdicts on the shared checks, so the filter is applied to the result instead.
    all_checks = [dict(c, _dimension=name) for c in _dimensions()[name]["checks"] if c["mode"] == "judge"]
    ids = [c["id"] for c in all_checks]
    system = _system_prompt(name, all_checks)
    user = _user_prompt(record, paper, paper_id=paper_id, ids=ids, layer=layer)
    reply = await _ask(model, sem, system, user, f"{name} of {paper_id or 'a record'}")
    items = {str(i.get("id", "")).strip().upper(): i for i in reply["checks"] if isinstance(i, Mapping)}

    out = []
    for check in all_checks:
        item = items.get(check["id"])
        if item is None:
            out.append(_result(check, "na", note="judge returned no verdict"))
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        # A verdict that is none of pass/fail/na is not a pass.
        out.append(_result(check, verdict if verdict in VERDICTS else "fail",
                           span=str(item.get("evidence_span") or ""), note=str(item.get("note") or "")))

    # A paraphrase is not a citation: one repair turn asks for a verbatim span or a fail, then the span is held to
    # the same test. Every unverifiable check in the dimension is repaired, so a subset run repairs the same ones.
    unverifiable = [c for c in out if c["verdict"] == "pass"
                    and not _span_ok(c["evidence_span"], paper, record, evidence=c["evidence"])]
    if unverifiable:
        bad = "\n".join(f"- {c['id']}: {c['evidence_span'][:200]!r}" for c in unverifiable)
        repaired = await _ask_once(model, sem, system, user + "\n\n" + REPAIR.format(bad=bad)) or {}
        fixes = {str(i.get("id", "")).strip().upper(): i for i in repaired.get("checks") or []
                 if isinstance(i, Mapping)}
        for check in unverifiable:
            item = fixes.get(check["id"])
            if not item:
                check.update(verdict="fail", note=(check["note"] + " [evidence not verifiable]").strip())
                continue
            verdict = str(item.get("verdict", "")).strip().lower()
            span = str(item.get("evidence_span") or "")
            note = str(item.get("note") or check["note"])
            if verdict == "pass" and _span_ok(span, paper, record, evidence=check["evidence"]):
                check.update(verdict="pass", evidence_span=span, note=note)
            else:
                if verdict == "pass":
                    note = (note + " [evidence not verifiable]").strip()
                check.update(verdict="fail", evidence_span=span, note=note)
    return [c for c in out if c["id"] in only]


# ------------------------------------------------------------------- synthesis

def dimension_score(checks: Sequence[Mapping[str, Any]]) -> int | None:
    applicable = [c for c in checks if c["verdict"] != "na"]
    if not applicable:
        return None
    hard = [c for c in applicable if c["hard"] and c["verdict"] == "fail"]
    soft = [c for c in applicable if not c["hard"] and c["verdict"] == "fail"]
    if len(hard) >= 2 or (hard and soft):
        return 0
    if hard or soft:
        return 1
    return 2


def synthesize(checks: Sequence[Mapping[str, Any]], *, layer: str = "l1") -> dict[str, Any]:
    scores: dict[str, int | None] = {}
    for name, dimension in _dimensions().items():
        ids = {c["id"] for c in dimension["checks"]}
        scores[name] = dimension_score([c for c in checks if c["id"] in ids])
    applicable = [c for c in checks if c["verdict"] != "na"]
    any_hard_fail = any(c["hard"] and c["verdict"] == "fail" for c in applicable)
    f, v, s, i = (scores.get(k) for k in ("faithfulness", "validity", "scope", "inference"))
    # PROVISIONAL has one meaning: the only thing wrong is an ungrounded mechanism claim.
    i1_fail = any(c["id"] == "I1" and c["verdict"] == "fail" for c in applicable)
    other_i_fail = any(c["dimension"] == "inference" and c["id"] != "I1" and c["verdict"] == "fail"
                       for c in applicable)
    if layer == "l2":   # no numeric or comparison gate: supported synthesis, scope control, honest mechanism label
        clean = f == 2 and (s if s is not None else 1) >= 1 and (i if i is not None else 1) >= 1 and not any_hard_fail
        reject = f == 0 or (s is not None and s == 0) or (i is not None and i == 0)
    else:
        clean = f == 2 and v == 2 and (s if s is not None else 1) >= 1 and not any_hard_fail
        reject = f == 0 or v == 0 or (f == 1 and v == 1 and s == 0)
    if len(applicable) < 3:
        status = "INSUFFICIENT"
    elif reject:
        status = "REJECT"
    elif clean and not i1_fail and not other_i_fail:
        status = "ACCEPT"
    elif clean and i1_fail and not other_i_fail:
        status = "PROVISIONAL"
    else:
        status = "REVISE"
    return {
        "scores": scores, "status": status,
        "hard_failures": [c["id"] for c in applicable if c["hard"] and c["verdict"] == "fail"],
        "reasons": [f"{c['id']}: {c['note'] or c['verdict']}" for c in applicable if c["verdict"] == "fail"],
        "n_applicable": len(applicable),
    }


async def evaluate_record(record: Mapping[str, Any], paper: str, *, paper_id: str, layer: str, model: Any,
                          sem: asyncio.Semaphore) -> dict[str, Any]:
    scored = layer_checks(layer)
    checks: list[dict[str, Any]] = []
    for name, dimension in _dimensions().items():
        for check in dimension["checks"]:
            if check["mode"] == "program":
                spec = dict(check, _dimension=name)
                checks.append(PROGRAM[check["id"]](spec, record, paper) if check["id"] in scored
                              else _result(spec, "na", note="not scored for this layer"))
    # A dimension none of whose judge checks is scored for this layer needs no call (validity, for L2).
    names = [n for n, d in _dimensions().items()
             if any(c["mode"] == "judge" and c["id"] in scored for c in d["checks"])]
    graded = await asyncio.gather(*(
        _judge_dimension(n, record, paper, paper_id=paper_id, layer=layer, only=scored, model=model, sem=sem)
        for n in names))
    for part in graded:
        checks += part
    return {"layer": layer, "checks": checks, **synthesize(checks, layer=layer)}


# ------------------------------------------------------------------------ gate

def _lineage_map(kept_l1: Sequence[int]) -> dict[int, int]:
    """Old `l1-N` position (1-based) -> new position, for the L1 records that stay."""
    return {old + 1: new + 1 for new, old in enumerate(kept_l1)}


def _remap_citations(cited: Sequence[Any], mapping: Mapping[int, int]) -> list[Any] | None:
    """Rewrite `l1-N` labels to the positions the kept L1 records will have, or None if one was dropped."""
    out: list[Any] = []
    for label in cited:
        match = re.fullmatch(r"l1-(\d+)", str(label))
        if match is None:
            out.append(label)
            continue
        new = mapping.get(int(match.group(1)))
        if new is None:
            return None
        out.append(f"l1-{new}")
    return out


def _mark_provisional(record: Mapping[str, Any]) -> dict[str, Any]:
    marked = {**record, "gate": "PROVISIONAL"}
    mechanism = record.get("mechanism")
    if isinstance(mechanism, str) and mechanism.strip() and UNVERIFIED_MECHANISM not in mechanism:
        marked["mechanism"] = f"{UNVERIFIED_MECHANISM}: {mechanism}"
    return marked


async def gate_extraction(*, paper_id: str, paper_text: str, payloads: Mapping[str, Mapping[str, Any]],
                          model: Any | None = None) -> tuple[Mapping[str, Mapping[str, Any]], dict[str, Any] | None]:
    """Grade the records of one extraction. Returns (payloads to store, report for `gate.json`).

    With the gate off, the payloads come back untouched and the report is None. In `audit` mode they also come back
    untouched, with the report saying what `enforce` would have dropped.
    """
    mode = gate_mode()
    if mode == "off":
        return payloads, None
    for level, shape in (("l1", is_current), ("l2", is_l2_current)):
        for record in (payloads.get(level) or {}).get("experiences") or []:
            if not isinstance(record, Mapping) or not shape(record) or (level == "l1" and is_l2_current(record)):
                raise GateError(f"{level.upper()} records are not in the current (v2) shape; the gate cannot grade them")
    if model is None:
        from ...EvoScientist import _ensure_auxiliary_chat_model

        model = _ensure_auxiliary_chat_model()
    paper = _without_lone_surrogates(paper_text)[:_V2_MAX_PAPER_CHARS]
    sem = asyncio.Semaphore(GATE_CONCURRENCY)
    jobs = [(level, index, record) for level in ("l1", "l2")
            for index, record in enumerate((payloads.get(level) or {}).get("experiences") or [])]
    graded: dict[tuple[str, int], dict[str, Any]] = {}

    async def grade(level: str, index: int, record: Mapping[str, Any]) -> None:
        graded[(level, index)] = await evaluate_record(
            record, paper, paper_id=paper_id, layer=level, model=model, sem=sem)

    try:
        async with asyncio.TaskGroup() as group:   # the first failure cancels the other records' calls
            for level, index, record in jobs:
                group.create_task(grade(level, index, record))
    except ExceptionGroup as failure:
        raise failure.exceptions[0] from None

    keep = {key for key, result in graded.items() if result["status"] in ADMIT}
    kept_l1 = [i for i in range(len((payloads.get("l1") or {}).get("experiences") or [])) if ("l1", i) in keep]
    lineage = _lineage_map(kept_l1)
    out: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for level in ("l1", "l2"):
        if level not in payloads:
            continue
        stored: list[Any] = []
        for index, record in enumerate(payloads[level].get("experiences") or []):
            result = graded[(level, index)]
            entry = {"level": level, "index": index, "problem": " ".join(str(record.get("problem") or "").split())[:160],
                     "status": result["status"], "scores": result["scores"], "hard_failures": result["hard_failures"],
                     "reasons": result["reasons"], "admitted": (level, index) in keep}
            new_record: Mapping[str, Any] = record
            if entry["admitted"] and level == "l2" and record.get("source_l1_ids"):
                cited = _remap_citations(record["source_l1_ids"], lineage)
                if cited is None:   # an induced finding that leans on a dropped L1 record is not trusted either
                    entry.update(admitted=False, reasons=[*entry["reasons"], "cites an L1 record that was dropped"])
                else:
                    limits = []
                    for lim in record.get("limits_stated") or []:
                        if isinstance(lim, Mapping) and isinstance(lim.get("from"), list):
                            moved = _remap_citations(lim["from"], lineage)
                            if not moved:   # its L1 source was dropped by the gate: the summary has no ground left
                                continue
                            lim = {**lim, "from": moved}
                        limits.append(lim)
                    new_record = {**record, "source_l1_ids": cited, "limits_stated": limits}
            if entry["admitted"] and result["status"] == "PROVISIONAL":
                new_record = _mark_provisional(new_record)
            if not entry["admitted"]:
                entry["record"] = record
            entries.append(entry)
            if entry["admitted"]:
                stored.append(new_record)
        out[level] = {**payloads[level], "experiences": stored}

    report = {
        "mode": mode, "rubric": _schema()["version"], "admit": sorted(ADMIT),
        "graded_at": datetime.now(UTC).isoformat(),
        "graded": len(entries), "admitted": sum(e["admitted"] for e in entries),
        "rejected": sum(not e["admitted"] for e in entries), "records": entries,
    }
    if mode == "audit":   # nothing is dropped: the report says what enforce would have done
        return payloads, report
    return {**payloads, **out}, report
