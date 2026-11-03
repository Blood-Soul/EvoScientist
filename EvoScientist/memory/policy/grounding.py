"""Keep "the task requires ..." honest.

The coach turns an intermediary's paraphrase of the agent's situation into
guidance. In two runs that paraphrase carried rules the task statement never
contained ("this task forbids spending has-answer accuracy", "this task forbids
extra GPU compute"), because the agent's own plan was restated as the task and
then quoted back as if the user had imposed it. The agent adopted them as hard
constraints.

The prompts now separate the verbatim task statement from the agent's plan, but a
prompt is a request, not a guarantee. This module is the guarantee: a deterministic
check that runs on the final text. A sentence that attributes a requirement,
prohibition or limit to *the task* has to be supported by the task statement
itself; otherwise the attribution is rewritten so the agent reads it as what it
is, a plan assumed earlier in the run.

The check is lexical on purpose. It never calls a model, it cannot be talked out
of its answer, and it costs microseconds. It is conservative: a claim is only
touched when it is attributed to the task *and* most of what it claims does not
appear in the task statement. Claims too short to verify are left alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

# "this task requires", "the task statement says", "which this task excludes" ...
_CLAIM = re.compile(
    r"\b(?P<subject>(?:this|the|current|our)\s+(?:task|benchmark|challenge|assignment)"
    r"(?:\s+statement)?)\b"
    r"(?P<gap>[^.;,\n]{0,30}?)"
    r"\b(?P<verb>requires?|required|requiring|forbids?|forbidden|prohibits?|prohibited|"
    r"disallows?|excludes?|excluded|rules?\s+out|states|stated|says?|stipulates?|"
    r"stipulated|specifies|specified|imposes?|imposed|mandates?|demands?|"
    r"asks?\s+for|asked\s+for|fixes|fixed|limits?|only\s+allows?|protects?|prioriti[sz]es?|"
    r"does\s+not\s+(?:allow|permit|include|require))\b",
    re.IGNORECASE,
)

# "... the pre-declared tolerance from the current task statement (TOL = 0.005)":
# a value or rule credited to the task statement as its source. The claim is the
# clause BEFORE the phrase. "Derive X from the task statement" is an instruction,
# not a claim, and has too little before it to be judged, so it is left alone.
_SOURCE = re.compile(
    r"\b(?:from|in|per|under|according\s+to|as\s+(?:stated|specified|given|defined|required)\s+(?:in|by)|"
    r"(?:stated|specified|required|mandated|imposed|demanded)\s+(?:in|by))\s+"
    r"(?P<subject>(?:the\s+|this\s+)?(?:current\s+|given\s+)?(?:task|problem)\s+(?:statement|description))\b",
    re.IGNORECASE,
)

# End of a sentence: terminal punctuation followed by space/end (so "0.005" and
# "has-answer" do not split), or a line break.
_SENTENCE_END = re.compile(r"[.;!?](?=\s|$)|\n")

_WORD = re.compile(r"[a-z0-9]+")

# "... trading X for Y, which this task excludes." The claim is the clause BEFORE
# "this task", and the verb has no object of its own.
_RELATIVE_LEAD = re.compile(r"\b(?:which|that|who|where)\s*$", re.IGNORECASE)
_CLAUSE_START = re.compile(r"[;\n]|[.!?](?=\s)")

# Words that carry no claim. Kept short on purpose: this only has to stop
# function words from inflating the overlap score.
_STOP = frozenset(
    """a about above after again all also an and any are as at be been being both but by
    can could did do does done each either for from had has have having how if in into is
    it its itself may might more most must no nor not of on one only or other our out over
    per same shall should so some such than that the their them then there these they this
    those through to too under until up upon use used using very via was we were what when
    where which while who will with within without would you your every whole entire
    given based make makes need needs""".split()
)

REWRITE = "a claim attributed to the task that could not be matched to the task statement (check the statement before relying on it)"
REWRITE_POSSESSIVE = "your plan"

# Fewer content words than this cannot be verified, so the claim is left alone.
MIN_CONTENT_WORDS = 2
# Share of a claim's content words that must appear in the task statement.
GROUNDED_SHARE = 0.75


def _stem(word: str) -> str:
    """Cheap normalisation so 'columns' matches 'column' and 'allowed' 'allow'."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


_NUMBER_WORDS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten".split()
    )
}


def _content_words(text: str) -> set[str]:
    words: set[str] = set()
    for w in _WORD.findall(text.lower()):
        w = _NUMBER_WORDS.get(w, w)
        if (len(w) >= 3 or w.isdigit()) and w not in _STOP:
            words.add(_stem(w))
    return words


@dataclass(frozen=True)
class UngroundedClaim:
    """One task-level claim that the task statement does not support."""

    claim: str
    missing: tuple[str, ...]
    share: float
    start: int  # offset of the attributed subject ("this task") in the scanned text


def _scan(text: str, spec: str) -> Iterator[tuple[re.Match[str], UngroundedClaim]]:
    """Yield every task-attributed claim in ``text`` that is not grounded in ``spec``."""
    hits = list(_scan_verbs(text, spec)) + list(_scan_sources(text, spec))
    taken: set[int] = set()
    for match, claim in sorted(hits, key=lambda h: h[1].start):
        if claim.start in taken:
            continue
        taken.add(claim.start)
        yield match, claim


def _scan_sources(
    text: str, spec: str
) -> Iterator[tuple[re.Match[str], UngroundedClaim]]:
    spec_words = _content_words(spec or "")
    for match in _SOURCE.finditer(text):
        before = text[: match.start()]
        starts = list(_CLAUSE_START.finditer(before))
        clause = before[starts[-1].end() :] if starts else before
        words = _content_words(clause)
        if len(words) < MIN_CONTENT_WORDS:
            continue
        missing = sorted(w for w in words if w not in spec_words)
        share = 1 - len(missing) / len(words)
        if share < GROUNDED_SHARE:
            yield (
                match,
                UngroundedClaim(
                    claim=clause.strip(),
                    missing=tuple(missing),
                    share=round(share, 2),
                    start=match.start("subject"),
                ),
            )


def _scan_verbs(
    text: str, spec: str
) -> Iterator[tuple[re.Match[str], UngroundedClaim]]:
    spec_words = _content_words(spec or "")
    for match in _CLAIM.finditer(text):
        end = _SENTENCE_END.search(text, match.end())
        claim_text = text[match.end() : end.start() if end else len(text)]
        words = _content_words(claim_text)
        if len(words) < MIN_CONTENT_WORDS and _RELATIVE_LEAD.search(
            text[: match.start("subject")]
        ):
            # Relative clause: judge the clause the subject is attached to.
            before = text[: match.start("subject")]
            starts = list(_CLAUSE_START.finditer(before))
            claim_text = before[starts[-1].end() :] if starts else before
            words = _content_words(claim_text)
        if len(words) < MIN_CONTENT_WORDS:
            continue
        missing = sorted(w for w in words if w not in spec_words)
        share = 1 - len(missing) / len(words)
        if share < GROUNDED_SHARE:
            yield (
                match,
                UngroundedClaim(
                    claim=claim_text.strip(),
                    missing=tuple(missing),
                    share=round(share, 2),
                    start=match.start("subject"),
                ),
            )


def find_ungrounded_claims(text: str, spec: str) -> list[UngroundedClaim]:
    """Task-level claims in ``text`` that ``spec`` (the task statement) lacks."""
    return [claim for _, claim in _scan(text or "", spec)]


def ground_text(text: str, spec: str) -> tuple[str, list[UngroundedClaim]]:
    """Rewrite unsupported task-level attributions; return the text and what changed.

    Only the attribution is replaced ("this task forbids X" becomes "a claim
    attributed to the task that could not be matched to the task statement (check
    the statement before relying on it) forbids X"), so the advice survives but its
    borrowed authority does not. The wording is deliberately neutral: the check is
    lexical, so it can fail to match a real rule that is merely phrased differently
    from the statement, and it must not tell the agent that a real rule is not one.
    """
    if not text:
        return text, []
    hits = list(_scan(text, spec))
    if not hits:
        return text, []
    pieces: list[str] = []
    cursor = 0
    for match, _ in hits:
        pieces.append(text[cursor : match.start("subject")])
        # "the task's fixed windows" -> "your plan's fixed windows"
        possessive = text[match.end("subject") : match.end("subject") + 2] in (
            "'s",
            "\u2019s",
        )
        pieces.append(REWRITE_POSSESSIVE if possessive else REWRITE)
        cursor = match.end("subject")
    pieces.append(text[cursor:])
    return "".join(pieces), [claim for _, claim in hits]


__all__ = [
    "GROUNDED_SHARE",
    "MIN_CONTENT_WORDS",
    "REWRITE",
    "REWRITE_POSSESSIVE",
    "UngroundedClaim",
    "find_ungrounded_claims",
    "ground_text",
]
