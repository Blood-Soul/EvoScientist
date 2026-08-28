"""Section-aware Markdown chunking for stored paper full text.

Jina Reader renders paper PDFs as Markdown with ATX headings ("## Method"),
so headings are the most reliable structural signal available. Chunking
follows them rather than a fixed character window, which keeps each chunk
inside one section and lets a retrieval hit name the section it came from --
the same section vocabulary the L1/L2 prompts already quote in their
`evidence[].section` fields.

Chunk ids are stable hashes over (project, paper, index) so a re-chunk of
unchanged text yields unchanged ids.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

CHUNKING_VERSION = 1

DEFAULT_MAX_CHUNK_CHARS = 2000
DEFAULT_OVERLAP_CHARS = 200
# Below this, a section is folded into its neighbour rather than standing
# alone -- a bare "## Results" heading with one line under it is not a
# useful retrieval unit.
MIN_CHUNK_CHARS = 200

# ATX heading at line start: 1-6 hashes, then the title text. Setext
# headings (underlined with === or ---) are not produced by Jina's
# conversion, so they are deliberately unhandled.
_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,6})[ \t]+(?P<title>\S.*?)[ \t]*$", re.MULTILINE
)

# Paragraph boundary used to split an oversized section: one or more blank
# lines. Falls back to a hard character cut when a section has no blank
# lines at all (dense single-paragraph PDFs).
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True)
class PaperChunk:
    """One retrievable span of a stored paper, addressed by character offset."""

    chunk_id: str
    chunk_index: int
    section: str
    section_path: str
    char_start: int
    char_end: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "section_path": self.section_path,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
        }


@dataclass
class _Section:
    """A heading and the body text beneath it, as offsets into the source."""

    section: str
    section_path: str
    start: int
    end: int
    heading_levels: list[tuple[int, str]] = field(default_factory=list)


def chunk_id_for(*, project_id: str, paper_key: str, chunk_index: int) -> str:
    """Return the stable id for one chunk position within one stored paper."""
    digest = hashlib.sha256(
        f"{project_id}:{paper_key}:{chunk_index}".encode()
    ).hexdigest()
    return f"C-{digest[:16]}"


def _section_spans(text: str) -> list[_Section]:
    """Split text into heading-delimited spans, tracking the heading path.

    Text before the first heading (title block, abstract preamble) becomes a
    leading span labelled "Preamble" so no content is dropped.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [_Section(section="", section_path="", start=0, end=len(text))]

    sections: list[_Section] = []
    if matches[0].start() > 0:
        sections.append(
            _Section(
                section="Preamble",
                section_path="Preamble",
                start=0,
                end=matches[0].start(),
            )
        )

    # Stack of (level, title) forming the path to the current heading.
    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group("hashes"))
        title = " ".join(match.group("title").split())
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(
            _Section(
                section=title,
                section_path=" > ".join(entry[1] for entry in stack),
                # Include the heading line itself: a chunk that opens with its
                # own heading reads correctly when handed to the model, and the
                # heading tokens contribute to lexical matching.
                start=match.start(),
                end=end,
                heading_levels=list(stack),
            )
        )
    return sections


def _merge_short_sections(
    sections: list[_Section], *, min_chars: int, max_chars: int
) -> list[_Section]:
    """Fold sections shorter than ``min_chars`` into the previous span.

    Merging only happens when the combined span still fits ``max_chars``, so
    folding never manufactures an oversized section that then needs splitting
    back apart. The merged span keeps the earlier section's label, since that
    is the heading a reader would scan to.
    """
    merged: list[_Section] = []
    for section in sections:
        length = section.end - section.start
        if (
            merged
            and length < min_chars
            and (section.end - merged[-1].start) <= max_chars
        ):
            merged[-1] = _Section(
                section=merged[-1].section,
                section_path=merged[-1].section_path,
                start=merged[-1].start,
                end=section.end,
                heading_levels=merged[-1].heading_levels,
            )
            continue
        merged.append(section)
    return merged


def _split_span(
    text: str, *, start: int, end: int, max_chars: int, overlap_chars: int
) -> list[tuple[int, int]]:
    """Split one over-long span into overlapping windows at paragraph breaks.

    Each window is grown to the last paragraph break that fits, so chunks end
    on a natural boundary. The next window starts ``overlap_chars`` earlier so
    a statement straddling a boundary appears whole in at least one chunk.
    """
    if end - start <= max_chars:
        return [(start, end)]

    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        limit = min(cursor + max_chars, end)
        if limit < end:
            window = text[cursor:limit]
            breaks = list(_PARAGRAPH_BREAK_RE.finditer(window))
            # Only honour a break past the halfway mark; an early break would
            # produce a runt chunk and stall progress.
            usable = [
                match for match in breaks if match.start() > (limit - cursor) // 2
            ]
            if usable:
                limit = cursor + usable[-1].start()
        spans.append((cursor, limit))
        if limit >= end:
            break
        # Step forward by at least one character even when overlap >= window,
        # so a pathological config cannot loop forever.
        advance = max(1, (limit - cursor) - overlap_chars)
        cursor += advance
    return spans


def chunk_paper_text(
    text: str,
    *,
    project_id: str,
    paper_key: str,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[PaperChunk]:
    """Split stored paper text into section-aware, offset-addressed chunks.

    Offsets are relative to ``text`` exactly as persisted, so a caller holding
    ``paper.md`` can recover any chunk by slicing without re-running this.
    """
    if not text.strip():
        return []
    max_chars = max(1, max_chunk_chars)
    # Overlap must stay strictly under the window or windows stop advancing.
    overlap = max(0, min(overlap_chars, max_chars - 1))

    sections = _merge_short_sections(
        _section_spans(text), min_chars=MIN_CHUNK_CHARS, max_chars=max_chars
    )

    chunks: list[PaperChunk] = []
    for section in sections:
        for span_start, span_end in _split_span(
            text,
            start=section.start,
            end=section.end,
            max_chars=max_chars,
            overlap_chars=overlap,
        ):
            body = text[span_start:span_end]
            if not body.strip():
                continue
            index = len(chunks)
            chunks.append(
                PaperChunk(
                    chunk_id=chunk_id_for(
                        project_id=project_id,
                        paper_key=paper_key,
                        chunk_index=index,
                    ),
                    chunk_index=index,
                    section=section.section,
                    section_path=section.section_path,
                    char_start=span_start,
                    char_end=span_end,
                    text=body,
                )
            )
    return chunks
