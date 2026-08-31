"""Coarse discipline vocabulary for experience browsing.

The browse facets need a *bounded* top level. ``domain`` cannot serve: it is
free text written by the extraction model, and 33 papers already produced 46
distinct values with heavy near-duplication (``automated_scientific_discovery``
/ ``autonomous_scientific_discovery`` / ``scientific_discovery_agents`` are one
subject). At library scale that facet dissolves into thousands of near-synonyms.

``domain_arxiv`` cannot serve either, for a different reason: its coverage
depends on where a paper came from. Only arXiv papers carry a primary category,
so biology, chemistry and medical literature arriving from PubMed, bioRxiv or
journals has none -- 11 of the first 98 records are already ``None``. A facet
whose population is decided by the source feed is not a facet.

So the top level is this fixed vocabulary, derived from ``domain_arxiv`` when
available and asked of the extraction model otherwise. Bounded by us, not by
the feed, and extensible: adding a discipline is one entry here plus its arXiv
prefixes.
"""

from __future__ import annotations

DISCIPLINES: tuple[str, ...] = (
    "cs",
    "math",
    "physics",
    "chem",
    "bio",
    "med",
    "materials",
    "earth",
    "econ",
    "eng",
    "other",
)

FALLBACK_DISCIPLINE = "other"

# Longest-prefix wins, so ``cond-mat.mtrl-sci`` reaches materials while the rest
# of condensed matter stays in physics.
_ARXIV_PREFIXES: tuple[tuple[str, str], ...] = (
    ("cond-mat.mtrl-sci", "materials"),
    ("physics.chem-ph", "chem"),
    ("physics.geo-ph", "earth"),
    ("physics.med-ph", "med"),
    ("cs.", "cs"),
    ("stat.", "math"),
    ("math.", "math"),
    ("math-ph", "physics"),
    ("cond-mat", "physics"),
    ("physics.", "physics"),
    ("quant-ph", "physics"),
    ("astro-ph", "physics"),
    ("gr-qc", "physics"),
    ("hep-", "physics"),
    ("nucl-", "physics"),
    ("nlin.", "physics"),
    ("chem-ph", "chem"),
    ("q-bio.", "bio"),
    ("q-fin.", "econ"),
    ("econ.", "econ"),
    ("eess.", "eng"),
)


def normalize_discipline(value: object) -> str | None:
    """Return ``value`` when it names a known discipline, else None."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().casefold()
    return candidate if candidate in DISCIPLINES else None


def discipline_from_arxiv(domain_arxiv: object) -> str | None:
    """Map an arXiv primary category onto the coarse vocabulary."""
    if not isinstance(domain_arxiv, str):
        return None
    category = domain_arxiv.strip().casefold()
    if not category:
        return None
    for prefix, discipline in _ARXIV_PREFIXES:
        if category.startswith(prefix):
            return discipline
    return None


def resolve_discipline(
    *, discipline: object = None, domain_arxiv: object = None
) -> str:
    """Resolve one record's discipline.

    An explicit, in-vocabulary value wins; otherwise it is derived from the
    arXiv primary category; otherwise the record lands in ``other`` so it stays
    reachable by browsing instead of dropping out of the facet entirely.
    """
    explicit = normalize_discipline(discipline)
    if explicit is not None:
        return explicit
    derived = discipline_from_arxiv(domain_arxiv)
    if derived is not None:
        return derived
    return FALLBACK_DISCIPLINE


__all__ = [
    "DISCIPLINES",
    "FALLBACK_DISCIPLINE",
    "discipline_from_arxiv",
    "normalize_discipline",
    "resolve_discipline",
]
