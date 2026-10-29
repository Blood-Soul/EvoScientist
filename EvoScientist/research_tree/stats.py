"""Deciding between nodes from per-sample holdout scores.

Every node is scored on the same held-out rows, so two nodes are compared
pairwise: the per-row difference carries all the information, and the noise
of the row set itself cancels. "Better" means the bootstrap interval of the
mean difference lies above zero; anything else is "not resolved", which is a
result in its own right, not a loss.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Delta:
    mean: float
    low: float
    high: float

    @property
    def better(self) -> bool:
        return self.low > 0

    @property
    def worse(self) -> bool:
        return self.high < 0

    def text(self) -> str:
        return f"{self.mean:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"


def paired_delta(
    a: list[float],
    b: list[float],
    *,
    direction: str = "max",
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Delta:
    """Mean of a - b with a paired bootstrap interval; positive means a is better."""
    if len(a) != len(b) or not a:
        raise ValueError(
            f"paired comparison needs equal, non-empty score lists ({len(a)} vs {len(b)})"
        )
    sign = 1.0 if direction == "max" else -1.0
    d = [sign * (x - y) for x, y in zip(a, b, strict=True)]
    n = len(d)
    rng = random.Random(seed)
    counts = Counter(d)
    if len(counts) <= 16 and hasattr(rng, "binomialvariate"):  # Python 3.12+
        # Accuracy-like scores: the differences take a few values, so a
        # resample is a multinomial draw over them, done with chained binomials.
        values, probs = zip(*((v, c / n) for v, c in counts.items()), strict=True)
        means = []
        for _ in range(n_boot):
            left, mass, total = n, 1.0, 0.0
            for v, p in zip(values[:-1], probs[:-1], strict=True):
                k = (
                    rng.binomialvariate(left, min(1.0, p / mass))
                    if left and mass > 0
                    else 0
                )
                total += k * v
                left -= k
                mass -= p
            total += left * values[-1]
            means.append(total / n)
    else:
        means = [sum(rng.choices(d, k=n)) / n for _ in range(n_boot)]
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return Delta(sum(d) / n, lo, hi)


def complementarity(a: list[float], b: list[float], direction: str = "max") -> float:
    """How much a per-row oracle over two nodes would gain over the better of them.

    A large value means the nodes are right on different rows, which is the
    case a merge can exploit; near zero means one dominates the other.
    """
    pick = max if direction == "max" else min
    best_single = pick(sum(a) / len(a), sum(b) / len(b))
    oracle = sum(pick(x, y) for x, y in zip(a, b, strict=True)) / len(a)
    return (oracle - best_single) if direction == "max" else (best_single - oracle)


def merge_candidates(
    scores: dict[str, list[float]],
    *,
    direction: str = "max",
    top: int = 3,
    exclude_pairs: set[frozenset[str]] | None = None,
) -> list[tuple[str, str, float]]:
    ids = sorted(scores)
    pairs = []
    for i, x in enumerate(ids):
        for y in ids[i + 1 :]:
            if exclude_pairs and frozenset((x, y)) in exclude_pairs:
                continue
            pairs.append((x, y, complementarity(scores[x], scores[y], direction)))
    return sorted(pairs, key=lambda p: -p[2])[:top]


def slice_table(slices: list[str], scores: list[float]) -> dict[str, dict[str, float]]:
    """Per-slice row count, share of all rows and mean score."""
    by: dict[str, list[float]] = {}
    for s, v in zip(slices, scores, strict=False):
        by.setdefault(s, []).append(v)
    n = len(scores) or 1
    return {
        s: {
            "n": len(v),
            "share": round(len(v) / n, 4),
            "mean": round(sum(v) / len(v), 4),
        }
        for s, v in sorted(by.items())
    }
