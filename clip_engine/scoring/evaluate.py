"""Measure the scorer against the golden set.

The metric: of our hand-picked clips for an episode, how many appear in the
scorer's top 10? A predicted clip counts as a hit when its overlap with a
golden clip is more than half of their union (IoU > 0.5). Matching is
one-to-one, so two near-duplicate predictions can't both claim the same pick.
Phase 1 exits when 5 or more hit on an episode the weights were never tuned
on -- EvalRow.held_out marks that episode so the report can't hide it.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from ..models import ClipCandidate

CLIP_TYPES = frozenset({"hot_take", "big_moment", "argument", "joke", "prediction"})
REQUIRED_COLUMNS = ("start_s", "end_s", "reason", "type")


def load_golden(path: Path) -> list[ClipCandidate]:
    """Parse golden/<episode_id>_<name>.csv, failing loudly on any bad row."""
    episode_id = path.stem.rsplit("_", 1)[0]
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")

        clips = []
        for row_num, row in enumerate(reader, start=2):
            where = f"{path.name}:{row_num}"
            try:
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
            except (TypeError, ValueError):
                raise ValueError(f"{where}: start_s and end_s must be numbers") from None
            if end_s <= start_s:
                raise ValueError(f"{where}: end_s must be after start_s")
            kind = (row["type"] or "").strip()
            if kind not in CLIP_TYPES:
                raise ValueError(f"{where}: type {kind!r} not one of {sorted(CLIP_TYPES)}")
            clips.append(
                ClipCandidate(
                    episode_id=episode_id,
                    start_ms=round(start_s * 1000),
                    end_ms=round(end_s * 1000),
                    score=1.0,
                    text=(row["reason"] or "").strip(),
                )
            )
    return clips


def iou(a: ClipCandidate, b: ClipCandidate) -> float:
    overlap = min(a.end_ms, b.end_ms) - max(a.start_ms, b.start_ms)
    if overlap <= 0:
        return 0.0
    union = (a.end_ms - a.start_ms) + (b.end_ms - b.start_ms) - overlap
    return overlap / union


def _match(a: list[ClipCandidate], b: list[ClipCandidate], min_overlap: float) -> int:
    """Greedy one-to-one matching, best overlaps first. Returns the match count."""
    pairs = sorted(
        ((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    for score, i, j in pairs:
        if score <= min_overlap:
            break
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return len(used_a)


def agreement(
    labels_a: list[ClipCandidate],
    labels_b: list[ClipCandidate],
    min_overlap: float = 0.5,
) -> float:
    """How much two independent label sets overlap, 0-1. The scorer's ceiling.

    Matched by timestamp only: two people labelling separately never share ids.
    """
    total = len(labels_a) + len(labels_b)
    if total == 0:
        return 0.0
    return 2 * _match(labels_a, labels_b, min_overlap) / total


def hits_at_k(
    predicted: list[ClipCandidate],
    golden: list[ClipCandidate],
    k: int = 10,
    min_overlap: float = 0.5,
) -> int:
    if k <= 0:
        raise ValueError("k must be positive")
    return _match(predicted[:k], golden, min_overlap)


def precision_at_k(
    predicted: list[ClipCandidate],
    golden: list[ClipCandidate],
    k: int = 10,
    min_overlap: float = 0.5,
) -> float:
    return hits_at_k(predicted, golden, k, min_overlap) / k


@dataclass
class EvalRow:
    episode_id: str
    k: int
    hits: int
    golden_count: int
    agreement: float | None = None
    held_out: bool = False

    @property
    def precision(self) -> float:
        return self.hits / self.k


def report(rows: list[EvalRow]) -> str:
    """Per-episode table, so weight changes are comparable run to run."""
    header = f"{'episode':<16} {'k':>3} {'hits':>4} {'golden':>6} {'p@k':>5} {'human_agr':>9}"
    lines = [header, "-" * len(header)]
    for r in rows:
        agr = f"{r.agreement:.2f}" if r.agreement is not None else "-"
        line = (
            f"{r.episode_id:<16} {r.k:>3} {r.hits:>4} {r.golden_count:>6} "
            f"{r.precision:>5.2f} {agr:>9}"
        )
        if r.held_out:
            line += "  held out"
        lines.append(line)
    return "\n".join(lines)
