"""Turn a transcript into candidate windows to score, then thin the results.

Windows start and end on segment boundaries -- a clip that opens mid-sentence
is dead on arrival. Overlapping windows get suppressed after scoring so one
good exchange yields one clip, not eight near-duplicates. Volume is a cap per
10 minutes plus a minimum score, never a fixed quota per episode.
"""

import bisect
from collections import Counter

from ..models import ClipCandidate, Transcript
from .evaluate import iou

TEN_MINUTES_MS = 10 * 60 * 1000


def candidate_windows(
    transcript: Transcript, min_s: int = 20, max_s: int = 60
) -> list[tuple[int, int]]:
    """Every (segment start, later segment end) span between min_s and max_s."""
    segs = transcript.segments
    min_ms, max_ms = min_s * 1000, max_s * 1000
    windows = []
    for i, first in enumerate(segs):
        for last in segs[i:]:
            duration = last.end_ms - first.start_ms
            if duration > max_ms:
                break
            if duration >= min_ms:
                windows.append((first.start_ms, last.end_ms))
    return windows


def non_max_suppress(
    candidates: list[ClipCandidate], iou_threshold: float = 0.4
) -> list[ClipCandidate]:
    """Greedy: keep the best-scoring window, drop anything overlapping it too much."""
    kept: list[ClipCandidate] = []
    # Kept windows sorted by start, so each candidate is only compared with
    # kept windows that could overlap it.
    kept_starts: list[int] = []
    kept_by_start: list[ClipCandidate] = []
    longest = 0
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        lo = bisect.bisect_left(kept_starts, c.start_ms - longest)
        hi = bisect.bisect_right(kept_starts, c.end_ms)
        if any(iou(c, k) > iou_threshold for k in kept_by_start[lo:hi]):
            continue
        kept.append(c)
        pos = bisect.bisect_right(kept_starts, c.start_ms)
        kept_starts.insert(pos, c.start_ms)
        kept_by_start.insert(pos, c)
        longest = max(longest, c.end_ms - c.start_ms)
    return kept


def cap_volume(
    candidates: list[ClipCandidate], max_per_10_min: int, min_score: float
) -> list[ClipCandidate]:
    """Drop weak clips, then keep at most max_per_10_min per 10-minute stretch."""
    per_bucket: Counter[int] = Counter()
    kept = []
    for c in sorted(candidates, key=lambda c: c.score, reverse=True):
        if c.score < min_score:
            break
        bucket = c.start_ms // TEN_MINUTES_MS
        if per_bucket[bucket] < max_per_10_min:
            per_bucket[bucket] += 1
            kept.append(c)
    return kept
