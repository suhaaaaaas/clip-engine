"""Turn a transcript into candidate windows to score.

Windows start and end on segment boundaries -- a clip that opens mid-sentence
is dead on arrival. Overlapping windows get suppressed after scoring so one
good exchange yields one clip, not eight near-duplicates.

TODO(phase-1):
  - [ ] candidate_windows(transcript, min_s=20, max_s=60) -> list[(start, end)]
  - [ ] non_max_suppress(candidates, iou_threshold) -> list[ClipCandidate]
  - [ ] cap per 10 minutes + minimum score, never a fixed quota per episode
"""

from ..models import ClipCandidate, Transcript


def candidate_windows(
    transcript: Transcript, min_s: int = 20, max_s: int = 60
) -> list[tuple[int, int]]:
    raise NotImplementedError("TODO(phase-1)")


def non_max_suppress(
    candidates: list[ClipCandidate], iou_threshold: float = 0.4
) -> list[ClipCandidate]:
    raise NotImplementedError("TODO(phase-1)")
