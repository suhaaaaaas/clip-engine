"""Measure the scorer against the golden set. Build this BEFORE the scorer.

The metric: of our 10 hand-picked clips for an episode, how many appear in the
scorer's top 10? A predicted clip counts as a hit when it overlaps a golden
clip by more than half. Phase 1 exits when 5 or more hit on an episode the
weights were never tuned on.

TODO(phase-0):
  - [ ] load_golden(path) -> list of labeled clips from golden/*.csv
  - [ ] agreement(labels_a, labels_b) -> float, how much two humans overlap.
        This is the ceiling; a scorer matching human-to-human agreement is done.
TODO(phase-1):
  - [ ] precision_at_k(predicted, golden, k=10, min_overlap=0.5)
  - [ ] report(): per-episode table so weight changes are comparable
  - [ ] never tune on the held-out episode
"""

from pathlib import Path

from ..models import ClipCandidate


def load_golden(path: Path) -> list[ClipCandidate]:
    raise NotImplementedError("TODO(phase-0)")


def agreement(labels_a: list[ClipCandidate], labels_b: list[ClipCandidate]) -> float:
    raise NotImplementedError("TODO(phase-0)")


def precision_at_k(
    predicted: list[ClipCandidate],
    golden: list[ClipCandidate],
    k: int = 10,
    min_overlap: float = 0.5,
) -> float:
    raise NotImplementedError("TODO(phase-1)")
