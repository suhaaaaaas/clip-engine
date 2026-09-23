"""Find faces and follow them through a shot.

Only run on the windows that became clips -- roughly 10 minutes per episode
instead of 90. Sampling at ~5 fps is plenty; faces do not move that fast.

TODO(phase-2):
  - [ ] detect_faces(video_path, start_ms, end_ms, fps=5) -> detections
  - [ ] track(detections) -> FaceTrack list (link by IoU across frames)
  - [ ] pick a detector and write down why (speed vs small-face recall)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FaceTrack:
    track_id: int
    # (at_ms, x, y, w, h) in normalized coordinates.
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)


def detect_faces(video_path: Path, start_ms: int, end_ms: int, fps: int = 5) -> list:
    raise NotImplementedError("TODO(phase-2)")


def track(detections: list) -> list[FaceTrack]:
    raise NotImplementedError("TODO(phase-2)")
