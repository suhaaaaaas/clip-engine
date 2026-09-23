"""Split the source into continuous camera shots.

Multi-camera podcasts already cut between angles, so face tracks and the seat
map are only valid within a single shot. Everything downstream runs per shot.

TODO(phase-2):
  - [ ] detect_shots(video_path) -> list[(start_ms, end_ms)] via PySceneDetect
  - [ ] group similar shots into angles, so a seat map is learned per angle
  - [ ] handle the single-camera case as one shot spanning the episode
"""

from pathlib import Path


def detect_shots(video_path: Path) -> list[tuple[int, int]]:
    raise NotImplementedError("TODO(phase-2)")
