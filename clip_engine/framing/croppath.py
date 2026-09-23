"""Turn 'who is speaking when' into a smooth crop path.

Smoothing matters more than detection accuracy. A crop that cuts on every
speaker change feels robotic even when every cut is technically correct.

Rules:
  - hold every shot at least ~1.5s; ignore shorter interjections
  - ease between positions, never jump
  - rapid back-and-forth -> stacked two-person layout
  - no clear speaker (laughter, crosstalk) -> hold the last framing

TODO(phase-2):
  - [ ] build_crop_path(turns, seatmap, face_tracks) -> list[CropKeyframe]
  - [ ] min_shot_ms and easing as tunable parameters
  - [ ] detect crosstalk windows and mark them two-up
  - [ ] a debug renderer that draws the crop box on the wide video
"""

from ..models import CropKeyframe, SpeakerTurn
from .faces import FaceTrack


def build_crop_path(
    turns: list[SpeakerTurn],
    seatmap: dict[str, int],
    face_tracks: list[FaceTrack],
    min_shot_ms: int = 1500,
) -> list[CropKeyframe]:
    raise NotImplementedError("TODO(phase-2)")
