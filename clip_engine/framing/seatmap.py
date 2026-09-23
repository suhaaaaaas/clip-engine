"""Map each voice to a face on screen.

V1 leans on the fact that podcast seating barely changes: whichever face is
on screen while a given speaker talks is that speaker's seat. Learn it once per
show, store it, let a reviewer fix it once.

V2 (later) reads lip movement directly with an active-speaker model
(TalkNet / Light-ASD) for shows where seats move or guests rotate.

TODO(phase-2):
  - [ ] build_seatmap(turns, face_tracks) -> {speaker: track_id}, per angle
  - [ ] confidence per mapping; below a threshold, ask the reviewer
  - [ ] persist to the show config so later episodes reuse it
"""

from ..models import SpeakerTurn
from .faces import FaceTrack


def build_seatmap(
    turns: list[SpeakerTurn], face_tracks: list[FaceTrack]
) -> dict[str, int]:
    raise NotImplementedError("TODO(phase-2)")
