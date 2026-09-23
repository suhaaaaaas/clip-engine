"""Join ASR segments to diarization turns.

Whisper's segment boundaries and pyannote's turn boundaries never line up, so
each segment takes the speaker it overlaps most. Both lists are sorted, so walk
them together rather than comparing every pair.

TODO(phase-0):
  - [ ] assign_speakers(segments, turns) -> list[Segment]
  - [ ] tests: majority overlap wins, gaps stay None, one turn can cover many
        segments, timings and text are never altered
"""

from ..models import Segment, SpeakerTurn


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    raise NotImplementedError("TODO(phase-0)")
