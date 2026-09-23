"""Who spoke when (pyannote).

Needs a HuggingFace token and acceptance of the model terms. Slow on CPU --
expect several minutes for a long episode.

TODO(phase-0):
  - [ ] diarize(audio_path, num_speakers) -> list[SpeakerTurn]
  - [ ] cache results; this is the slowest local step
  - [ ] degrade gracefully: an unlabelled transcript is still usable
"""

from pathlib import Path

from ..models import SpeakerTurn


def diarize(audio_path: Path, num_speakers: int | None = None) -> list[SpeakerTurn]:
    raise NotImplementedError("TODO(phase-0)")
