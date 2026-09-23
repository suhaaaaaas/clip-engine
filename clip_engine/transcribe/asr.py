"""Speech to text with word-level timestamps (faster-whisper, local).

Word timings matter twice: captions are built from them, and clip boundaries
snap to them so no clip starts mid-word.

TODO(phase-0):
  - [ ] transcribe(audio_path, model_size, vocabulary) -> list[Segment]
  - [ ] pass the show vocabulary as an initial prompt (helps proper nouns)
  - [ ] cache by (audio hash, model) so re-runs are instant
  - [ ] pick a model size: start with 'small' for speed, compare to 'large-v3'
"""

from pathlib import Path

from ..models import Segment


def transcribe(
    audio_path: Path,
    model_size: str = "small",
    vocabulary: list[str] | None = None,
) -> list[Segment]:
    raise NotImplementedError("TODO(phase-0)")
