"""Who spoke when (pyannote).

Needs a HuggingFace token and acceptance of the model terms. Slow on CPU --
expect several minutes for a long episode.

Callers should treat this as best-effort: catch its exceptions and fall back
to an unlabelled transcript rather than failing the whole ASR stage, since a
missing token or a first-pass skip shouldn't block transcription.

TODO(phase-1):
  - [ ] cache results; this is the slowest local step
"""

import os
from pathlib import Path

from ..models import SpeakerTurn


def diarize(audio_path: Path, num_speakers: int | None = None) -> list[SpeakerTurn]:
    from pyannote.audio import Pipeline

    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACE_TOKEN not set -- accept the model terms at "
            "huggingface.co/pyannote/speaker-diarization-3.1 and set the "
            "token in .env"
        )

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    output = pipeline(str(audio_path), num_speakers=num_speakers)

    # exclusive_speaker_diarization has overlapping speech already resolved to
    # one speaker per turn, which is what assign_speakers wants.
    turns = [
        SpeakerTurn(start_ms=round(turn.start * 1000), end_ms=round(turn.end * 1000), speaker=speaker)
        for turn, _, speaker in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t.start_ms)
    return turns
