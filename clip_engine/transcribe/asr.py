"""Speech to text with word-level timestamps (faster-whisper, local).

Word timings matter twice: captions are built from them, and clip boundaries
snap to them so no clip starts mid-word.

Model size: 'small' on CPU handles a 90-minute episode in roughly 15-30
minutes. 'large-v3' is much slower without a GPU -- start with 'small' while
you're evaluating clip picks, not transcript perfection.

TODO(phase-1):
  - [ ] cache by (audio hash, model) instead of relying on the caller to skip
        re-transcribing when transcript.json already exists
"""

from pathlib import Path

from ..models import Segment, Word


def transcribe(
    audio_path: Path,
    model_size: str = "small",
    vocabulary: list[str] | None = None,
) -> list[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    initial_prompt = ", ".join(vocabulary) if vocabulary else None
    raw_segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        initial_prompt=initial_prompt,
    )

    segments = []
    for seg in raw_segments:
        words = [
            Word(
                start_ms=round(w.start * 1000),
                end_ms=round(w.end * 1000),
                text=w.word.strip(),
            )
            for w in (seg.words or [])
        ]
        segments.append(
            Segment(
                start_ms=round(seg.start * 1000),
                end_ms=round(seg.end * 1000),
                text=seg.text.strip(),
                words=words,
            )
        )
    return segments
