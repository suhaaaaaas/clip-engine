"""Speech to text with word-level timestamps (faster-whisper, local).

Word timings matter twice: captions are built from them, and clip boundaries
snap to them so no clip starts mid-word.

Results are cached by (audio hash, model size), so re-running is instant and
switching model sizes never returns a stale transcript.

TODO(phase-1): pick the default model size. Options:
  - small (current): ~15-30 min per 90-min episode on a laptop CPU; good
    enough to evaluate clip picks.
  - medium: noticeably better on names and crosstalk, roughly 2x slower.
  - large-v3: best captions, impractical on CPU; worth it once a GPU worker
    exists (post-Phase 3).
  Decide after transcribing one episode with two sizes and diffing captions.
"""

import hashlib
from pathlib import Path

from ..models import Segment, Transcript, Word

DEFAULT_MODEL_SIZE = "small"


def transcribe(
    audio_path: Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    vocabulary: list[str] | None = None,
) -> tuple[list[Segment], str | None]:
    """Return (segments, detected language)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    initial_prompt = ", ".join(vocabulary) if vocabulary else None
    raw_segments, info = model.transcribe(
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
    return segments, getattr(info, "language", None)


def audio_hash(audio_path: Path) -> str:
    with audio_path.open("rb") as f:
        return hashlib.file_digest(f, "sha1").hexdigest()


def transcribe_cached(
    audio_path: Path,
    episode_id: str,
    cache_dir: Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    vocabulary: list[str] | None = None,
    force: bool = False,
) -> Transcript:
    """Unlabelled transcript, cached at cache_dir/asr_<model>_<hash8>.json."""
    cache_path = cache_dir / f"asr_{model_size}_{audio_hash(audio_path)[:8]}.json"
    if cache_path.exists() and not force:
        return Transcript.from_json(cache_path)

    segments, language = transcribe(audio_path, model_size, vocabulary)
    transcript = Transcript(episode_id, segments, language=language, model=model_size)
    transcript.to_json(cache_path)
    return transcript
