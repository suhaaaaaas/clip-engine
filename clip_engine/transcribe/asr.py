"""Speech to text with word-level timestamps (faster-whisper, local).

Word timings matter twice: captions are built from them, and clip boundaries
snap to them so no clip starts mid-word.

Results are cached by (audio hash, vocabulary, model size), so re-running is
instant and changing any of them never returns a stale transcript.

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

#Setting model size to large-v3 for now. 
#large-v3-turbo seems promising with similar performance as large but faster processing and less VRAM usage
DEFAULT_MODEL_SIZE = "large-v3"

def transcribe(
    audio_path: Path,
    model_size: str = DEFAULT_MODEL_SIZE,
    vocabulary: list[str] | None = None,
) -> tuple[list[Segment], str | None]:
    """Return (segments, detected language)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    initial_prompt = ", ".join(vocabulary) if vocabulary else None
    raw_segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        initial_prompt=initial_prompt,
        vad_filter=True,
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
    """Unlabelled transcript, cached at cache_dir/asr_<model>_<hash8>.json.

    The hash covers the audio and the vocabulary prompt, since both change
    what Whisper writes.
    """
    key = hashlib.sha1(f"{audio_hash(audio_path)}\n{vocabulary or []}".encode()).hexdigest()
    cache_path = cache_dir / f"asr_{model_size}_{key[:8]}.json"
    if cache_path.exists() and not force:
        return Transcript.from_json(cache_path)

    segments, language = transcribe(audio_path, model_size, vocabulary)
    transcript = Transcript(episode_id, segments, language=language, model=model_size)
    transcript.to_json(cache_path)
    return transcript
