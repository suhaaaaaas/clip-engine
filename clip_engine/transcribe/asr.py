"""Speech to text with word-level timestamps (faster-whisper, local).

Word timings matter twice: captions are built from them, and clip boundaries
snap to them so no clip starts mid-word.

Results are cached by (audio hash, vocabulary, model size), so re-running is
instant and changing any of them never returns a stale transcript.

Runs on an Nvidia GPU when one is usable (needs `pip install -e ".[gpu]"`),
otherwise falls back to CPU with a smaller model:
  - GPU: large-v3, float16. ~5-10 min per 90-min episode on an RTX 3070 Ti.
  - CPU: small, int8. ~15-30 min per 90-min episode on a laptop.

TODO(phase-1): settle the GPU model. large-v3-turbo looks promising: close to
large-v3 on English, faster, less VRAM. Transcribe one episode with both and
diff the captions, especially names.
"""

import ctypes
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple

from ..models import Segment, Transcript, Word

log = logging.getLogger(__name__)

GPU_MODEL_SIZE = "large-v3"
CPU_MODEL_SIZE = "small"

# Libraries ctranslate2 loads by name on first GPU use. If they are missing it
# only fails mid-transcription, so check up front and fall back to CPU instead.
_CUDA_LIBS = {
    "win32": ["cublas64_12.dll", "cudnn64_9.dll"],
    "linux": ["libcublas.so.12", "libcudnn.so.9"],
}


class Device(NamedTuple):
    device: str
    compute_type: str
    model_size: str


CPU = Device("cpu", "int8", CPU_MODEL_SIZE)
GPU = Device("cuda", "float16", GPU_MODEL_SIZE)


def _add_cuda_dlls_on_windows() -> None:
    """Make pip-installed CUDA DLLs (nvidia-cublas-cu12, nvidia-cudnn-cu12) findable.

    The wheels put the DLLs in site-packages/nvidia/*/bin, which Windows does
    not search. ctranslate2 loads cuBLAS lazily by name, so the folders go on
    PATH as well as add_dll_directory.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        for bin_dir in Path(root).glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def _cuda_device_count() -> int:
    try:
        import ctranslate2
    except ImportError:
        return 0
    return ctranslate2.get_cuda_device_count()


def _missing_cuda_libs() -> list[str]:
    missing = []
    for name in _CUDA_LIBS.get(sys.platform, []):
        try:
            ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


def pick_device() -> Device:
    """GPU when an Nvidia card and the CUDA libraries are both there, else CPU."""
    if _cuda_device_count() == 0:
        log.info("no CUDA GPU found; transcribing on CPU with %s", CPU_MODEL_SIZE)
        return CPU
    _add_cuda_dlls_on_windows()
    missing = _missing_cuda_libs()
    if missing:
        log.warning(
            "GPU found but %s won't load; transcribing on CPU with %s. "
            'Install the CUDA libraries with: pip install -e ".[gpu]"',
            ", ".join(missing),
            CPU_MODEL_SIZE,
        )
        return CPU
    log.info("transcribing on GPU with %s", GPU_MODEL_SIZE)
    return GPU


def transcribe(
    audio_path: Path,
    model_size: str,
    vocabulary: list[str] | None = None,
    device: Device = CPU,
) -> tuple[list[Segment], str | None]:
    """Return (segments, detected language)."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device.device, compute_type=device.compute_type)
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
    model_size: str | None = None,
    vocabulary: list[str] | None = None,
    force: bool = False,
) -> Transcript:
    """Unlabelled transcript, cached at cache_dir/asr_<model>_<hash8>.json.

    model_size=None picks by hardware: large-v3 on GPU, small on CPU. The hash
    covers the audio and the vocabulary prompt, since both change what Whisper
    writes.
    """
    device = pick_device()
    model_size = model_size or device.model_size
    key = hashlib.sha1(f"{audio_hash(audio_path)}\n{vocabulary or []}".encode()).hexdigest()
    cache_path = cache_dir / f"asr_{model_size}_{key[:8]}.json"
    if cache_path.exists() and not force:
        return Transcript.from_json(cache_path)

    segments, language = transcribe(audio_path, model_size, vocabulary, device)
    transcript = Transcript(episode_id, segments, language=language, model=model_size)
    transcript.to_json(cache_path)
    return transcript
