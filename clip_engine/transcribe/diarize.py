"""Who spoke when (pyannote).

Needs a HuggingFace token and acceptance of the model terms (three gated
repos: speaker-diarization-3.1, segmentation-3.0, speaker-diarization-
community-1). Slow on CPU -- about an hour for a 65-minute episode.

diarize() raises on failure. diarize_cached() is what stages call: it caches
the turns the moment they exist, and on failure logs and returns no turns,
because an unlabelled transcript is still usable.
"""

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from ..models import SpeakerTurn

log = logging.getLogger(__name__)


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


def diarize_cached(
    audio_path: Path,
    cache_path: Path,
    num_speakers: int | None = None,
    force: bool = False,
) -> list[SpeakerTurn]:
    if cache_path.exists() and not force:
        return [SpeakerTurn(**t) for t in json.loads(cache_path.read_text())]

    try:
        turns = diarize(audio_path, num_speakers)
    except Exception as exc:  # noqa: BLE001 -- any failure degrades to unlabelled
        log.warning("diarization skipped (%s); transcript will be unlabelled", exc)
        return []

    cache_path.write_text(json.dumps([asdict(t) for t in turns]))
    return turns
