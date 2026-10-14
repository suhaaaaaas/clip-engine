"""Score every window and rank them.

Pure function: transcript (plus an optional loudness envelope) in, ranked
candidates out. No I/O, no config-file reads, so it can be swept over weight
configurations offline. Weights come only from the ShowConfig passed in.
"""

import bisect
from typing import NamedTuple

import numpy as np

from ..config import ShowConfig
from ..models import ClipCandidate, Transcript
from . import features
from .windows import candidate_windows, cap_volume, non_max_suppress


class AudioEnvelope(NamedTuple):
    rms: np.ndarray
    hop_ms: int


def score_episode(
    transcript: Transcript,
    config: ShowConfig,
    audio: AudioEnvelope | None = None,
) -> list[ClipCandidate]:
    segs = transcript.segments
    starts = [s.start_ms for s in segs]
    ends = [s.end_ms for s in segs]
    mean_wps = features.episode_words_per_second(transcript)
    threshold = features.energy_threshold(audio.rms) if audio is not None else 0.0

    weights = {name: w for name, w in config.weights.items() if audio is not None or name != "audio_energy"}
    total_weight = sum(weights.values())

    candidates = []
    for start_ms, end_ms in candidate_windows(transcript, config.min_clip_s, config.max_clip_s):
        window = segs[bisect.bisect_left(starts, start_ms) : bisect.bisect_right(ends, end_ms)]
        values = {
            "keyword_hit": features.keyword_hit(window, config.keywords),
            "take_markers": features.take_markers(window, config.take_markers),
            "speaker_churn": features.speaker_churn(window),
            "density": features.density(window, mean_wps),
        }
        if audio is not None:
            values["audio_energy"] = features.audio_energy(
                audio.rms, audio.hop_ms, threshold, start_ms, end_ms
            )

        breakdown = {
            name: (weights.get(name, 0.0) * value / total_weight if total_weight else 0.0)
            for name, value in values.items()
        }
        candidates.append(
            ClipCandidate(
                episode_id=transcript.episode_id,
                start_ms=start_ms,
                end_ms=end_ms,
                score=sum(breakdown.values()),
                breakdown=breakdown,
                text=" ".join(s.text for s in window),
                speakers=sorted({s.speaker for s in window if s.speaker}),
            )
        )

    kept = non_max_suppress(candidates, config.nms_iou)
    return cap_volume(kept, config.max_per_10_min, config.min_score)
