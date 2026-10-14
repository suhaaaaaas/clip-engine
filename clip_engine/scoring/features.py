"""Per-window signals. Each returns a 0-1 score so weights stay comparable.

All pure: segments (and, for audio, a precomputed RMS envelope) in, a float
out. Start with the cheap text signals and only lean on audio once the text
ones are measured -- you cannot tell whether a feature helps without the eval
harness.

TODO(phase-1): the saturation constants below are first guesses, to revisit
once golden labels exist. Options per feature: keep them, set them from the
distribution on episodes 1-2 (e.g. the 90th percentile maps to 1.0), or drop
saturation and rank-normalize within each episode.
  - TAKE_MARKER_SATURATION = 3 markers
  - CHURN_SATURATION = 0.5 speaker changes per second
  - DENSITY_SATURATION = 2x the episode's mean words per second
  - LOUD_SHARE_SATURATION = 20% of frames above the loudness threshold
"""

import re
from functools import cache
from itertools import pairwise

import numpy as np

from ..models import Segment, Transcript

TAKE_MARKER_SATURATION = 3
CHURN_SATURATION = 0.5
DENSITY_SATURATION = 2.0
LOUD_SHARE_SATURATION = 0.2
LOUDNESS_MADS = 3.0


def _text(segments: list[Segment]) -> str:
    return " ".join(s.text for s in segments).lower().replace("’", "'")


@cache
def _pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term.lower())}(?!\w)")


def keyword_hit(segments: list[Segment], keywords: dict[str, list[str]]) -> float:
    """Fraction of keyword groups with at least one term in the window."""
    if not keywords:
        return 0.0
    text = _text(segments)
    touched = sum(any(_pattern(t).search(text) for t in terms) for terms in keywords.values())
    return touched / len(keywords)


def take_markers(segments: list[Segment], phrases: list[str]) -> float:
    """Phrases that open a take or an argument, saturating at a few hits."""
    text = _text(segments)
    count = sum(len(_pattern(p).findall(text)) for p in phrases)
    return min(1.0, count / TAKE_MARKER_SATURATION)


def speaker_churn(segments: list[Segment]) -> float:
    """Speaker changes per second, a proxy for argument. Unlabelled segments are skipped."""
    if not segments:
        return 0.0
    speakers = [s.speaker for s in segments if s.speaker]
    changes = sum(a != b for a, b in pairwise(speakers))
    seconds = (segments[-1].end_ms - segments[0].start_ms) / 1000
    if seconds <= 0:
        return 0.0
    return min(1.0, changes / seconds / CHURN_SATURATION)


def _word_count(segments: list[Segment]) -> int:
    return sum(len(s.words) if s.words else len(s.text.split()) for s in segments)


def episode_words_per_second(transcript: Transcript) -> float:
    segs = transcript.segments
    if not segs:
        return 0.0
    seconds = (segs[-1].end_ms - segs[0].start_ms) / 1000
    return _word_count(segs) / seconds if seconds > 0 else 0.0


def density(segments: list[Segment], episode_mean_wps: float) -> float:
    """Words per second relative to this episode's own mean."""
    if not segments or episode_mean_wps <= 0:
        return 0.0
    seconds = (segments[-1].end_ms - segments[0].start_ms) / 1000
    if seconds <= 0:
        return 0.0
    ratio = _word_count(segments) / seconds / episode_mean_wps
    return min(1.0, ratio / DENSITY_SATURATION)


def energy_threshold(envelope: np.ndarray, mads: float = LOUDNESS_MADS) -> float:
    """Loud = this many median absolute deviations above the episode's median."""
    median = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - median)))
    return median + mads * mad


def audio_energy(
    envelope: np.ndarray, hop_ms: int, threshold: float, start_ms: int, end_ms: int
) -> float:
    """Share of the window's frames above the loudness threshold (laughter, shouting)."""
    frames = envelope[start_ms // hop_ms : end_ms // hop_ms]
    if frames.size == 0:
        return 0.0
    loud_share = float(np.mean(frames > threshold))
    return min(1.0, loud_share / LOUD_SHARE_SATURATION)
