"""Every feature gets a hand-built transcript. All return 0-1."""

import wave

import numpy as np
import pytest

from clip_engine.ingest.audio import rms_envelope
from clip_engine.models import Segment, Transcript, Word
from clip_engine.scoring import features as f


def seg(start_s: float, end_s: float, text: str, speaker: str | None = None) -> Segment:
    return Segment(round(start_s * 1000), round(end_s * 1000), text, speaker)


# keyword_hit

KEYWORDS = {"contracts": ["trade", "max deal"], "rings": ["championship"]}


def test_keyword_hit_is_fraction_of_groups_touched():
    assert f.keyword_hit([seg(0, 5, "They should TRADE him")], KEYWORDS) == 0.5
    assert f.keyword_hit([seg(0, 5, "max deal"), seg(5, 9, "a championship")], KEYWORDS) == 1.0


def test_keyword_hit_needs_whole_words():
    assert f.keyword_hit([seg(0, 5, "the trademark issue")], KEYWORDS) == 0.0


def test_keyword_hit_with_no_keywords_is_zero():
    assert f.keyword_hit([seg(0, 5, "trade")], {}) == 0.0


# take_markers

PHRASES = ["i'll say it", "no way"]


def test_take_markers_counts_and_saturates():
    assert f.take_markers([seg(0, 5, "nothing here")], PHRASES) == 0.0
    assert f.take_markers([seg(0, 5, "I'll say it.")], PHRASES) == pytest.approx(1 / 3)
    many = [seg(i, i + 1, "No way! no way, no way. I'll say it") for i in range(3)]
    assert f.take_markers(many, PHRASES) == 1.0


def test_take_markers_handles_curly_apostrophes():
    assert f.take_markers([seg(0, 5, "I’ll say it")], PHRASES) > 0


# speaker_churn


def test_speaker_churn_rises_with_turn_changes():
    calm = [seg(0, 10, "a", "S0"), seg(10, 20, "b", "S0")]
    heated = [seg(2 * i, 2 * i + 2, "x", f"S{i % 2}") for i in range(10)]
    assert f.speaker_churn(calm) == 0.0
    assert f.speaker_churn(heated) > 0.8


def test_speaker_churn_ignores_unlabelled_segments():
    assert f.speaker_churn([seg(0, 5, "a", "S0"), seg(5, 6, "b"), seg(6, 10, "c", "S0")]) == 0.0


def test_speaker_churn_empty_is_zero():
    assert f.speaker_churn([]) == 0.0


# density


def test_density_relative_to_episode_mean():
    fast = [Segment(0, 2000, "", words=[Word(0, 100, "w")] * 8)]  # 4 wps
    assert f.density(fast, episode_mean_wps=2.0) == 1.0
    assert f.density(fast, episode_mean_wps=4.0) == 0.5


def test_density_falls_back_to_text_and_guards_zero_mean():
    assert f.density([seg(0, 2, "one two three four")], episode_mean_wps=2.0) == 0.5
    assert f.density([seg(0, 2, "words")], episode_mean_wps=0.0) == 0.0


def test_episode_words_per_second():
    t = Transcript("ep", [seg(0, 2, "a b"), seg(2, 4, "c d e f")])
    assert f.episode_words_per_second(t) == 1.5


# audio_energy


def test_audio_energy_is_share_of_loud_frames():
    envelope = np.full(100, 0.1)
    envelope[50:60] = 1.0  # loud stretch at 5.0-6.0s with 100ms hops
    threshold = f.energy_threshold(envelope)
    assert f.audio_energy(envelope, 100, threshold, 0, 4000) == 0.0
    assert f.audio_energy(envelope, 100, threshold, 5000, 6000) == 1.0
    # 10 loud frames out of 100 is a 10% share, half of saturation.
    assert f.audio_energy(envelope, 100, threshold, 0, 10_000) == pytest.approx(0.5)


def test_audio_energy_outside_envelope_is_zero():
    envelope = np.ones(10)
    assert f.audio_energy(envelope, 100, 0.5, 5000, 6000) == 0.0


def test_rms_envelope_reads_a_wav(tmp_path):
    path = tmp_path / "tone.wav"
    rate = 16_000
    samples = np.zeros(rate, dtype=np.int16)
    samples[rate // 2 :] = 10_000  # quiet first half, loud second half
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())

    env = rms_envelope(path, hop_ms=100)
    assert len(env) == 10
    assert env[0] == 0.0
    assert env[-1] == pytest.approx(10_000 / 32_768, rel=1e-3)
