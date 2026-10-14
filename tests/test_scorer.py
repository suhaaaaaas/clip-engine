"""The scorer: pure, weights from config, per-feature breakdown kept."""

import builtins

import numpy as np
import pytest

from clip_engine.config import ShowConfig
from clip_engine.models import Segment, Transcript
from clip_engine.scoring.scorer import AudioEnvelope, score_episode

ZERO = {"keyword_hit": 0.0, "take_markers": 0.0, "speaker_churn": 0.0, "density": 0.0, "audio_energy": 0.0}


def episode(hot_text: str = "they have to trade him, no way") -> Transcript:
    """Five minutes of 5-second segments; 100-130s is the hot stretch."""
    segs = []
    for i in range(60):
        start = i * 5000
        hot = 100_000 <= start < 130_000
        text = hot_text if hot else "just talking about the weather"
        speaker = f"S{i % 2}" if hot else "S0"
        segs.append(Segment(start, start + 5000, text, speaker))
    return Transcript("ep", segs)


def config(**weights: float) -> ShowConfig:
    return ShowConfig(
        slug="test",
        weights={**ZERO, **weights},
        keywords={"trades": ["trade"]},
        take_markers=["no way"],
        max_per_10_min=5,
        min_score=0.0,
    )


def test_top_candidate_follows_the_weighted_feature():
    top = score_episode(episode(), config(keyword_hit=1.0))[0]
    assert top.start_ms < 130_000 and top.end_ms > 100_000
    assert top.score > 0


def test_breakdown_holds_each_features_contribution_and_sums_to_score():
    for c in score_episode(episode(), config(keyword_hit=1.0, speaker_churn=1.0)):
        assert set(c.breakdown) == {"keyword_hit", "take_markers", "speaker_churn", "density"}
        assert sum(c.breakdown.values()) == pytest.approx(c.score)


def test_zero_weight_feature_has_no_effect():
    cfg = config(keyword_hit=1.0)
    with_markers = score_episode(episode("trade talk, no way no way"), cfg)
    without = score_episode(episode("trade talk"), cfg)
    assert [(c.start_ms, c.end_ms, c.score) for c in with_markers] == [
        (c.start_ms, c.end_ms, c.score) for c in without
    ]


def test_candidates_carry_text_speakers_and_episode():
    top = score_episode(episode(), config(keyword_hit=1.0, speaker_churn=1.0))[0]
    assert "trade" in top.text
    assert top.speakers == ["S0", "S1"]
    assert top.episode_id == "ep"


def test_audio_energy_used_only_when_an_envelope_is_given():
    envelope = np.full(3000, 0.1)
    envelope[2000:2300] = 1.0  # loud at 200-230s with 100ms hops
    cfg = config(audio_energy=1.0)
    top = score_episode(episode(), cfg, AudioEnvelope(envelope, 100))[0]
    assert "audio_energy" in top.breakdown
    assert top.start_ms < 230_000 and top.end_ms > 200_000
    assert all("audio_energy" not in c.breakdown for c in score_episode(episode(), cfg))


def test_volume_controls_from_config_apply():
    cfg = config(keyword_hit=1.0)
    cfg.max_per_10_min = 1
    assert len(score_episode(episode(), cfg)) == 1
    cfg.max_per_10_min, cfg.min_score = 5, 0.5
    assert all(c.score >= 0.5 for c in score_episode(episode(), cfg))


def test_empty_transcript_scores_nothing():
    assert score_episode(Transcript("ep", []), config(keyword_hit=1.0)) == []


def test_scorer_does_no_file_io(monkeypatch):
    def no_io(*args, **kwargs):
        raise AssertionError("scorer opened a file")

    monkeypatch.setattr(builtins, "open", no_io)
    assert score_episode(episode(), config(keyword_hit=1.0))
