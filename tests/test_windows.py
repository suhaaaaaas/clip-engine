"""Candidate windows, overlap suppression, and the volume cap."""

from clip_engine.models import ClipCandidate, Segment, Transcript
from clip_engine.scoring.windows import candidate_windows, cap_volume, non_max_suppress


def transcript_of(seconds_per_segment: int, count: int) -> Transcript:
    step = seconds_per_segment * 1000
    segs = [Segment(i * step, (i + 1) * step, f"s{i}") for i in range(count)]
    return Transcript("ep", segs)


def cand(start_s: float, end_s: float, score: float) -> ClipCandidate:
    return ClipCandidate("ep", round(start_s * 1000), round(end_s * 1000), score)


def test_windows_snap_to_segment_boundaries():
    t = transcript_of(7, 30)
    starts = {s.start_ms for s in t.segments}
    ends = {s.end_ms for s in t.segments}
    windows = candidate_windows(t, min_s=20, max_s=60)
    assert windows
    assert all(start in starts and end in ends for start, end in windows)


def test_windows_respect_min_and_max_duration():
    windows = candidate_windows(transcript_of(7, 30), min_s=20, max_s=60)
    assert all(20_000 <= end - start <= 60_000 for start, end in windows)
    # 21s (3 segments) through 56s (8 segments), from every start that fits.
    assert (0, 21_000) in windows and (0, 56_000) in windows
    assert (0, 14_000) not in windows and (0, 63_000) not in windows


def test_single_segment_longer_than_max_yields_nothing():
    t = Transcript("ep", [Segment(0, 90_000, "monologue")])
    assert candidate_windows(t, min_s=20, max_s=60) == []


def test_empty_transcript_yields_no_candidates():
    assert candidate_windows(Transcript("ep", []), 20, 60) == []
    assert non_max_suppress([]) == []
    assert cap_volume([], max_per_10_min=2, min_score=0.0) == []


def test_nms_keeps_the_best_of_an_overlapping_cluster():
    best = cand(10, 40, 0.9)
    cluster = [cand(12, 42, 0.5), best, cand(8, 38, 0.7)]
    separate = cand(200, 230, 0.3)
    assert non_max_suppress(cluster + [separate], iou_threshold=0.4) == [best, separate]


def test_nms_keeps_lightly_overlapping_windows():
    a, b = cand(0, 30, 0.9), cand(25, 55, 0.8)  # IoU 5/55
    assert non_max_suppress([a, b], iou_threshold=0.4) == [a, b]


def test_cap_drops_low_scores_and_limits_each_ten_minutes():
    first_ten = [cand(60 * i, 60 * i + 30, score) for i, score in enumerate([0.9, 0.8, 0.7, 0.1])]
    next_ten = [cand(700, 730, 0.6)]
    kept = cap_volume(first_ten + next_ten, max_per_10_min=2, min_score=0.2)
    assert [c.score for c in kept] == [0.9, 0.8, 0.6]


def test_cap_scales_with_episode_length_not_a_fixed_quota():
    def episode(minutes: int) -> list[ClipCandidate]:
        return [cand(60 * m, 60 * m + 30, 0.9) for m in range(minutes)]

    assert len(cap_volume(episode(20), 2, 0.0)) == 4
    assert len(cap_volume(episode(60), 2, 0.0)) == 12
