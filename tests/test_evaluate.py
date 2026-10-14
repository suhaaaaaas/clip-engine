"""The eval harness. Test this first -- every later decision leans on it."""

from pathlib import Path

import pytest

from clip_engine.models import ClipCandidate
from clip_engine.scoring.evaluate import (
    EvalRow,
    agreement,
    hits_at_k,
    iou,
    load_golden,
    precision_at_k,
    report,
)

FIXTURES = Path(__file__).parent / "fixtures"


def clip(start_s: float, end_s: float, episode: str = "ep") -> ClipCandidate:
    return ClipCandidate(episode, round(start_s * 1000), round(end_s * 1000), score=1.0)


def write_csv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "ep_tester.csv"
    path.write_text(body)
    return path


# load_golden


def test_load_golden_parses_fixture():
    clips = load_golden(FIXTURES / "synthetic_labels_a.csv")
    assert [(c.start_ms, c.end_ms) for c in clips] == [(10_000, 40_000), (100_000, 130_000), (300_000, 330_000)]
    assert clips[0].text == "synthetic: host doubles down"
    assert clips[0].episode_id == "synthetic_labels"


def test_load_golden_rejects_unknown_type_with_row_number():
    with pytest.raises(ValueError, match=r"synthetic_malformed\.csv:3"):
        load_golden(FIXTURES / "synthetic_malformed.csv")


def test_load_golden_rejects_missing_column(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        load_golden(write_csv(tmp_path, "start_s,end_s,reason\n1,30,x\n"))


def test_load_golden_rejects_non_numeric_time(tmp_path):
    with pytest.raises(ValueError, match=":2"):
        load_golden(write_csv(tmp_path, "start_s,end_s,reason,type\nabc,30,x,joke\n"))


def test_load_golden_rejects_end_before_start(tmp_path):
    with pytest.raises(ValueError, match="end_s"):
        load_golden(write_csv(tmp_path, "start_s,end_s,reason,type\n40,30,x,joke\n"))


def test_load_golden_rejects_short_row(tmp_path):
    with pytest.raises(ValueError):
        load_golden(write_csv(tmp_path, "start_s,end_s,reason,type\n10\n"))


# iou


def test_iou_identical_disjoint_and_partial():
    assert iou(clip(0, 10), clip(0, 10)) == 1.0
    assert iou(clip(0, 10), clip(10, 20)) == 0.0
    assert iou(clip(0, 10), clip(5, 15)) == pytest.approx(5 / 15)


# agreement


def test_agreement_identical_sets():
    a = load_golden(FIXTURES / "synthetic_labels_a.csv")
    assert agreement(a, a) == 1.0


def test_agreement_disjoint_sets():
    assert agreement([clip(0, 30)], [clip(100, 130)]) == 0.0


def test_agreement_fixtures_is_symmetric():
    a = load_golden(FIXTURES / "synthetic_labels_a.csv")
    b = load_golden(FIXTURES / "synthetic_labels_b.csv")
    assert agreement(a, b) == pytest.approx(2 / 3)
    assert agreement(b, a) == agreement(a, b)


def test_agreement_matches_by_timestamp_not_id():
    assert agreement([clip(0, 30, episode="x")], [clip(1, 30, episode="y")]) == 1.0


def test_agreement_is_one_to_one():
    # Two of b's picks overlap the same pick in a; only one can match it.
    a = [clip(0, 30)]
    b = [clip(0, 30), clip(1, 30)]
    assert agreement(a, b) == pytest.approx(2 * 1 / 3)


def test_agreement_of_two_empty_sets_is_zero():
    assert agreement([], []) == 0.0


# precision_at_k


def test_overlap_above_half_is_a_hit():
    golden = [clip(0, 30)]
    assert hits_at_k([clip(0, 20)], golden, k=1) == 1  # IoU 0.67


def test_overlap_of_exactly_half_is_not_a_hit():
    golden = [clip(0, 20)]
    assert hits_at_k([clip(0, 10)], golden, k=1) == 0  # IoU 0.5


def test_precision_at_k_ignores_predictions_beyond_k():
    golden = [clip(100, 130)]
    predicted = [clip(0, 30), clip(100, 130)]
    assert precision_at_k(predicted, golden, k=1) == 0.0
    assert precision_at_k(predicted, golden, k=2) == 0.5


def test_duplicate_predictions_hit_a_golden_clip_once():
    golden = [clip(0, 30)]
    predicted = [clip(0, 30), clip(1, 30)]
    assert hits_at_k(predicted, golden, k=2) == 1


def test_precision_at_k_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        precision_at_k([], [], k=0)


# report


def test_report_has_one_row_per_episode():
    text = report(
        [
            EvalRow("ep1", k=10, hits=5, golden_count=10, agreement=0.4),
            EvalRow("ep3", k=10, hits=3, golden_count=10, held_out=True),
        ]
    )
    lines = text.splitlines()
    assert len(lines) == 4
    assert "ep1" in lines[2] and "0.50" in lines[2] and "0.40" in lines[2]
    assert "ep3" in lines[3] and "held out" in lines[3]
