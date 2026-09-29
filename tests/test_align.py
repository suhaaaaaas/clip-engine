"""Speaker alignment: the join between ASR segments and diarization turns.

Write these before align.assign_speakers. They are fast, they need no audio,
and they catch the bug that silently mislabels every clip.
"""

from clip_engine.models import Segment, SpeakerTurn
from clip_engine.transcribe.align import assign_speakers


def test_majority_overlap_wins():
    segments = [Segment(0, 1000, "x")]
    turns = [SpeakerTurn(0, 300, "S0"), SpeakerTurn(300, 1200, "S1")]
    assert assign_speakers(segments, turns)[0].speaker == "S1"


def test_no_turns_stays_unlabelled():
    segments = [Segment(0, 1000, "x"), Segment(1000, 2000, "y")]
    result = assign_speakers(segments, [])
    assert [s.speaker for s in result] == [None, None]


def test_gap_stays_none():
    segments = [Segment(0, 1000, "x")]
    turns = [SpeakerTurn(2000, 3000, "S0")]
    assert assign_speakers(segments, turns)[0].speaker is None


def test_one_turn_covers_many_segments():
    segments = [Segment(0, 500, "a"), Segment(500, 1000, "b"), Segment(1000, 1500, "c")]
    turns = [SpeakerTurn(0, 1500, "S0")]
    result = assign_speakers(segments, turns)
    assert [s.speaker for s in result] == ["S0", "S0", "S0"]


def test_timings_and_text_never_modified():
    segments = [Segment(0, 1000, "hello world")]
    turns = [SpeakerTurn(0, 1000, "S0")]
    result = assign_speakers(segments, turns)
    assert result[0].start_ms == 0
    assert result[0].end_ms == 1000
    assert result[0].text == "hello world"
