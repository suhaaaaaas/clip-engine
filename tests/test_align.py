"""Speaker alignment: the join between ASR segments and diarization turns.

Write these before align.assign_speakers. They are fast, they need no audio,
and they catch the bug that silently mislabels every clip.

TODO(phase-0):
  - [ ] majority overlap wins
  - [ ] no turns -> every segment stays unlabelled
  - [ ] a segment in a diarization gap stays None
  - [ ] one long turn covering several segments labels all of them
  - [ ] timings and text are never modified
"""

import pytest

from clip_engine.models import Segment, SpeakerTurn
from clip_engine.transcribe.align import assign_speakers


@pytest.mark.skip(reason="TODO(phase-0): implement assign_speakers")
def test_majority_overlap_wins():
    segments = [Segment(0, 1000, "x")]
    turns = [SpeakerTurn(0, 300, "S0"), SpeakerTurn(300, 1200, "S1")]
    assert assign_speakers(segments, turns)[0].speaker == "S1"
