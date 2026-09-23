"""Candidate windows and overlap suppression.

TODO(phase-1):
  - [ ] windows never start or end mid-segment
  - [ ] windows respect min/max duration
  - [ ] non-max suppression keeps the best of an overlapping cluster
  - [ ] an empty transcript yields no candidates instead of raising
"""

import pytest


@pytest.mark.skip(reason="TODO(phase-1)")
def test_windows_snap_to_segment_boundaries():
    raise NotImplementedError
