"""The eval harness. Test this first -- every later decision leans on it.

TODO(phase-0):
  - [ ] load_golden parses a label CSV, rejects malformed rows loudly
  - [ ] agreement() is 1.0 for identical label sets, 0.0 for disjoint ones
TODO(phase-1):
  - [ ] a predicted clip overlapping a golden clip by >50% counts as a hit
  - [ ] precision_at_k ignores predictions beyond k
"""

import pytest


@pytest.mark.skip(reason="TODO(phase-0)")
def test_agreement_identical_sets():
    raise NotImplementedError
