"""Burned-in captions from word timings.

Corrections come from the show vocabulary: names the ASR mangles get fixed
here rather than by retyping every caption.

TODO(phase-2):
  - [ ] group words into caption lines (max ~3 words per line, karaoke style)
  - [ ] apply vocabulary corrections (fuzzy match against known names)
  - [ ] emit ASS/SRT with per-word highlight timing
  - [ ] keep captions clear of the bottom UI overlay on each platform
"""

from ..models import Word


def build_caption_file(words: list[Word], vocabulary: list[str]) -> str:
    raise NotImplementedError("TODO(phase-2)")
