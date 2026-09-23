"""Score every window and rank them.

Pure function: transcript in, ranked candidates out. No I/O, no framework
imports, so it can be swept over weight configurations offline.

TODO(phase-1):
  - [ ] score_episode(transcript, audio_path, config) -> list[ClipCandidate]
  - [ ] keep per-feature contributions in ClipCandidate.breakdown
  - [ ] weights come from the show config, never hardcoded here
"""

from ..config import ShowConfig
from ..models import ClipCandidate, Transcript


def score_episode(
    transcript: Transcript,
    audio_path: str | None = None,
    config: ShowConfig | None = None,
) -> list[ClipCandidate]:
    raise NotImplementedError("TODO(phase-1)")
