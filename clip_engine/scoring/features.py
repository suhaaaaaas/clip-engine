"""Per-window signals. Each returns a 0-1 score so weights stay comparable.

Start with the cheap text signals and only add audio once the text ones are
measured -- you cannot tell whether a feature helps without the eval harness.

TODO(phase-1):
  - [ ] keyword_hit: show vocabulary and topic terms in the window
  - [ ] take_markers: "I'll say it", "nobody's talking about", disagreement cues
  - [ ] speaker_churn: turns per second, a proxy for argument
  - [ ] density: words per second vs this episode's own mean
  - [ ] audio_energy: RMS spikes over a rolling baseline (laughter, shouting)
  - [ ] every feature gets a unit test with a hand-built transcript
"""

from ..models import Segment


def keyword_hit(segments: list[Segment], keywords: dict[str, list[str]]) -> float:
    raise NotImplementedError("TODO(phase-1)")


def take_markers(segments: list[Segment]) -> float:
    raise NotImplementedError("TODO(phase-1)")


def speaker_churn(segments: list[Segment]) -> float:
    raise NotImplementedError("TODO(phase-1)")


def density(segments: list[Segment], episode_mean_wps: float) -> float:
    raise NotImplementedError("TODO(phase-1)")


def audio_energy(audio_path: str, start_ms: int, end_ms: int) -> float:
    raise NotImplementedError("TODO(phase-1): RMS over a rolling baseline")
