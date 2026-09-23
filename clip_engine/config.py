"""Per-show configuration.

A show is the unit of tuning: its vocabulary, its seat map, its scoring
weights. Nothing here should be global -- if it differs between shows, it
belongs in a show config file under ``shows/<slug>.toml``.

TODO(phase-1):
  - [ ] define ShowConfig (slug, display name, source url, speaker names)
  - [ ] scoring weights + keyword groups, loaded from TOML
  - [ ] vocabulary list for caption correction (host names, recurring topics)
  - [ ] load_show(slug) with sane defaults so a new show works unconfigured
TODO(phase-2):
  - [ ] seat map: speaker label -> face track id, per camera angle
"""

from dataclasses import dataclass, field
from pathlib import Path

SHOWS_DIR = Path("shows")


@dataclass
class ShowConfig:
    slug: str
    display_name: str = ""
    # Whisper does better with a vocabulary hint; also used to fix captions.
    vocabulary: list[str] = field(default_factory=list)
    # None lets diarization guess; pinning it sharply improves turn boundaries.
    num_speakers: int | None = None
    weights: dict[str, float] = field(default_factory=dict)
    keywords: dict[str, list[str]] = field(default_factory=dict)


def load_show(slug: str) -> ShowConfig:
    """Load shows/<slug>.toml into a ShowConfig."""
    raise NotImplementedError("TODO(phase-1): read TOML, merge over defaults")
