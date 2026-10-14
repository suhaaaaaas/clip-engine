"""Per-show configuration.

A show is the unit of tuning: its vocabulary, its seat map, its scoring
weights. Nothing here should be global -- if it differs between shows, it
belongs in a show config file under ``shows/<slug>.toml``. A show with no
file still works: every field has a default.

TODO(phase-1): default weights are all 1.0 until the eval harness has labels
to tune against. Options once it does: hand-tune per feature watching
precision@10 on episodes 1-2, or grid-search the weights offline (the scorer
is pure, so a sweep is cheap). Never tune on the held-out episode.
TODO(phase-2):
  - [ ] seat map: speaker label -> face track id, per camera angle
"""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

SHOWS_DIR = Path("shows")

DEFAULT_WEIGHTS = {
    "keyword_hit": 1.0,
    "take_markers": 1.0,
    "speaker_churn": 1.0,
    "density": 1.0,
    "audio_energy": 1.0,
}

# Generic phrases that tend to open a take or an argument on any talk show.
DEFAULT_TAKE_MARKERS = [
    "i'll say it",
    "hear me out",
    "unpopular opinion",
    "hot take",
    "nobody's talking about",
    "you're wrong",
    "no way",
    "that's crazy",
    "respectfully",
    "i don't care",
    "let me tell you",
]


@dataclass
class ShowConfig:
    slug: str
    display_name: str = ""
    source_url: str = ""
    # Whisper does better with a vocabulary hint; also used to fix captions.
    vocabulary: list[str] = field(default_factory=list)
    # None lets diarization guess; pinning it sharply improves turn boundaries.
    num_speakers: int | None = None
    # Diarization label -> person, e.g. {"SPEAKER_02" = "Jeff Teague"}.
    speaker_names: dict[str, str] = field(default_factory=dict)
    # Episode ids nobody tunes against. `clip eval` flags them in its report.
    held_out: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    keywords: dict[str, list[str]] = field(default_factory=dict)
    take_markers: list[str] = field(default_factory=lambda: list(DEFAULT_TAKE_MARKERS))
    min_clip_s: int = 20
    max_clip_s: int = 60
    # Volume control: at most this many clips per 10 minutes, each above
    # min_score. Never a fixed quota per episode.
    max_per_10_min: int = 2
    min_score: float = 0.25
    nms_iou: float = 0.4


def load_show(slug: str, shows_dir: Path = SHOWS_DIR) -> ShowConfig:
    """Load shows/<slug>.toml over the defaults. A missing file means defaults."""
    path = shows_dir / f"{slug}.toml"
    if not path.exists():
        return ShowConfig(slug=slug)

    data = tomllib.loads(path.read_text())
    allowed = {f.name for f in fields(ShowConfig)} - {"slug"}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")

    weights = data.pop("weights", {})
    bad_weights = sorted(set(weights) - set(DEFAULT_WEIGHTS))
    if bad_weights:
        raise ValueError(f"{path}: unknown features in [weights]: {bad_weights}")

    return ShowConfig(slug=slug, weights={**DEFAULT_WEIGHTS, **weights}, **data)
