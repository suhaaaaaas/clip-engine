"""Data shapes passed between stages.

These are the contract. Every stage reads and writes these, so they can be
built and tested independently.
"""

from dataclasses import dataclass, field
import hashlib


@dataclass(frozen=True, slots=True)
class Word:
    """One word with its timing. Captions are built from these."""

    start_ms: int
    end_ms: int
    text: str


@dataclass(frozen=True, slots=True)
class Segment:
    """A transcript segment, optionally attributed to a speaker."""

    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A diarization turn, before it is aligned onto transcript segments."""

    start_ms: int
    end_ms: int
    speaker: str


@dataclass
class Transcript:
    episode_id: str
    segments: list[Segment]
    language: str | None = None
    model: str = ""

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments if s.speaker})

    # TODO(phase-0): to_json / from_json so transcripts round-trip to disk


@dataclass
class ClipCandidate:
    """A scored window. Phase 1's output, Phase 2's input."""

    episode_id: str
    start_ms: int
    end_ms: int
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    text: str = ""
    speakers: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        raw = f"{self.episode_id}:{self.start_ms}:{self.end_ms}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


@dataclass
class CropKeyframe:
    """Where the 9:16 window sits at a point in time. Phase 2's output."""

    at_ms: int
    # Center of the crop in normalized source coordinates (0.0-1.0).
    center_x: float
    center_y: float
    # 1.0 = full source height; below that zooms in.
    scale: float = 1.0
    # TODO(phase-2): layout enum -- single speaker vs stacked two-up


@dataclass
class EditList:
    """The recipe a clip renders from. Reviewers edit this, never the video."""

    candidate: ClipCandidate
    crop_path: list[CropKeyframe] = field(default_factory=list)
    caption_words: list[Word] = field(default_factory=list)
    title: str = ""
    approved: bool = False
    # TODO(phase-3): reviewer notes, per-platform metadata
