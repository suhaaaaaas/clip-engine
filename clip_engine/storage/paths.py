"""One place that knows the layout, so no stage hardcodes a path.

    data/<show>/<episode>/source.mp4
    data/<show>/<episode>/audio.wav
    data/<show>/<episode>/transcript.json
    data/<show>/<episode>/candidates.json
    data/<show>/<episode>/clips/<candidate_key>.mp4

TODO(later):
  - [ ] swap the same interface for S3 when this moves to the cloud
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("CLIP_DATA_ROOT", "data"))


def episode_dir(show: str, episode: str) -> Path:
    d = DATA_ROOT / show / episode
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_video_path(show: str, episode: str) -> Path:
    return episode_dir(show, episode) / "source.mp4"


def audio_path(show: str, episode: str) -> Path:
    return episode_dir(show, episode) / "audio.wav"


def transcript_path(show: str, episode: str) -> Path:
    return episode_dir(show, episode) / "transcript.json"


def candidates_path(show: str, episode: str) -> Path:
    return episode_dir(show, episode) / "candidates.json"


def clips_dir(show: str, episode: str) -> Path:
    d = episode_dir(show, episode) / "clips"
    d.mkdir(parents=True, exist_ok=True)
    return d
