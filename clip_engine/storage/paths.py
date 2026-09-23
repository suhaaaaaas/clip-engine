"""One place that knows the layout, so no stage hardcodes a path.

    data/<show>/<episode>/source.mp4
    data/<show>/<episode>/audio.wav
    data/<show>/<episode>/transcript.json
    data/<show>/<episode>/candidates.json
    data/<show>/<episode>/clips/<candidate_key>.mp4

TODO(phase-0):
  - [ ] episode_dir(show, episode) and the per-artifact helpers
  - [ ] make DATA_ROOT overridable by env var
TODO(later):
  - [ ] swap the same interface for S3 when this moves to the cloud
"""

from pathlib import Path

DATA_ROOT = Path("data")


def episode_dir(show: str, episode: str) -> Path:
    raise NotImplementedError("TODO(phase-0)")
