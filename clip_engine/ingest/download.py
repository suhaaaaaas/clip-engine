"""Fetch an episode and split it into the two artifacts every later stage wants.

Two files, because they have different consumers: the mp4 feeds framing and
rendering, the wav feeds ASR and diarization (both want mono 16kHz, and the
wav is a fraction of the size).

Phase 0 runs this on episodes downloaded for local study only. Nothing
downloaded here gets posted anywhere.

TODO(phase-0):
  - [ ] download(url, dest) via yt-dlp, capped at 1080p
  - [ ] extract_audio(video_path) -> wav via ffmpeg (-ac 1 -ar 16000)
  - [ ] probe duration + title, return an Episode record
  - [ ] skip work when the output files already exist
"""

from pathlib import Path


def download(url: str, dest_dir: Path) -> Path:
    """Download the source video to dest_dir/source.mp4 and return its path."""
    raise NotImplementedError("TODO(phase-0)")


def extract_audio(video_path: Path) -> Path:
    """Write a 16kHz mono wav beside the video and return its path."""
    raise NotImplementedError("TODO(phase-0)")
