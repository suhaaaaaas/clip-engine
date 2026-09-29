"""Fetch an episode and split it into the two artifacts every later stage wants.

Two files, because they have different consumers: the mp4 feeds framing and
rendering, the wav feeds ASR and diarization (both want mono 16kHz, and the
wav is a fraction of the size).

Phase 0 runs this on episodes downloaded for local study only. Nothing
downloaded here gets posted anywhere.

TODO(phase-1):
  - [ ] probe duration, return it alongside title so later stages don't need
        to open the video just to know its length
"""

import json
import subprocess
from pathlib import Path


def probe(url: str) -> dict:
    """Fetch metadata (id, title, ...) via yt-dlp without downloading."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-warnings", "--no-playlist", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def download(url: str, dest_dir: Path) -> Path:
    """Download the source video to dest_dir/source.mp4 and return its path."""
    dest = dest_dir / "source.mp4"
    if dest.exists():
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f", "bv*[height<=1080]+ba/b[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", str(dest),
            url,
        ],
        check=True,
    )
    return dest


def extract_audio(video_path: Path) -> Path:
    """Write a 16kHz mono wav beside the video and return its path."""
    audio_path = video_path.with_name("audio.wav")
    if audio_path.exists():
        return audio_path
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    return audio_path
