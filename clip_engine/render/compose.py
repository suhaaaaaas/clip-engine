"""ffmpeg: cut the window, apply the crop path, burn captions, encode 9:16.

Renders from an EditList and nothing else, so a reviewer's fix is a data change
plus a re-render, never a manual edit of a video file.

TODO(phase-2):
  - [ ] render(edit_list, source_video, out_path) -> Path
  - [ ] express the crop path as a sendcmd/zoompan filter chain
  - [ ] 1080x1920, sensible bitrate; GPU encoder later when it matters
  - [ ] side-by-side debug output (wide + vertical) for the demo video
"""

from pathlib import Path

from ..models import EditList


def render(edit_list: EditList, source_video: Path, out_path: Path) -> Path:
    raise NotImplementedError("TODO(phase-2)")
