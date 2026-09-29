"""Command line entry point. One subcommand per pipeline stage.

Stages are separately runnable on purpose: you will re-run scoring a hundred
times against one transcript, and you should never re-transcribe to do it.

    clip ingest   <show> <url>
    clip asr      <show> <episode>
    clip score    <show> <episode>
    clip eval     <show> <episode>
    clip frame    <show> <episode>
    clip render   <show> <episode>
    clip review   <show> <episode>

TODO(phase-0):
  - [ ] wire eval
  - [ ] --force to bypass caches
TODO(phase-1):
  - [ ] score + eval printing a per-episode precision table
TODO(phase-2):
  - [ ] frame + render, with a --debug side-by-side output
"""

import argparse

from dotenv import load_dotenv

from .ingest.download import download, extract_audio, probe
from .models import Transcript
from .storage.paths import audio_path as audio_file
from .storage.paths import episode_dir, transcript_path
from .transcribe.align import assign_speakers
from .transcribe.asr import transcribe
from .transcribe.diarize import diarize


def _cmd_ingest(show: str, url: str) -> int:
    info = probe(url)
    episode = info["id"]
    print(f"[{show}/{episode}] {info.get('title', url)}")

    dest = episode_dir(show, episode)
    video_path = download(url, dest)
    extract_audio(video_path)

    print(f"episode id: {episode}")
    return 0


def _cmd_asr(show: str, episode: str) -> int:
    audio = audio_file(show, episode)
    if not audio.exists():
        raise SystemExit(f"no audio at {audio}; run `clip ingest {show} <url>` first")

    out_path = transcript_path(show, episode)
    if out_path.exists():
        print(f"{out_path} already exists, skipping (delete it to re-run)")
        return 0

    print("transcribing...")
    segments = transcribe(audio)

    try:
        turns = diarize(audio)
        segments = assign_speakers(segments, turns)
    except Exception as exc:  # noqa: BLE001 -- diarize() is meant to degrade gracefully
        print(f"diarization skipped ({exc}); transcript will be unlabeled")

    Transcript(episode_id=episode, segments=segments).to_json(out_path)
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="clip")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("show")
    p_ingest.add_argument("url")

    for name in ("asr", "score", "eval", "frame", "render", "review"):
        p = sub.add_parser(name)
        p.add_argument("show")
        p.add_argument("episode")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _cmd_ingest(args.show, args.url)
    if args.command == "asr":
        return _cmd_asr(args.show, args.episode)

    raise NotImplementedError(f"TODO: implement `clip {args.command}`")


if __name__ == "__main__":
    raise SystemExit(main())
