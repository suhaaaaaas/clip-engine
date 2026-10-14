"""Command line entry point. One subcommand per pipeline stage.

Stages are separately runnable on purpose: you will re-run scoring a hundred
times against one transcript, and you should never re-transcribe to do it.
Every stage is a no-op when its output exists; --force redoes it.

    clip ingest   <show> <url>
    clip asr      <show> <episode>
    clip score    <show> <episode>
    clip eval     <show> <episode>
    clip frame    <show> <episode>
    clip render   <show> <episode>
    clip review   <show> <episode>

TODO(phase-1):
  - [ ] score + eval printing a per-episode precision table
TODO(phase-2):
  - [ ] frame + render, with a --debug side-by-side output
"""

import argparse
import json
import logging

from dotenv import load_dotenv

from .ingest.download import download, extract_audio, probe
from .storage import paths
from .transcribe.align import assign_speakers
from .transcribe.asr import transcribe_cached
from .transcribe.diarize import diarize_cached


def _cmd_ingest(args: argparse.Namespace) -> int:
    meta = probe(args.url)
    episode = meta["id"]
    print(f"[{args.show}/{episode}] {meta['title']}")

    dest = paths.episode_dir(args.show, episode)
    paths.episode_meta_path(args.show, episode).write_text(json.dumps(meta, indent=2))
    video_path = download(args.url, dest, force=args.force)
    extract_audio(video_path, force=args.force)

    print(f"episode id: {episode}")
    return 0


def _cmd_asr(args: argparse.Namespace) -> int:
    show, episode = args.show, args.episode
    audio = paths.audio_path(show, episode)
    if not audio.exists():
        raise SystemExit(f"no audio at {audio}; run `clip ingest {show} <url>` first")

    out_path = paths.transcript_path(show, episode)
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists, skipping (--force to redo)")
        return 0

    print("transcribing...")
    transcript = transcribe_cached(audio, episode, paths.episode_dir(show, episode), force=args.force)

    print("diarizing...")
    turns = diarize_cached(audio, paths.diarization_path(show, episode), force=args.force)
    transcript.segments = assign_speakers(transcript.segments, turns)

    transcript.to_json(out_path)
    print(f"wrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="clip")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("show")
    p_ingest.add_argument("url")
    p_ingest.add_argument("--force", action="store_true")

    for name in ("asr", "score", "eval", "frame", "render", "review"):
        p = sub.add_parser(name)
        p.add_argument("show")
        p.add_argument("episode")
        p.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "asr":
        return _cmd_asr(args)

    raise NotImplementedError(f"TODO: implement `clip {args.command}`")


if __name__ == "__main__":
    raise SystemExit(main())
