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
  - [ ] wire ingest / asr / eval
  - [ ] --force to bypass caches
TODO(phase-1):
  - [ ] score + eval printing a per-episode precision table
TODO(phase-2):
  - [ ] frame + render, with a --debug side-by-side output
"""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clip")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("ingest", "asr", "score", "eval", "frame", "render", "review"):
        p = sub.add_parser(name)
        p.add_argument("show")
        p.add_argument("episode")

    args = parser.parse_args(argv)
    raise NotImplementedError(f"TODO: implement `clip {args.command}`")


if __name__ == "__main__":
    raise SystemExit(main())
