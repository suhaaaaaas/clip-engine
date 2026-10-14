"""Command line entry point. One subcommand per pipeline stage.

Stages are separately runnable on purpose: you will re-run scoring a hundred
times against one transcript, and you should never re-transcribe to do it.
ingest and asr are no-ops when their output exists (--force redoes them).
score always reruns: it takes seconds and its whole point is re-running
after a config change.

    clip ingest   <show> <url>
    clip asr      <show> <episode>
    clip score    <show> <episode>
    clip eval     <show> <episode|all>
    clip frame    <show> <episode>
    clip render   <show> <episode>
    clip review   <show> <episode>

TODO(phase-2):
  - [ ] frame + render, with a --debug side-by-side output
"""

import argparse
import itertools
import json
import logging
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .config import ShowConfig, load_show
from .ingest.audio import rms_envelope
from .ingest.download import download, extract_audio, probe
from .models import ClipCandidate, Transcript
from .scoring.evaluate import EvalRow, agreement, hits_at_k, load_golden, report
from .scoring.scorer import AudioEnvelope, score_episode
from .storage import paths
from .transcribe.align import assign_speakers
from .transcribe.asr import transcribe_cached
from .transcribe.diarize import diarize_cached

EVAL_K = 10
HOP_MS = 100


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
    config = load_show(show)
    audio = paths.audio_path(show, episode)
    if not audio.exists():
        raise SystemExit(f"no audio at {audio}; run `clip ingest {show} <url>` first")

    out_path = paths.transcript_path(show, episode)
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists, skipping (--force to redo)")
        return 0

    print("transcribing...")
    transcript = transcribe_cached(
        audio,
        episode,
        paths.episode_dir(show, episode),
        vocabulary=config.vocabulary,
        force=args.force,
    )

    print("diarizing...")
    turns = diarize_cached(
        audio,
        paths.diarization_path(show, episode),
        num_speakers=config.num_speakers,
        force=args.force,
    )
    transcript.segments = assign_speakers(transcript.segments, turns)

    transcript.to_json(out_path)
    print(f"wrote {out_path}")
    return 0


def _clock(ms: int) -> str:
    return f"{ms // 60_000}:{ms // 1000 % 60:02d}"


def _cmd_score(args: argparse.Namespace) -> int:
    show, episode = args.show, args.episode
    transcript_file = paths.transcript_path(show, episode)
    if not transcript_file.exists():
        raise SystemExit(f"no transcript at {transcript_file}; run `clip asr {show} {episode}` first")

    config = load_show(show)
    transcript = Transcript.from_json(transcript_file)
    wav = paths.audio_path(show, episode)
    audio = AudioEnvelope(rms_envelope(wav, HOP_MS), HOP_MS) if wav.exists() else None

    candidates = score_episode(transcript, config, audio)
    out = paths.candidates_path(show, episode)
    out.write_text(json.dumps([asdict(c) for c in candidates], indent=2))

    print(f"{len(candidates)} candidates -> {out}")
    for rank, c in enumerate(candidates[:EVAL_K], start=1):
        top = max(c.breakdown, key=c.breakdown.get) if c.breakdown else "-"
        snippet = c.text[:70] + ("..." if len(c.text) > 70 else "")
        print(f"{rank:>2}. {_clock(c.start_ms)}-{_clock(c.end_ms)}  {c.score:.2f}  [{top}]  {snippet}")
    return 0


def _eval_row(
    show: str, episode: str, config: ShowConfig, golden_dir: Path
) -> tuple[EvalRow | None, list[str]]:
    """One table row plus notes for the episode. Row is None when there's no truth yet."""
    notes = []
    candidates_file = paths.candidates_path(show, episode)
    if not candidates_file.exists():
        return None, [f"{episode}: no candidates; run `clip score {show} {episode}` first"]

    merged_file = golden_dir / f"{episode}_merged.csv"
    individual = sorted(p for p in golden_dir.glob(f"{episode}_*.csv") if p != merged_file)
    if not merged_file.exists() and not individual:
        hint = f"{golden_dir}/{episode}_<name>.csv (see golden/README.md)"
        return None, [f"{episode}: no labels yet -- each of you add {hint}"]

    labels = {p.stem.removeprefix(f"{episode}_"): load_golden(p) for p in individual}
    pair_scores = []
    for (name_a, a), (name_b, b) in itertools.combinations(labels.items(), 2):
        score = agreement(a, b)
        pair_scores.append(score)
        notes.append(f"{episode}: human agreement {name_a} vs {name_b}: {score:.2f}")
    human = sum(pair_scores) / len(pair_scores) if pair_scores else None

    if merged_file.exists():
        golden = load_golden(merged_file)
    elif len(individual) == 1:
        golden = next(iter(labels.values()))
        notes.append(f"{episode}: scoring against {individual[0].name} alone until a merged file exists")
    else:
        notes.append(f"{episode}: merge your labels into {merged_file.name} to get a precision number")
        return None, notes

    predicted = [ClipCandidate(**c) for c in json.loads(candidates_file.read_text())]
    row = EvalRow(
        episode_id=episode,
        k=EVAL_K,
        hits=hits_at_k(predicted, golden, k=EVAL_K),
        golden_count=len(golden),
        agreement=human,
        held_out=episode in config.held_out,
    )
    if row.held_out:
        notes.append(f"{episode}: HELD OUT -- read this number, don't tune weights against it")
    return row, notes


def _cmd_eval(args: argparse.Namespace) -> int:
    config = load_show(args.show)
    golden_dir = Path(args.golden_dir)
    if args.episode == "all":
        show_dir = paths.DATA_ROOT / args.show
        episodes = sorted(p.name for p in show_dir.iterdir() if p.is_dir()) if show_dir.exists() else []
    else:
        episodes = [args.episode]

    rows = []
    for episode in episodes:
        row, notes = _eval_row(args.show, episode, config, golden_dir)
        for note in notes:
            print(note)
        if row:
            rows.append(row)

    if not rows:
        return 1
    print()
    print(report(rows))
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
        if name == "eval":
            p.add_argument("--golden-dir", default="golden")

    args = parser.parse_args(argv)
    handlers = {"ingest": _cmd_ingest, "asr": _cmd_asr, "score": _cmd_score, "eval": _cmd_eval}
    if args.command in handlers:
        return handlers[args.command](args)

    raise NotImplementedError(f"TODO(phase-2/3): implement `clip {args.command}`")


if __name__ == "__main__":
    raise SystemExit(main())
