# clip-engine

Turn long video podcasts into short vertical clips: transcribe, find the
moments worth clipping, crop 16:9 to 9:16 following whoever is talking, burn
captions, and hold everything in a review queue before anything gets posted.

A two-person learning project. Phases are ordered by what they teach, and each
one also tests whether a business is here. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Status

Skeleton. Every module is stubbed with its contract and a TODO list. Start at
[docs/PHASE0.md](docs/PHASE0.md).

| Stage | Module | Phase |
| --- | --- | --- |
| Download + audio extract | `clip_engine/ingest/` | 0 |
| Transcribe + diarize + align | `clip_engine/transcribe/` | 0 |
| Eval harness | `clip_engine/scoring/evaluate.py` | 0 |
| Candidate windows + features + scorer | `clip_engine/scoring/` | 1 |
| Shots, faces, seat map, crop path | `clip_engine/framing/` | 2 |
| Captions + ffmpeg compose | `clip_engine/render/` | 2 |
| Review loop | `clip_engine/review/` | 3 |

## Setup

```bash
make setup                 # venv + dev deps
source .venv/bin/activate
cp .env.example .env       # add your HuggingFace token when you reach diarization
make test
```

Diarization (`torch`, `pyannote`) and framing (`opencv`, `scenedetect`) are
optional extras. Install them when you reach those stages:

```bash
make setup-full
```

`ffmpeg` must be on your PATH.

## Working on it

```bash
clip ingest <show> <url>     # download + extract audio
clip asr    <show> <episode> # transcript with word timings and speakers
clip score  <show> <episode> # rank candidate clips
clip eval   <show> <episode> # score against the golden set
clip frame  <show> <episode> # crop path following the speaker
clip render <show> <episode> # cut, caption, encode 9:16
```

Stages are separately runnable on purpose: you will re-run scoring many times
against one transcript, and re-transcribing to do it wastes an hour each time.

## Conventions

- **Labels before models.** `golden/` holds hand-picked clips; the scorer is
  measured against them. See [golden/README.md](golden/README.md).
- **Media never gets committed.** `data/` is gitignored. Episodes are
  downloaded for local study; clips go only to the host who owns the show.
- **Clips are recipes.** A clip is an `EditList` (trim points, caption words,
  crop path) and the video is rendered from it, so a fix is a data change.
- **Every TODO carries its phase**, e.g. `TODO(phase-1)`. Grep for your phase
  to find your work: `grep -rn "TODO(phase-1)" clip_engine/`
