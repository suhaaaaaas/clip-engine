# clip-engine

Turn long video podcasts into short vertical clips: transcribe, find the
moments worth clipping, crop 16:9 to 9:16 following whoever is talking, burn
captions, and hold everything in a review queue before anything gets posted.

A two-person learning project. Phases are ordered by what they teach, and each
one also tests whether a business is here. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Status

Phases 0 and 1 are built; the golden labels are not. `clip eval` can't
produce a number until someone labels clips in `golden/` (see
[docs/PHASE0.md](docs/PHASE0.md)). One episode of Club 520 is transcribed
and diarized locally.

| Stage | Module | Phase | Status |
| --- | --- | --- | --- |
| Download + audio extract | `clip_engine/ingest/` | 0 | Done, cached |
| Transcribe + diarize + align | `clip_engine/transcribe/` | 0 | Done, cached; diarization optional |
| Eval harness | `clip_engine/scoring/evaluate.py` | 0 | Done; waiting on labels |
| Show config | `clip_engine/config.py`, `shows/` | 1 | Done; `club520.toml` seeded |
| Candidate windows + features + scorer | `clip_engine/scoring/` | 1 | Done; weights untuned |
| Shots, faces, seat map, crop path | `clip_engine/framing/` | 2 | Stub |
| Captions + ffmpeg compose | `clip_engine/render/` | 2 | Stub |
| Review loop | `clip_engine/review/` | 3 | Stub |

## Setup

```bash
make setup                 # venv + dev deps
source .venv/bin/activate
echo "HUGGINGFACE_TOKEN=hf_..." > .env   # only needed for diarization
make test
```

Diarization (`torch`, `pyannote`) and framing (`opencv`, `scenedetect`) are
optional extras. Install them when you reach those stages:

```bash
make setup-full
```

Diarization needs the HuggingFace token plus accepted terms on three gated
repos: `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, and
`pyannote/speaker-diarization-community-1`. Without them `clip asr` still
writes a transcript, just with no speaker labels.

`ffmpeg` must be on your PATH.

With an Nvidia GPU, install the CUDA libraries so `clip asr` runs large-v3 on
it (minutes per episode instead of half an hour):

```bash
pip install -e ".[gpu]"
```

`clip asr` checks at startup: no GPU, or the CUDA libraries missing, and it
logs why and falls back to the `small` model on CPU. On Linux the libraries
may also need to be on `LD_LIBRARY_PATH`; the fallback warning says when.

## Working on it

```bash
clip ingest <show> <url>           # download + extract audio; prints the episode id
clip asr    <show> <episode>       # transcript with word timings and speakers
clip score  <show> <episode>       # rank candidate clips -> candidates.json
clip eval   <show> <episode|all>   # precision@10 vs golden/, plus human agreement
clip frame  <show> <episode>       # phase 2, not built
clip render <show> <episode>       # phase 2, not built
```

`ingest` and `asr` skip work whose output already exists; pass `--force` to
redo it. `score` always reruns (about 10s an episode) because rerunning it
after editing `shows/<show>.toml` is the point. Per-show settings (vocabulary,
speaker count, keywords, weights, held-out episode) live in
`shows/<show>.toml`.

Stages are separately runnable on purpose: you will re-run scoring many times
against one transcript, and re-transcribing to do it wastes an hour each time.

## Conventions

- **Labels before models.** `golden/` holds hand-picked clips; the scorer is
  measured against them. See [golden/README.md](golden/README.md).
- **Media never gets committed.** `data/` is gitignored except each
  episode's `transcript.json`. Episodes are downloaded for local study; clips
  go only to the host who owns the show.
- **Clips are recipes.** A clip is an `EditList` (trim points, caption words,
  crop path) and the video is rendered from it, so a fix is a data change.
- **Every TODO carries its phase**, e.g. `TODO(phase-1)`. Grep for your phase
  to find your work: `grep -rn "TODO(phase-1)" clip_engine/`
