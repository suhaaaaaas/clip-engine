# clip-engine

Turns long video podcasts into short vertical clips: transcribe, find the
moments worth clipping, crop 16:9 to 9:16 following whoever is talking, burn
captions, hold everything in a review queue before anything gets posted. A
two-person learning project — see [README.md](README.md) and
[docs/ROADMAP.md](docs/ROADMAP.md) for phases and status.

## Conventions

- **Labels before models.** `golden/` holds hand-picked clips; every scorer
  change is measured against them, not against taste. Build the eval harness
  (`clip_engine/scoring/evaluate.py`) before the scorer.
- **Media never gets committed.** `data/` is gitignored, as are loose
  `*.mp4`/`*.wav`/etc. Episodes are downloaded for local study; clips go only
  to the host who owns the show. Never suggest committing files under `data/`
  except where the gitignore explicitly carves out an exception (e.g.
  `golden/*.csv`).
- **Clips are recipes.** A clip is an `EditList` (trim points, caption words,
  crop path) — the video is rendered from it. A fix to a clip is a data
  change, not a re-render or a re-edit of the video file.

## Working conventions

- **Every TODO carries its phase**, e.g. `TODO(phase-1)`. Check a module's
  phase before extending it — don't implement phase-2 work while phase-0/1 is
  still open. `grep -rn "TODO(phase-N)" clip_engine/` finds the open work for
  a phase.
- Pipeline stages (`ingest → transcribe/diarize/align → scoring → framing →
  render → review`) are separately runnable and cached to disk under
  `data/<show>/<episode>/` — re-transcribing to test a scoring change wastes
  real time, so stage boundaries should stay hard.
- Diarization (pyannote) needs a HuggingFace token and is expected to degrade
  gracefully — an unlabeled transcript (no speaker field) is still usable
  downstream, so don't make diarization a hard dependency of the ASR stage.
