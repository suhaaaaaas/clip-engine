"""CLI wiring against a temp data root and temp golden dir -- never golden/."""

import importlib
import json
import shutil
from pathlib import Path

import pytest

from clip_engine import cli
from clip_engine.models import Segment, SpeakerTurn, Transcript
from clip_engine.storage import paths

FIXTURES = Path(__file__).parent / "fixtures"
SHOW_TOML = """
min_score = 0.0
max_per_10_min = 10
num_speakers = 2
vocabulary = ["Kawhi Leonard"]
held_out = ["held"]

[keywords]
trades = ["trade"]
"""


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Temp cwd with shows/demo.toml, a data root, and an empty golden dir."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "shows").mkdir()
    (tmp_path / "shows" / "demo.toml").write_text(SHOW_TOML)
    (tmp_path / "golden").mkdir()
    monkeypatch.setenv("CLIP_DATA_ROOT", str(tmp_path / "data"))
    importlib.reload(paths)
    yield tmp_path
    monkeypatch.delenv("CLIP_DATA_ROOT")
    importlib.reload(paths)


def write_transcript(episode: str) -> None:
    segs = []
    for i in range(120):  # 10 minutes of 5-second segments
        hot = 100 <= i * 5 < 140
        segs.append(
            Segment(i * 5000, i * 5000 + 5000, "trade him now" if hot else "weather", f"S{i % 2}" if hot else "S0")
        )
    Transcript(episode, segs).to_json(paths.transcript_path("demo", episode))


def label(workspace: Path, episode: str, fixture: str, name: str) -> None:
    shutil.copy(FIXTURES / fixture, workspace / "golden" / f"{episode}_{name}.csv")


def run(*argv: str) -> int:
    return cli.main(list(argv))


def test_asr_passes_show_vocabulary_and_speaker_count(workspace, monkeypatch):
    paths.audio_path("demo", "ep").write_bytes(b"wav")
    seen = {}

    def fake_transcribe(audio, episode, cache_dir, vocabulary=None, force=False):
        seen["vocabulary"] = vocabulary
        return Transcript(episode, [Segment(0, 1000, "hi")], language="en", model="small")

    def fake_diarize(audio, cache_path, num_speakers=None, force=False):
        seen["num_speakers"] = num_speakers
        return [SpeakerTurn(0, 1000, "SPEAKER_00")]

    monkeypatch.setattr(cli, "transcribe_cached", fake_transcribe)
    monkeypatch.setattr(cli, "diarize_cached", fake_diarize)

    assert run("asr", "demo", "ep") == 0
    assert seen == {"vocabulary": ["Kawhi Leonard"], "num_speakers": 2}
    written = Transcript.from_json(paths.transcript_path("demo", "ep"))
    assert written.segments[0].speaker == "SPEAKER_00"

    seen.clear()
    assert run("asr", "demo", "ep") == 0
    assert seen == {}  # transcript exists: no-op


def test_score_writes_candidates_and_prints_top(workspace, capsys):
    write_transcript("ep")
    assert run("score", "demo", "ep") == 0
    candidates = json.loads(paths.candidates_path("demo", "ep").read_text())
    assert candidates and all(20_000 <= c["end_ms"] - c["start_ms"] <= 60_000 for c in candidates)
    assert " 1. " in capsys.readouterr().out


def test_score_without_transcript_explains(workspace):
    with pytest.raises(SystemExit, match="clip asr"):
        run("score", "demo", "missing")


def test_eval_without_labels_says_so(workspace, capsys):
    write_transcript("ep")
    run("score", "demo", "ep")
    assert run("eval", "demo", "ep") == 1
    assert "no labels yet" in capsys.readouterr().out


def test_eval_with_two_unmerged_label_sets_reports_agreement_only(workspace, capsys):
    write_transcript("ep")
    run("score", "demo", "ep")
    label(workspace, "ep", "synthetic_labels_a.csv", "alice")
    label(workspace, "ep", "synthetic_labels_b.csv", "bob")
    assert run("eval", "demo", "ep") == 1
    out = capsys.readouterr().out
    assert "alice vs bob: 0.67" in out
    assert "merge your labels" in out


def test_eval_with_merged_labels_prints_precision_table(workspace, capsys):
    write_transcript("ep")
    run("score", "demo", "ep")
    label(workspace, "ep", "synthetic_labels_a.csv", "alice")
    label(workspace, "ep", "synthetic_labels_b.csv", "bob")
    label(workspace, "ep", "synthetic_labels_a.csv", "merged")
    assert run("eval", "demo", "ep") == 0
    out = capsys.readouterr().out
    assert "p@k" in out
    row = next(line for line in out.splitlines() if line.startswith("ep "))
    assert "0.67" in row  # human agreement column


def test_eval_all_covers_every_episode_and_flags_held_out(workspace, capsys):
    for episode in ("ep", "held"):
        write_transcript(episode)
        run("score", "demo", episode)
        label(workspace, episode, "synthetic_labels_a.csv", "merged")
    assert run("eval", "demo", "all") == 0
    out = capsys.readouterr().out
    assert "HELD OUT" in out
    assert any(line.startswith("held ") and "held out" in line for line in out.splitlines())
    assert any(line.startswith("ep ") for line in out.splitlines())
