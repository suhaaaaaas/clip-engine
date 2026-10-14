"""Ingest with subprocess mocked: no network, no yt-dlp, no ffmpeg."""

import json
from types import SimpleNamespace

import pytest

from clip_engine.ingest import download as dl


@pytest.fixture
def calls(monkeypatch):
    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        stdout = json.dumps({"id": "abc_123", "title": "Ep", "duration": 3908, "webpage_url": "u"})
        return SimpleNamespace(stdout=stdout, returncode=0)

    monkeypatch.setattr(dl.subprocess, "run", fake_run)
    return recorded


def test_probe_returns_id_title_and_duration(calls):
    assert dl.probe("u") == {"id": "abc_123", "title": "Ep", "duration_s": 3908, "url": "u"}
    assert calls[0][0] == "yt-dlp"


def test_download_caps_resolution_and_targets_source_mp4(calls, tmp_path):
    dest = dl.download("u", tmp_path / "ep")
    assert dest == tmp_path / "ep" / "source.mp4"
    cmd = calls[0]
    assert "bv*[height<=1080]+ba/b[height<=1080]" in cmd
    assert cmd[cmd.index("-o") + 1] == str(dest)


def test_download_is_a_noop_when_video_exists(calls, tmp_path):
    (tmp_path / "source.mp4").touch()
    dl.download("u", tmp_path)
    assert calls == []


def test_download_force_redownloads(calls, tmp_path):
    (tmp_path / "source.mp4").touch()
    dl.download("u", tmp_path, force=True)
    assert len(calls) == 1


def test_extract_audio_writes_mono_16k(calls, tmp_path):
    out = dl.extract_audio(tmp_path / "source.mp4")
    assert out == tmp_path / "audio.wav"
    cmd = calls[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"


def test_extract_audio_is_a_noop_when_wav_exists(calls, tmp_path):
    (tmp_path / "audio.wav").touch()
    dl.extract_audio(tmp_path / "source.mp4")
    assert calls == []
