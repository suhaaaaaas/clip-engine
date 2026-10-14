"""ASR and diarization caching, with faster_whisper and pyannote faked.

The fakes go into sys.modules, so the real (torch-heavy) packages are never
imported and these run without a GPU, model downloads, or network.
"""

import json
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from clip_engine.models import SpeakerTurn
from clip_engine.transcribe.asr import transcribe_cached
from clip_engine.transcribe.diarize import diarize_cached


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"not really audio")
    return path


@pytest.fixture
def whisper(monkeypatch):
    loads = []

    class WhisperModel:
        def __init__(self, size, **kwargs):
            loads.append(size)

        def transcribe(self, path, **kwargs):
            word = SimpleNamespace(start=0.0, end=0.5, word=" hi")
            seg = SimpleNamespace(start=0.0, end=0.5, text=" hi", words=[word])
            return iter([seg]), SimpleNamespace(language="en")

    fake = ModuleType("faster_whisper")
    fake.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return loads


def test_transcribe_cached_records_language_and_model(whisper, audio, tmp_path):
    t = transcribe_cached(audio, "ep", tmp_path, model_size="small")
    assert t.language == "en"
    assert t.model == "small"
    assert t.segments[0].words[0].text == "hi"


def test_transcribe_cached_rerun_is_a_noop(whisper, audio, tmp_path):
    first = transcribe_cached(audio, "ep", tmp_path)
    second = transcribe_cached(audio, "ep", tmp_path)
    assert whisper == ["small"]
    assert first == second


def test_transcribe_cache_is_keyed_by_model_and_audio(whisper, audio, tmp_path):
    transcribe_cached(audio, "ep", tmp_path, model_size="small")
    transcribe_cached(audio, "ep", tmp_path, model_size="medium")
    audio.write_bytes(b"different audio")
    transcribe_cached(audio, "ep", tmp_path, model_size="small")
    assert whisper == ["small", "medium", "small"]


def test_transcribe_cached_force_recomputes(whisper, audio, tmp_path):
    transcribe_cached(audio, "ep", tmp_path)
    transcribe_cached(audio, "ep", tmp_path, force=True)
    assert whisper == ["small", "small"]


def fake_pyannote(monkeypatch, pipeline_factory):
    pkg = ModuleType("pyannote")
    audio_mod = ModuleType("pyannote.audio")
    audio_mod.Pipeline = SimpleNamespace(from_pretrained=pipeline_factory)
    pkg.audio = audio_mod
    monkeypatch.setitem(sys.modules, "pyannote", pkg)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_mod)


@pytest.fixture
def pipeline_calls(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")
    calls = []

    def from_pretrained(name, token=None):
        def run(path, num_speakers=None):
            calls.append(path)
            tracks = [(SimpleNamespace(start=1.0, end=2.0), None, "SPEAKER_00")]
            annotation = SimpleNamespace(itertracks=lambda yield_label: iter(tracks))
            return SimpleNamespace(exclusive_speaker_diarization=annotation)

        return run

    fake_pyannote(monkeypatch, from_pretrained)
    return calls


def test_diarize_cached_writes_cache_and_reuses_it(pipeline_calls, audio, tmp_path):
    cache = tmp_path / "diarization.json"
    first = diarize_cached(audio, cache)
    second = diarize_cached(audio, cache)
    assert first == second == [SpeakerTurn(1000, 2000, "SPEAKER_00")]
    assert len(pipeline_calls) == 1
    assert json.loads(cache.read_text())[0]["speaker"] == "SPEAKER_00"


def test_diarize_failure_logs_and_returns_no_turns(monkeypatch, audio, tmp_path, caplog):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")

    def from_pretrained(name, token=None):
        raise RuntimeError("403 gated repo")

    fake_pyannote(monkeypatch, from_pretrained)
    cache = tmp_path / "diarization.json"
    with caplog.at_level(logging.WARNING):
        assert diarize_cached(audio, cache) == []
    assert "403 gated repo" in caplog.text
    assert not cache.exists()


def test_diarize_without_token_degrades(pipeline_calls, monkeypatch, audio, tmp_path):
    monkeypatch.delenv("HUGGINGFACE_TOKEN")
    assert diarize_cached(audio, tmp_path / "diarization.json") == []
    assert pipeline_calls == []
