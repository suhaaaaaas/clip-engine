import importlib

import pytest

from clip_engine.storage import paths


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIP_DATA_ROOT", str(tmp_path))
    importlib.reload(paths)
    yield tmp_path
    monkeypatch.delenv("CLIP_DATA_ROOT")
    importlib.reload(paths)


def test_data_root_is_overridable_by_env(data_root):
    assert paths.episode_dir("show", "ep") == data_root / "show" / "ep"
    assert (data_root / "show" / "ep").is_dir()


def test_artifact_paths_live_in_the_episode_dir(data_root):
    ep = data_root / "show" / "ep"
    assert paths.audio_path("show", "ep") == ep / "audio.wav"
    assert paths.transcript_path("show", "ep") == ep / "transcript.json"
    assert paths.candidates_path("show", "ep") == ep / "candidates.json"
    assert paths.clips_dir("show", "ep").is_dir()
