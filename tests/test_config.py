from pathlib import Path

import pytest

from clip_engine.config import DEFAULT_TAKE_MARKERS, DEFAULT_WEIGHTS, ShowConfig, load_show

REPO_SHOWS = Path(__file__).parent.parent / "shows"


def write_show(tmp_path: Path, body: str) -> Path:
    (tmp_path / "demo.toml").write_text(body)
    return tmp_path


def test_unconfigured_show_gets_defaults(tmp_path):
    config = load_show("newshow", shows_dir=tmp_path)
    assert config == ShowConfig(slug="newshow")
    assert config.weights == DEFAULT_WEIGHTS
    assert config.take_markers == DEFAULT_TAKE_MARKERS


def test_toml_merges_over_defaults(tmp_path):
    shows = write_show(
        tmp_path,
        """
display_name = "Demo"
num_speakers = 3
held_out = ["ep3"]

[weights]
density = 2.5

[keywords]
trades = ["trade", "max"]
""",
    )
    config = load_show("demo", shows_dir=shows)
    assert config.display_name == "Demo"
    assert config.num_speakers == 3
    assert config.held_out == ["ep3"]
    assert config.weights == {**DEFAULT_WEIGHTS, "density": 2.5}
    assert config.keywords == {"trades": ["trade", "max"]}
    assert config.max_per_10_min == ShowConfig(slug="x").max_per_10_min


def test_unknown_key_is_rejected(tmp_path):
    shows = write_show(tmp_path, 'dispaly_name = "typo"\n')
    with pytest.raises(ValueError, match="dispaly_name"):
        load_show("demo", shows_dir=shows)


def test_unknown_weight_is_rejected(tmp_path):
    shows = write_show(tmp_path, "[weights]\nvibes = 1.0\n")
    with pytest.raises(ValueError, match="vibes"):
        load_show("demo", shows_dir=shows)


def test_club520_config_loads():
    config = load_show("club520", shows_dir=REPO_SHOWS)
    assert config.num_speakers == 4
    assert "Kawhi Leonard" in config.vocabulary
    assert set(config.weights) == set(DEFAULT_WEIGHTS)
