from __future__ import annotations

import os
from pathlib import Path

import pytest

from three_trees.config import DATA_ROOT_ENV, TOURNAMENTS, ConfigError, load_config, repo_root

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    return tmp_path.resolve()


def test_example_config_resolves_under_data_root(data_root: Path) -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.data_root == data_root
    assert set(cfg.statsbomb) == set(TOURNAMENTS)
    assert all(p.is_relative_to(data_root) for p in cfg.data_paths().values())
    assert all(url.startswith("https://") for url in cfg.external_urls.values())


def test_event_parts_per_tournament_in_order(data_root: Path) -> None:
    folder = data_root / "statsbomb"
    folder.mkdir()
    for n in (10, 2, 1):
        (folder / f"u17_world_cup_2025_events_part{n:03d}.csv").touch()
    (folder / "u17_womens_world_cup_2025_events_part001.csv").touch()

    cfg = load_config(EXAMPLE)

    assert [p.name for p in cfg.statsbomb["men"].event_files()] == [
        "u17_world_cup_2025_events_part001.csv",
        "u17_world_cup_2025_events_part002.csv",
        "u17_world_cup_2025_events_part010.csv",
    ]
    assert [p.name for p in cfg.statsbomb["women"].event_files()] == [
        "u17_womens_world_cup_2025_events_part001.csv",
    ]


def test_data_root_inside_repo_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    root = repo_root()
    if root is None:
        pytest.skip("package is not inside a git work tree")
    monkeypatch.setenv(DATA_ROOT_ENV, str(root / "data"))
    with pytest.raises(ConfigError, match="outside the repo"):
        load_config(EXAMPLE)


def test_missing_key_is_named(tmp_path: Path, data_root: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("data_root: /x\nbuild_dir: build\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="statsbomb.men.events"):
        load_config(config)


def test_missing_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THREE_TREES_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="config.example.yaml"):
        load_config()


def test_load_env_file_keeps_existing_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from three_trees.config import load_env_file

    env = tmp_path / ".env"
    env.write_text("# key for the Ask page\nexport APP_TEST_KEY='abc'\nAPP_TEST_SET=new\nAPP_TEST_EMPTY=\n",
                   encoding="utf-8")
    monkeypatch.delenv("APP_TEST_KEY", raising=False)
    monkeypatch.setenv("APP_TEST_SET", "old")
    monkeypatch.delenv("APP_TEST_EMPTY", raising=False)
    assert load_env_file(env) == ["APP_TEST_KEY"]
    assert os.environ["APP_TEST_KEY"] == "abc" and os.environ["APP_TEST_SET"] == "old"
    assert "APP_TEST_EMPTY" not in os.environ
    monkeypatch.delenv("APP_TEST_KEY")
    assert load_env_file(tmp_path / "missing.env") == []
