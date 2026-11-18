from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from three_trees.config import DATA_ROOT_ENV, load_config
from three_trees.storage import read_table, table_exists, table_path, write_table
from three_trees.teams import clean_team_name, elo_name

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"


@pytest.mark.parametrize(
    "raw, clean",
    [
        ("Qatar U17", "Qatar"),
        ("Spain U17 W", "Spain"),
        ("Canada W U17", "Canada"),
        ("Colombia U17\tW", "Colombia"),
        ("Korea Republic U17", "Korea Republic"),
        ("Republic of Ireland U17", "Republic of Ireland"),
    ],
)
def test_clean_team_name(raw: str, clean: str) -> None:
    assert clean_team_name(raw) == clean


@pytest.mark.parametrize(
    "raw, name",
    [
        ("Korea Republic U17", "South Korea"),
        ("Korea DPR U17 W", "North Korea"),
        ("Côte d'Ivoire U17", "Ivory Coast"),
        ("Czechia U17", "Czech Republic"),
        ("USA U17 W", "United States"),
        ("Italy U17", "Italy"),
    ],
)
def test_elo_name(raw: str, name: str) -> None:
    assert elo_name(raw) == name


def test_table_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = load_config(EXAMPLE)
    df = pd.DataFrame({"tournament": ["men", "women"], "match_id": [1, 2]})

    assert not table_exists(cfg, "tracking/M01/running")
    path = write_table(cfg, "tracking/M01/running", df)

    assert path == table_path(cfg, "tracking/M01/running")
    assert path.is_relative_to(cfg.build_dir)
    assert not list(path.parent.glob("*.tmp"))
    pd.testing.assert_frame_equal(read_table(cfg, "tracking/M01/running"), df)


def test_missing_table_is_named(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="'matches' is not built yet"):
        read_table(load_config(EXAMPLE), "matches")
