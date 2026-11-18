from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from three_trees import cli, db, measures, synthetic
from three_trees.config import repo_root
from three_trees.storage import read_table, write_table


# ---------------------------------------------------------------- measure registry


def test_registry_is_complete_and_unique() -> None:
    reg = measures.load_measures()
    assert reg["measure"].is_unique
    assert not reg[["measure", "label", "unit", "table", "column", "tier", "group", "definition"]].eq("").any().any()
    assert set(reg["tier"]) <= set(measures.TIERS)
    assert sorted(reg["signifier"].dropna().unique()) == [1, 2, 3, 4, 5, 6]


def test_registry_filters() -> None:
    women = measures.measures(tournament="women")
    assert "hsr_without_ball_per_dominance" not in set(women["measure"])
    assert "hsr_without_ball_per_dominance" in set(measures.measures(tournament="men")["measure"])
    assert set(measures.measures(group="dominance")["measure"]) == {
        "contests_share", "territory_share", "chances_share", "dominance"
    }
    assert measures.label("territory_share") == "Field tilt"
    assert measures.measure("security")["higher_is_better"] is False
    with pytest.raises(ValueError):
        measures.measure("nope")


def test_available_keeps_present_columns_only() -> None:
    table = pd.DataFrame({"territory_share": [0.5], "dominance": [np.nan]})
    assert measures.available(table, measures.load_measures())["measure"].tolist() == ["territory_share"]


# ---------------------------------------------------------------- DuckDB layer


def test_db_views_over_build_folder(tmp_path: Path) -> None:
    cfg = synthetic.demo_config(tmp_path)
    write_table(cfg, "alpha", pd.DataFrame({"x": [1, 2, 3]}))
    write_table(cfg, "tracking/M01/running", pd.DataFrame({"x": [9]}))
    assert db.tables(cfg) == ["alpha"]
    assert db.query(cfg, "SELECT sum(x) AS s FROM alpha WHERE x > ?", [1])["s"].iloc[0] == 5


# ---------------------------------------------------------------- synthetic demo build


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    return synthetic.demo_tables()


def test_demo_tournament_formats(tables: dict[str, pd.DataFrame]) -> None:
    m = tables["matches"]
    counts = m.groupby("tournament")["match_id"].nunique().to_dict()
    assert counts == {"men": 104, "women": 52}
    assert (m["shootout"] == m["shootout_winner"].notna()).all()
    assert (m.loc[m["shootout"], "home_score"] == m.loc[m["shootout"], "away_score"]).all()
    assert set(tables["tds_team"]["team"]) == set(tables["elo_pre_tournament"].query("tournament == 'men'")["team"])


def test_demo_team_match_matches_registry_and_mirrors(tables: dict[str, pd.DataFrame]) -> None:
    tm = tables["team_match"]
    assert set(measures.measures(table="team_match")["column"]) <= set(tm.columns)
    assert len(tm) == 2 * len(tables["matches"])
    for col in ("territory_share", "contests_share", "chances_share", "possession_share"):
        assert np.allclose(tm.groupby(["tournament", "match_id"])[col].sum(), 1.0)
    assert tm.loc[tm["tournament"] == "women", "hsr_km"].isna().all()


def test_demo_running_is_men_only_and_nameless(tables: dict[str, pd.DataFrame]) -> None:
    running = tables["running_player_match"]
    assert set(running["tournament"]) == {"men"}
    assert not {"name", "player_name", "person_name"} & set(running.columns)
    assert set(running["position_group"]) == {"Goalkeeper", "Defender", "Midfielder", "Forward"}


def test_write_demo_refuses_real_folders(tmp_path: Path) -> None:
    real = synthetic.demo_config(tmp_path / "real")
    write_table(real, "matches", pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="not synthetic"):
        synthetic.write_demo(tmp_path / "real", run_expectation=False)
    with pytest.raises(ValueError, match="real data root"):
        synthetic.write_demo(tmp_path / "other", run_expectation=False, real_data_root=tmp_path / "other")
    root = repo_root()
    if root is not None:
        with pytest.raises(ValueError, match="outside the repo"):
            synthetic.write_demo(root / "data" / "demo", run_expectation=False)


def test_demo_data_command_writes_marked_build(tmp_path: Path) -> None:
    assert cli.main(["demo-data", "--out", str(tmp_path), "--small", "--no-expectation"]) == 0
    cfg = synthetic.demo_config(tmp_path)
    assert synthetic.is_synthetic(cfg)
    assert {"matches", "team_match", "team_match_zone", "running_player_match"} <= set(db.tables(cfg))
    assert read_table(cfg, "matches").groupby("tournament")["match_id"].nunique().to_dict() == {"men": 52, "women": 32}


ROUND2_CONTRACTS = {
    "player_match_minutes": ["tournament", "match_id", "team", "player_id", "jersey_number", "position",
                             "position_group", "started", "minutes", "on_minute", "off_minute"],
    "team_match_phase": ["tournament", "match_id", "team", "phase", "possessions", "possession_seconds", "shots",
                         "xg", "obv", "xg_per_possession", "share_of_possessions"],
    "team_tournament_phase": ["tournament", "team", "phase", "possessions", "possession_seconds", "shots", "xg",
                              "obv", "xg_per_possession", "share_of_possessions"],
    "team_match_provider": ["tournament", "match_id", "team", *synthetic.PROVIDER_COLUMNS],
    "phase_clips": ["tournament", "match_id", "team", "phase", "period", "start_s", "end_s"],
    "running_fifa_phase": ["tournament", "match_id", "team", "position_group", "phase", "seconds", "distance_m",
                           "hsr_m", "sprint_m"],
    "shape_team_second": ["tournament", "match_id", "team", "period", "clock_s", "state", "line_height_m",
                          "team_length_m", "team_width_m"],
    "event_running_windows": ["tournament", "match_id", "event_id", "team", "type", "location_x", "location_y",
                              "team_hsr_10s_m", "team_distance_10s_m"],
    "player_lookup": ["tournament", "match_id", "team", "statsbomb_player_id", "fifa_player_id",
                      "tracking_person_id", "jersey_number", "position_group", "method"],
    "provider_check_team_match": ["tournament", "match_id", "team", "measure", "statsbomb_value", "fifa_value",
                                  "statsbomb_definition", "fifa_definition"],
}


def test_round2_tables_follow_their_contracts(tables: dict[str, pd.DataFrame]) -> None:
    for name, columns in ROUND2_CONTRACTS.items():
        assert set(columns) <= set(tables[name].columns), (name, set(columns) - set(tables[name].columns))
    report = tables["report_team_match"]
    assert {"possession_pct", "attempts_on_target", "pass_completion_pct", "zone4_km"} <= set(report.columns)
    assert report.filter(like="phase_").shape[1] == 17
    tt = tables["team_tournament"]
    assert len(tt) == tables["elo_pre_tournament"].shape[0] and tt["matches"].min() >= 3
    assert set(tables["running_player_match"].columns) >= {"tracking_team", "frames", "dist_25_plus_kmh_m"}


def test_round2_men_only_tables_and_privacy(tables: dict[str, pd.DataFrame]) -> None:
    for name in ("team_match_provider", "phase_clips", "running_fifa_phase", "shape_team_second",
                 "event_running_windows", "player_lookup", "fifa_events"):
        assert set(tables[name]["tournament"]) == {"men"}, name
    for name, df in tables.items():
        assert not {"player_name", "person_name", "name", "date_of_birth"} & set(df.columns), name
    provider_table = tables["team_match_provider"]
    shares = provider_table[[c for c in provider_table.columns if c.startswith("fifa_")]].sum(axis=1)
    assert np.allclose(shares, 1)
    assert tables["event_running_windows"]["event_id"].is_unique
    assert tables["player_match_minutes"].groupby(["tournament", "match_id", "team"])["started"].sum().eq(11).all()
