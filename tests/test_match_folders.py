"""Tests for the match folder files (three_trees.ingest.match_folders).

Every test writes its own small fake file, so no real data is used.
"""

from __future__ import annotations

import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from three_trees.ingest.match_folders import (
    FIFA_EVENT_COLUMNS,
    PHYSICAL_COLUMNS,
    folder_info,
    map_folders,
    pair_team_names,
    physical_player_match,
    player_lookup_match,
    read_fifa_events,
    read_physical,
    statsbomb_lineups,
    team_key,
)


def write_physical_csv(path: Path, rows: list[dict] | None = None) -> Path:
    """Write a fake FIFA physical metrics CSV with the 16 columns of the real file."""
    rows = rows or [
        {"Team Name": "AUSTRIA", "Player ID": 501, "Jersey #": 7, "Total Distance (m)": 10000},
        {"Team Name": "ITALY", "Player ID": 601, "Jersey #": 9, "Total Distance (m)": 9000},
    ]
    full = []
    for r in rows:
        row = {"Competition Name": "FAKE CUP", "Match ID": 1, "Player Name": "Name Hidden"}
        row.update({c: 0 for c in PHYSICAL_COLUMNS if c not in r and c != "Match ID"})
        row.update(r)
        full.append(row)
    pd.DataFrame(full).to_csv(path, index=False)
    return path


def test_read_physical_renames_and_drops_names(tmp_path: Path) -> None:
    """Columns get the snake_case names; Player Name and Competition Name are not kept."""
    df = read_physical(write_physical_csv(tmp_path / "physical.csv"))

    assert list(df.columns) == list(PHYSICAL_COLUMNS.values())
    assert "Player Name" not in df.columns and "player_name" not in df.columns
    assert df["jersey_number"].tolist() == [7, 9]
    assert df["total_distance_m"].tolist() == [10000, 9000]


def test_read_physical_names_missing_columns(tmp_path: Path) -> None:
    """A file without a needed column gives an error naming that column."""
    path = tmp_path / "physical.csv"
    pd.DataFrame({"Team Name": ["ITALY"], "Jersey #": [9]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="Total Distance"):
        read_physical(path)


# ---------------------------------------------------------------------------
# Match folder -> StatsBomb match_id
# ---------------------------------------------------------------------------

# A fake StatsBomb matches table with three matches.
MATCHES = pd.DataFrame({
    "match_id": [1001, 1002, 1003],
    "match_date": ["2025-11-03", "2025-11-04", "2025-11-04"],
    "home_team": ["Qatar", "Côte d'Ivoire", "Mexico"],
    "away_team": ["Italy", "Korea Republic", "Czechia"],
})


def test_team_key_ignores_suffix_case_and_accents() -> None:
    """'U17', capitals and accents make no difference when comparing team names."""
    names = pd.Series(["COTE D'IVOIRE", "Côte d'Ivoire U17", "côte  d'ivoire"])
    assert len(set(team_key(names))) == 1


def test_team_key_joins_provider_names() -> None:
    """Different providers' names for the same team give the same key (via lookup/countries.csv)."""
    assert len(set(team_key(pd.Series(["USA", "United States U17"])))) == 1
    assert len(set(team_key(pd.Series(["North Korea", "Korea DPR U17", "KOREA DPR"])))) == 1
    assert len(set(team_key(pd.Series(["Korea Republic", "Korea DPR"])))) == 2


def test_map_folders() -> None:
    """Folders match on date + both teams, in either order and in any case; others are listed as unmatched."""
    folders = pd.DataFrame({
        "folder": ["M1 Qatar - Italy", "M5 Korea - Cote", "M9 Wrong day", "M10 Unknown"],
        "match_date": ["2025-11-03", "2025-11-04", "2025-11-05", "2025-11-04"],
        "home_team": ["Qatar", "KOREA REPUBLIC", "Mexico", "Atlantis"],   # M5: home and away swapped
        "away_team": ["Italy", "COTE D'IVOIRE", "Czechia", "Mexico"],
    })
    out = map_folders(folders, MATCHES).set_index("folder")

    assert out.loc["M1 Qatar - Italy", "match_id"] == 1001
    assert out.loc["M5 Korea - Cote", "match_id"] == 1002
    assert out["matched"].tolist() == [True, True, False, False]  # wrong date; unknown team
    assert out.loc["M9 Wrong day", "match_id"] is pd.NA


def test_map_folders_falls_back_to_fifa_names() -> None:
    """When the tracking names do not match (or there is no tracking), FIFA's events-metadata names are tried."""
    folders = pd.DataFrame({
        "folder": ["M6 Mexico - Czechia", "M73 No tracking"],
        "match_date": ["2025-11-04", "2025-11-03"],
        "home_team": ["Mexico", None],
        "away_team": ["CZE", None],                      # tracking abbreviation not in the alias list
        "fifa_home_team": ["MEXICO", "QATAR"],
        "fifa_away_team": ["CZECHIA", "ITALY"],
    })
    out = map_folders(folders, MATCHES)
    assert out["match_id"].tolist() == [1003, 1001]


def test_folder_info_reads_date_and_teams(tmp_path: Path) -> None:
    """Date from the file-name prefix; team names from metadata.json in the tracking zip."""
    folder = tmp_path / "M1 Qatar - Italy"
    folder.mkdir()
    (folder / "20251103_FU17WC_01_QAT-ITA_FIFA_Enhanced.json").write_text("{}")
    with zipfile.ZipFile(folder / "20251103_FU17WC_01_QAT-ITA_RSV_Broadcast_Tracking_Data_Tactical_Wide.zip", "w") as zf:
        zf.writestr("metadata.json", '{"HomeTeam": {"Name": "Qatar"}, "AwayTeam": {"Name": "Italy"}}')

    info = folder_info(folder)
    assert info == {"folder": "M1 Qatar - Italy", "match_date": date(2025, 11, 3),
                    "home_team": "Qatar", "away_team": "Italy",
                    "fifa_home_team": None, "fifa_away_team": None}  # no FIFA events metadata file here


def test_physical_player_match_adds_match_id_and_statsbomb_team(tmp_path: Path) -> None:
    """FIFA rows get the match_id and the StatsBomb team name; folders unmatched or without a CSV are left out."""
    csv = write_physical_csv(tmp_path / "physical.csv")  # AUSTRIA #7 and ITALY #9
    folders = pd.DataFrame({
        "folder": ["M102 Austria - Italy", "M5 Not matched", "M6 No CSV"],
        "matched": [True, False, True],
        "match_id": pd.array([4035980, pd.NA, 77], dtype="Int64"),
        "physical_csv": [csv, csv, None],
    })
    matches = pd.DataFrame({"match_id": [4035980, 77], "home_team": ["Austria U17", "Qatar U17"],
                            "away_team": ["Italy U17", "Mali U17"]})

    out = physical_player_match(folders, matches)

    assert out["match_id"].tolist() == [4035980, 4035980]
    assert out["team"].tolist() == ["Austria", "Italy"]
    assert set(out["tournament"]) == {"men"}
    assert "Player Name" not in out.columns


def test_pair_team_names_uses_the_other_team() -> None:
    """A name with no alias ("UAE") maps when the match's other team matches; two unknowns stay unmapped."""
    statsbomb = ["Costa Rica", "United Arab Emirates"]
    assert pair_team_names(["Costa Rica U17", "UAE"], statsbomb) == {
        "Costa Rica U17": "Costa Rica", "UAE": "United Arab Emirates"}
    assert pair_team_names(["USA", "Burkina Faso"], ["United States", "Burkina Faso"]) == {
        "USA": "United States", "Burkina Faso": "Burkina Faso"}
    assert pair_team_names(["Atlantis", "Lemuria"], statsbomb) == {}


# ---------------------------------------------------------------------------
# FIFA events and the player lookup
# ---------------------------------------------------------------------------

def test_read_fifa_events_drops_names_referees_and_shootouts(tmp_path: Path) -> None:
    """Team names paired with StatsBomb, game_period from half_time, no player names in the output."""
    path = tmp_path / "20251103_FU17WC_01_QAT-ITA_FIFA_Unified_Event_Data_Events.csv"
    base = {c: None for c in FIFA_EVENT_COLUMNS}
    rows = [
        {**base, "team_name": "QATAR", "half_time": 1, "from_player_name": "Name Hidden", "event": "pass"},
        {**base, "team_name": "ITALY", "half_time": 2, "from_player_name": "Name Hidden", "event": "shot"},
        {**base, "team_name": "TEAM REFEREES", "half_time": 1, "from_player_name": "Name Hidden", "event": "card"},
        {**base, "team_name": "ITALY", "half_time": 5, "from_player_name": "Name Hidden", "event": "penalty"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    out = read_fifa_events(path, ["Qatar", "Italy"])
    assert out["team"].tolist() == ["Qatar", "Italy"]
    assert out["game_period"].tolist() == ["first_half", "second_half"]
    assert not any("name" in c for c in out.columns)


def test_player_lookup_by_shirt_with_name_check_fallback() -> None:
    """Italy's goalkeeper is #12 in FIFA and tracking but #1 in StatsBomb: paired by the name check.
    A substitute only in StatsBomb stays listed as unmatched."""
    sb = pd.DataFrame({"team": "Italy", "statsbomb_player_id": [101, 102, 103], "jersey_number": [1, 9, 20],
                       "position": ["Goalkeeper", "Center Forward", "Left Back"],
                       "name_check": ["Alessandro Keeper Rossi", "Mario Striker", "Luca Unused"]})
    fifa = pd.DataFrame({"team": "Italy", "fifa_player_id": [501, 502], "jersey_number": [12, 9],
                         "name_check": ["ALESSANDRO ROSSI", "MARIO STRIKER"]})
    tracking = pd.DataFrame({"team": "Italy", "tracking_person_id": [901, 902], "jersey_number": [12, 9],
                             "position_group": ["Goalkeeper", "Forward"]})
    out = player_lookup_match(sb, fifa, tracking).set_index("statsbomb_player_id")

    assert out.loc[102, "method"] == "shirt number" and out.loc[102, "fifa_player_id"] == 502
    assert out.loc[101, "method"] == "name check"
    assert (out.loc[101, "fifa_player_id"], out.loc[101, "tracking_person_id"], out.loc[101, "jersey_number"]) == (501, 901, 12)
    assert out.loc[103, "method"] == "unmatched" and out.loc[103, "position_group"] == "Defender"
    assert not any("name" in c for c in out.columns)


def test_statsbomb_lineups_from_tactics() -> None:
    tactics = str({"formation": 442, "lineup": [
        {"player": {"id": 7, "name": "Name Hidden"}, "position": {"id": 1, "name": "Goalkeeper"}, "jersey_number": 1}]})
    events = pd.DataFrame({"match_id": [1, 1], "team": ["Italy", "Italy"], "type": ["Starting XI", "Tactical Shift"],
                           "tactics": [tactics, tactics]})
    out = statsbomb_lineups(events)
    assert out[["statsbomb_player_id", "jersey_number", "position"]].values.tolist() == [[7, 1, "Goalkeeper"]]


def test_player_lookup_substitute_by_name_and_goalkeeper_rule() -> None:
    """A substitute from a Substitution event (no shirt number) pairs by name; goalkeepers whose names differ
    completely pair by the goalkeeper rule (one unpaired goalkeeper each side)."""
    sb = pd.DataFrame({"team": "Italy", "statsbomb_player_id": [101, 104],
                       "jersey_number": pd.array([1, None], dtype="Int64"),
                       "position": ["Goalkeeper", "Center Forward"], "name_check": ["Alpha Beta", "Gianni Subby"]})
    fifa = pd.DataFrame({"team": "Italy", "fifa_player_id": [501, 504], "jersey_number": [12, 21],
                         "name_check": ["GAMMA DELTA", "GIANNI SUBBY"]})
    tracking = pd.DataFrame({"team": "Italy", "tracking_person_id": [901, 904], "jersey_number": [12, 21],
                             "position_group": ["Goalkeeper", "Forward"]})
    out = player_lookup_match(sb, fifa, tracking).set_index("statsbomb_player_id")
    assert out.loc[104, "method"] == "name check" and out.loc[104, "jersey_number"] == 21
    assert out.loc[101, "method"] == "goalkeeper" and out.loc[101, "tracking_person_id"] == 901


def test_player_lookup_only_one_left_and_tracking_names() -> None:
    """With no FIFA row, the tracking team sheet's name is used for the name check; a team's last unpaired
    player on each side pairs as "only one left"."""
    sb = pd.DataFrame({"team": "Fiji", "statsbomb_player_id": [1, 2], "jersey_number": pd.array([6, 8], dtype="Int64"),
                       "position": ["Left Back", "Right Back"], "name_check": ["Tomasi Wise", "Epeli Other"]})
    fifa = pd.DataFrame(columns=["team", "fifa_player_id", "jersey_number", "name_check"])
    tracking = pd.DataFrame({"team": "Fiji", "tracking_person_id": [11, 12], "jersey_number": [16, 18],
                             "position_group": ["Defender", "Defender"], "name_check": ["TOMASI WISE", "SOMEONE ELSE"]})
    out = player_lookup_match(sb, fifa, tracking).set_index("statsbomb_player_id")
    assert out.loc[1, "method"] == "name check" and out.loc[1, "tracking_person_id"] == 11
    assert out.loc[2, "method"] == "only one left" and out.loc[2, "tracking_person_id"] == 12
