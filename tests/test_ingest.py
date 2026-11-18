from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from three_trees.config import DATA_ROOT_ENV, load_config
from three_trees.ingest.statsbomb import (
    check_statsbomb,
    goals_by_team,
    load_events,
    parse_literal,
    prepare_matches,
    split_shootouts,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"
HOME, AWAY = 10, 20


def _event(match_id: int, index: int, period: int = 1, type_: str = "Pass", team_id: int = HOME, **extra) -> dict:
    return {
        "id": f"{match_id}-{index}",
        "match_id": match_id,
        "index": index,
        "period": period,
        "timestamp": "00:01:02.500",
        "minute": 1,
        "second": 2,
        "type": type_,
        "team": "Qatar U17" if team_id == HOME else "Italy U17",
        "team_id": team_id,
        "location": "[60.0, 40.0]",
        "shot_outcome": None,
        "under_pressure": None,
        **extra,
    }


def _events(rows: list[dict], tournament: str = "men") -> pd.DataFrame:
    ev = pd.DataFrame(rows)
    ev.insert(0, "tournament", tournament)
    return ev


def _matches(home_score: int = 0, away_score: int = 0) -> pd.DataFrame:
    raw = pd.DataFrame(
        [{
            "match_id": 1, "match_date": "2025-11-03", "kick_off": "15:15:00.000",
            "competition_stage": "Round of 32", "home_team": "Qatar U17", "away_team": "Italy U17",
            "home_team_id": HOME, "away_team_id": AWAY, "home_score": home_score, "away_score": away_score,
            "neutral_ground": False, "collection_status": "Complete", "data_version": "1.1.0",
            "home_manager_dob": "1970-01-01",
        }]
    )
    return raw


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    (tmp_path / "statsbomb").mkdir()
    return load_config(EXAMPLE)


def _write_parts(cfg, parts: list[list[dict]]) -> None:
    pattern = cfg.statsbomb["men"].events
    for i, rows in enumerate(parts, start=1):
        path = pattern.parent / pattern.name.replace("*", f"{i:03d}")
        pd.DataFrame(rows).to_csv(path, index=False)


def test_match_spanning_two_parts_is_one_match(cfg) -> None:
    _write_parts(cfg, [
        [_event(1, 1), _event(1, 2)],
        [_event(1, 3), _event(1, 4), _event(2, 1)],
    ])
    ev = load_events(cfg, "men")

    assert ev.groupby("match_id").size().to_dict() == {1: 4, 2: 1}
    assert ev.loc[ev["match_id"] == 1, "index"].tolist() == [1, 2, 3, 4]
    assert set(ev["team"]) == {"Qatar"}
    assert ev["team_name"].iloc[0] == "Qatar U17"
    assert ev["period_seconds"].iloc[0] == pytest.approx(62.5)
    assert ev["location_x"].iloc[0] == 60.0 and ev["location_y"].iloc[0] == 40.0


def test_flag_column_without_true_values_comes_out_bool(cfg) -> None:
    _write_parts(cfg, [
        [_event(1, 1), _event(1, 2)],
        [_event(1, 3, under_pressure=True)],
    ])
    ev = load_events(cfg, "men")

    assert ev["under_pressure"].dtype == bool
    assert ev["under_pressure"].tolist() == [False, False, True]


def test_parse_literal() -> None:
    assert parse_literal("[{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]") == [
        {"id": 1, "name": "A"},
        {"id": 2, "name": "B"},
    ]
    assert parse_literal(None) is None
    assert parse_literal(float("nan")) is None
    assert parse_literal("") is None


def test_period_5_goes_to_shootouts() -> None:
    rows = [_event(1, 1), _event(1, 2, type_="Half End")]
    outcomes = [(HOME, "Goal"), (AWAY, "Goal"), (HOME, "Saved"), (AWAY, "Goal")]
    for i, (team, outcome) in enumerate(outcomes, start=3):
        rows.append(_event(1, i, period=5, type_="Shot", team_id=team, shot_outcome=outcome))
    rows.append(_event(1, 7, period=5, type_="Half End"))
    ev = _events(rows)
    ev["team"] = ev["team"].str.replace(" U17", "")

    kept, shootouts = split_shootouts(ev)
    assert set(kept["period"]) == {1}
    assert len(kept) == 2
    so = shootouts.set_index("team")
    assert so.loc["Italy", "won"] and not so.loc["Qatar", "won"]
    assert so.loc["Italy", "kicks_scored"] == 2 and so.loc["Qatar", "kicks_taken"] == 2
    assert so["period5_events"].sum() == 5

    matches = prepare_matches(_matches(), "men", shootouts)
    assert matches.loc[0, "shootout"] and matches.loc[0, "shootout_winner"] == "Italy"
    assert "home_manager_dob" not in matches


def test_goals_count_own_goal_for_for_the_benefiting_team() -> None:
    ev = _events([
        _event(1, 1, type_="Shot", team_id=HOME, shot_outcome="Goal"),
        _event(1, 2, type_="Own Goal Against", team_id=HOME),
        _event(1, 3, type_="Own Goal For", team_id=AWAY),
        _event(1, 4, period=5, type_="Shot", team_id=AWAY, shot_outcome="Goal"),
    ])
    goals = goals_by_team(ev).set_index("team_id")["goals"].to_dict()
    assert goals == {HOME: 1, AWAY: 1}

    matches = prepare_matches(_matches(home_score=1, away_score=1), "men")
    _, shootouts = split_shootouts(ev)
    checks = check_statsbomb(ev, shootouts, matches).set_index("check")
    assert checks.loc["men: goals in periods 1-2 = home_score, away_score", "passed"]
