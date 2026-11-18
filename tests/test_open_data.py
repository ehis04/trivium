from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from three_trees.config import DATA_ROOT_ENV, load_config
from three_trees.features import dominance as dom
from three_trees.features import tables
from three_trees.ingest import open_data as od
from three_trees.ingest import statsbomb as sb

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"
LABEL = "senior_women_euro2025"
MATCH_ID = 9001
HOME, AWAY = {"id": 1, "name": "Spain Women's"}, {"id": 2, "name": "England Women's"}


def _event(i: int, type_: str, team: dict, period: int = 1, secs: int = 0, **detail) -> dict:
    e = {
        "id": f"ev{i}", "index": i, "period": period, "timestamp": f"00:{secs // 60:02d}:{secs % 60:02d}.000",
        "minute": (45 if period == 2 else 0) + secs // 60, "second": secs % 60,
        "type": {"id": 0, "name": type_}, "possession": 1 + i // 4, "possession_team": team,
        "play_pattern": {"id": 1, "name": "Regular Play"}, "team": team,
        "player": {"id": 100 + i, "name": "Senior Player"},
        "position": {"id": 3, "name": "Right Center Back"},
        "location": [60.0, 40.0],
    }
    return e | detail


def _lineup(team: dict, first: int) -> dict:
    lineup = [{"player": {"id": first + k, "name": "Senior Player"},
               "position": {"id": 1, "name": "Goalkeeper"}, "jersey_number": k + 1} for k in range(11)]
    return {"id": f"xi{team['id']}", "index": 0, "period": 1, "timestamp": "00:00:00.000", "minute": 0,
            "second": 0, "type": {"id": 35, "name": "Starting XI"}, "possession": 1, "possession_team": HOME,
            "play_pattern": {"id": 1, "name": "Regular Play"}, "team": team,
            "tactics": {"formation": 433, "lineup": lineup}}


def _events() -> list[dict]:
    events = [_lineup(HOME, 200), _lineup(AWAY, 300)]
    events += [
        _event(1, "Pass", HOME, secs=10, location=[70.0, 30.0], related_events=["ev2"],
               **{"pass": {"end_location": [100.0, 30.0], "recipient": {"id": 7, "name": "Senior Player"},
                           "shot_assist": True, "assisted_shot_id": "ev2",
                           "height": {"id": 1, "name": "Ground Pass"}}}),
        _event(2, "Shot", HOME, secs=12, location=[105.0, 40.0],
               shot={"statsbomb_xg": 0.4, "key_pass_id": "ev1", "outcome": {"id": 97, "name": "Goal"},
                     "type": {"id": 87, "name": "Open Play"}, "end_location": [120.0, 40.0, 1.0],
                     "freeze_frame": [{"player": {"id": 9, "name": "Senior Player"}, "location": [1, 2]}]}),
        _event(3, "Duel", AWAY, secs=20, duel={"type": {"id": 10, "name": "Aerial Lost"}}),
        _event(4, "50/50", AWAY, secs=30, **{"50_50": {"outcome": {"id": 108, "name": "Won"}}}),
        _event(5, "Foul Committed", AWAY, secs=40, foul_committed={"card": {"id": 5, "name": "Red Card"}}),
        _event(6, "Substitution", AWAY, period=2, secs=60,
               substitution={"replacement": {"id": 399, "name": "Senior Player"},
                             "outcome": {"id": 103, "name": "Tactical"}}),
        _event(7, "Shot", AWAY, period=3, secs=30, location=[110.0, 40.0],
               shot={"statsbomb_xg": 0.7, "outcome": {"id": 97, "name": "Goal"},
                     "type": {"id": 87, "name": "Open Play"}}),
        _event(8, "Shot", HOME, period=5, secs=0, shot={"outcome": {"id": 97, "name": "Goal"},
                                                        "type": {"id": 88, "name": "Penalty"}}),
        _event(9, "Shot", AWAY, period=5, secs=60, shot={"outcome": {"id": 98, "name": "Off T"},
                                                         "type": {"id": 88, "name": "Penalty"}}),
    ]
    return events


def _matches() -> list[dict]:
    return [{
        "match_id": MATCH_ID, "match_date": "2025-07-02", "kick_off": "18:00:00.000",
        "competition_stage": {"id": 1, "name": "Group Stage"},
        "home_team": {"home_team_id": 1, "home_team_name": "Spain Women's", "country": {"name": "Spain"}},
        "away_team": {"away_team_id": 2, "away_team_name": "England Women's", "country": {"name": "England"}},
        "home_score": 1, "away_score": 1, "match_status": "available", "metadata": {"data_version": "1.1.0"},
        "stadium": {"id": 1, "name": "x", "country": {"name": "Switzerland"}},
    }]


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))

    def no_network(*args, **kwargs):
        raise AssertionError("network used in a test")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    cfg = load_config(EXAMPLE)
    root = od.open_data_dir(cfg)
    cid, sid = od.COMPETITIONS[LABEL]
    (root / "matches" / str(cid)).mkdir(parents=True)
    (root / "events").mkdir()
    (root / "matches" / str(cid) / f"{sid}.json").write_text(json.dumps(_matches()), encoding="utf-8")
    (root / "events" / f"{MATCH_ID}.json").write_text(json.dumps(_events()), encoding="utf-8")
    return cfg


@pytest.fixture
def loaded(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    return od.load_tournament(cfg, LABEL)


def test_flatten_uses_provider_names_and_forms() -> None:
    row = od.flatten_event(_events()[2])
    assert row["type"] == "Pass" and row["team"] == "Spain Women's" and row["team_id"] == 1
    assert row["pass_recipient_id"] == 7 and row["pass_height"] == "Ground Pass"
    assert row["pass_end_location"] == "[100.0, 30.0]" and row["related_events"] == "['ev2']"
    ff = od.flatten_event(_events()[5])
    assert ff["50_50"] == "{'outcome': {'id': 108, 'name': 'Won'}}"
    assert sb.parse_literal(ff["50_50"])["outcome"]["name"] == "Won"


def test_no_player_names_are_kept(loaded) -> None:
    events, _ = loaded
    assert "player" not in events.columns and "pass_recipient" not in events.columns
    assert not events.astype(str).apply(lambda s: s.str.contains("Senior Player")).any().any()
    lineup = sb.parse_literal(events.loc[events["type"] == "Starting XI", "tactics"].iloc[0])["lineup"]
    assert set(lineup[0]["player"]) == {"id"} and lineup[0]["jersey_number"] == 1


def test_columns_and_forms(loaded) -> None:
    events, raw = loaded
    assert list(events.columns) == list(od.SENIOR_EVENT_COLUMNS)
    assert set(events["team"]) == {"Spain", "England"}
    assert set(raw["home_team"]) == {"Spain"} and bool(raw["neutral_ground"].iloc[0])
    shot = events[events["id"] == "ev2"].iloc[0]
    assert shot["location_x"] == 105.0 and shot["shot_key_pass_id"] == "ev1"
    assert events["obv_total_net"].isna().all() and events["location_bucket"].isna().all()
    assert events["is_transition"].dtype == bool and not events["is_transition"].any()
    assert events.loc[events["type"] == "Foul Committed", "foul_committed_card"].iloc[0] == "Red Card"


def test_goals_reconcile_and_shootout(loaded) -> None:
    events, raw = loaded
    kept, shootouts = sb.split_shootouts(events)
    matches = sb.prepare_matches(raw, LABEL, shootouts)
    assert 5 not in set(kept["period"]) and 3 in set(kept["period"])
    assert matches["shootout"].iloc[0] and matches["shootout_winner"].iloc[0] == "Spain"
    checks = od.check_open_data(events, matches).set_index("check")
    goals = f"{LABEL}: goals in periods 1-4 = home_score, away_score (extra time included)"
    assert bool(checks.loc[goals, "passed"])
    assert bool(checks.loc[f"{LABEL}: matches with two Starting XI events", "passed"])


def test_feature_code_reads_senior_events(loaded) -> None:
    events, raw = loaded
    kept, shootouts = sb.split_shootouts(events)
    matches = sb.prepare_matches(raw, LABEL, shootouts)
    tm, minutes = tables.build_team_match(kept, matches)
    spain = tm.set_index("team").loc["Spain"]
    assert spain["np_xg"] == pytest.approx(0.4) and spain["xa"] == pytest.approx(0.4)
    assert spain["aerials_won"] == 1 and tm.set_index("team").loc["England", "fifty_fifties_won"] == 1
    assert len(minutes[minutes["started"]]) == 22
    england = minutes[minutes["team"] == "England"].set_index("player_id")
    assert england.loc[399, "started"] == False  # noqa: E712
    assert len(dom.team_match_zone(kept, matches)) == 2 * dom.N_ZONES


def test_corrupt_saved_file_is_downloaded_again(cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    root = od.open_data_dir(cfg)
    (root / "events" / f"{MATCH_ID}.json").write_text('[{"id": ', encoding="utf-8")
    good = json.dumps(_events()).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(good))
    assert len(od.read_json("https://example.org", f"events/{MATCH_ID}.json", root)) == len(_events())
    assert json.loads((root / "events" / f"{MATCH_ID}.json").read_bytes()) == _events()


def test_truncated_response_is_fetched_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter([b'[{"id": ', json.dumps([{"id": "a"}]).encode()])
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(next(replies)))
    path = od.download("https://example.org", "events/1.json", tmp_path)
    assert json.loads(path.read_bytes()) == [{"id": "a"}]


def test_women_team_names_are_cleaned() -> None:
    assert od._team("Spain Women's") == "Spain" and od._team("WNT Finland") == "Finland"
