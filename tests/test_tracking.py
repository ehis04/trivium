"""Tests for reading the tracking zip (trivium.tracking.stream) and
running per player (trivium.tracking.running).

Every test builds its own small fake tracking zip, so no real data is used.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trivium.config import DATA_ROOT_ENV, load_config
from trivium.ingest.match_folders import PHYSICAL_COLUMNS
from trivium.storage import read_table, write_table
from trivium.tracking import stream
from trivium.tracking.running import (
    check_against_fifa,
    match_to_fifa,
    player_totals,
    possession_state,
    running_player_match,
    third,
)
from trivium.tracking.stream import PERIODS, iter_frames, parse_clock, read_metadata
from trivium.tracking.sync import left_teams, sync_match, to_statsbomb

# Fake tracking lines, written like the real file: one frame per line, with bare NaN
# for missing values (not valid JSON) and one blank line that should be skipped.
FRAMES = [
    b'{"frame_id": 1, "time": "00:00.000000", "period": "half_1", "ball": {"x": 52.5, "y": 34.0, "z": NaN}}',
    b'{"frame_id": 2, "time": "00:00.040000", "period": "half_1", "ball": {"x": NaN, "y": NaN, "z": NaN}}',
    b"",
    b'{"frame_id": 3, "time": "45:01.960000", "period": "half_1", "ball": {"x": 1.0, "y": 2.0, "z": 0.1}}',
    b'{"frame_id": 4, "time": "00:00.000000", "period": "half_2", "ball": {"x": 3.0, "y": NaN, "z": 0.0}}',
    b'{"frame_id": 5, "time": "12:30.500000", "period": "half_2", "ball": {"x": 4.0, "y": 5.0, "z": 0.0}}',
]

# Fake metadata.json, also with a bare NaN.
METADATA = b'{"MatchID": 99, "FrameRate": 25, "PitchLongSideMeters": 105, "PitchShortSideMeters": 68, "Phase3Start": NaN}'


def make_zip(path: Path, folder: str = "") -> Path:
    """Write a fake tracking zip. `folder` puts the files inside a folder in the zip."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{folder}tracking_data.jsonl", b"\n".join(FRAMES) + b"\n")
        zf.writestr(f"{folder}metadata.json", METADATA)
    return path


# Runs twice: files at the top of the zip, and files inside a folder in the zip.
@pytest.mark.parametrize("folder", ["", "match/"])
def test_iter_frames_parses_every_line_with_nan(tmp_path: Path, folder: str) -> None:
    """Every frame is read, the blank line is skipped, and NaN becomes None (empty)."""
    frames = list(iter_frames(make_zip(tmp_path / "t.zip", folder)))

    assert [f["frame_id"] for f in frames] == [1, 2, 3, 4, 5]
    assert frames[0]["ball"]["z"] is None
    assert frames[1]["ball"] == {"x": None, "y": None, "z": None}
    assert frames[3]["ball"]["x"] == 3.0


def test_iter_frames_does_not_extract(tmp_path: Path) -> None:
    """Reading the frames leaves nothing new on disk: the zip is never unzipped."""
    zip_path = make_zip(tmp_path / "t.zip")
    list(iter_frames(zip_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t.zip"]


def test_missing_member_is_named(tmp_path: Path) -> None:
    """Asking for a file that is not in the zip gives an error naming that file."""
    with pytest.raises(KeyError, match="tracking_data_fifa_ifes.jsonl"):
        list(iter_frames(make_zip(tmp_path / "t.zip"), "tracking_data_fifa_ifes.jsonl"))


def test_read_metadata(tmp_path: Path) -> None:
    """metadata.json is read from the zip: frame rate, pitch size, and NaN becomes None."""
    meta = read_metadata(make_zip(tmp_path / "t.zip"))
    assert meta["FrameRate"] == 25
    assert (meta["PitchLongSideMeters"], meta["PitchShortSideMeters"]) == (105, 68)
    assert meta["Phase3Start"] is None


# Each pair: clock text as in the tracking file, and the seconds it should give.
@pytest.mark.parametrize(
    "clock, seconds",
    [("00:00.000000", 0.0), ("00:00.040000", 0.04), ("12:30.500000", 750.5), ("45:01.960000", 2701.96)],
)
def test_parse_clock(clock: str, seconds: float) -> None:
    """Clock "MM:SS.ffffff" turns into seconds within the period."""
    assert parse_clock(clock) == pytest.approx(seconds)


def test_periods() -> None:
    """Tracking period names map to period numbers 1 to 4."""
    assert PERIODS == {"half_1": 1, "half_2": 2, "extra_time_1": 3, "extra_time_2": 4}


# ---------------------------------------------------------------------------
# Running per player (running.py)
# ---------------------------------------------------------------------------

# Two fake teams. In half 1 "Aland" attacks left to right; in half 2 they swap ends.
SIDES_H1 = {"left": "Aland", "right": "Bland"}
SIDES_H2 = {"left": "Bland", "right": "Aland"}


def kmh(v: float) -> float:
    """km/h to m/s, the unit of speed in the tracking file."""
    return v / 3.6


def player(pid: int, team: str, x: float | None, speed: float | None, imputed: bool = False) -> dict:
    """One fake player entry in a frame. person_name is there, as in the real file."""
    return {"person_id": pid, "person_name": "Name Hidden", "team_name": team, "jersey_number": pid % 100,
            "category_name": "player", "x": x, "y": 30.0, "speed": speed, "is_imputed": imputed}


def frame(period: str, clock: str, possession: str, players: list[dict], ball=(50.0, 30.0)) -> dict:
    """One fake frame."""
    return {"period": period, "time": clock, "ball_possession": possession,
            "pitch_side": SIDES_H1 if period == "half_1" else SIDES_H2,
            "ball": {"x": ball[0], "y": ball[1]}, "players": players}


def make_match_zip(path: Path, frames: list[dict]) -> Path:
    """Write a fake tracking zip holding these frames and a small metadata.json."""
    meta = {"MatchID": 7, "FrameRate": 25,
            "HomeTeam": {"Name": "Aland", "Players": [{"ID": 101, "Position": "M"}]},
            "AwayTeam": {"Name": "Bland", "Players": [{"ID": 201, "Position": "D"}]},
            # Half 1: home (Aland) attacks left to right; half 2: ends swapped. No extra time.
            "Phase1StartFrame": 1, "Phase1HomeGKLeft": True, "Phase2StartFrame": 1, "Phase2HomeGKLeft": False,
            "Phase3StartFrame": 0, "Phase3HomeGKLeft": False, "Phase4StartFrame": 0, "Phase4HomeGKLeft": False}
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("tracking_data.jsonl", "\n".join(json.dumps(f) for f in frames))
        zf.writestr("metadata.json", json.dumps(meta))
    return path


def test_possession_state() -> None:
    """Own team on the ball = with_ball; other team = without_ball; anything else = out_of_play."""
    assert possession_state("Aland", "Aland", SIDES_H1) == "with_ball"
    assert possession_state("Bland", "Aland", SIDES_H1) == "without_ball"
    assert possession_state("out of play", "Aland", SIDES_H1) == "out_of_play"
    assert possession_state("none", "Aland", SIDES_H1) == "out_of_play"


def test_third_ignores_spelling_of_the_left_team() -> None:
    """pitch_side may spell the team differently from team_name ("Switzerland U17" v "Switzerland")."""
    sides = {"left": "Switzerland U17", "right": "Egypt"}
    assert third(10.0, "Switzerland", sides) == "defensive"
    assert third(10.0, "Egypt", sides) == "attacking"


def test_possession_state_ignores_u17_case_and_accents() -> None:
    """Some tracking files write ball_possession as "Switzerland U17" or "Cote d'Ivoire" while
    team_name is "Switzerland" or "Côte d'Ivoire": still the same team."""
    sides = {"left": "Switzerland", "right": "Côte d'Ivoire"}
    assert possession_state("Switzerland U17", "Switzerland", sides) == "with_ball"
    assert possession_state("Switzerland U17", "Côte d'Ivoire", sides) == "without_ball"
    assert possession_state("Cote d'Ivoire", "Côte d'Ivoire", sides) == "with_ball"
    assert possession_state("out of play", "Switzerland", sides) == "out_of_play"


def test_third_for_both_directions() -> None:
    """x = 10 is the defensive third for the team attacking left to right,
    and the attacking third for the team attacking right to left (105 - 10 = 95)."""
    assert third(10.0, "Aland", SIDES_H1) == "defensive"
    assert third(10.0, "Bland", SIDES_H1) == "attacking"
    assert third(52.5, "Bland", SIDES_H1) == "middle"
    assert third(10.0, "Aland", SIDES_H2) == "attacking"  # ends swapped at half-time
    assert third(None, "Aland", SIDES_H1) == "unknown"


def test_running_distance_bands_and_states(tmp_path: Path) -> None:
    """Distance is speed / 25 per frame; the speed bands, states and thirds add up."""
    frames = [
        # Aland on the ball. Player 101 at 18 km/h (not high-speed), 201 at 22 km/h (high-speed).
        frame("half_1", "00:00.000000", "Aland", [player(101, "Aland", 10.0, kmh(18)), player(201, "Bland", 10.0, kmh(22))]),
        frame("half_1", "00:00.040000", "Aland", [player(101, "Aland", 10.0, kmh(18)), player(201, "Bland", 10.0, kmh(22))]),
        # Ball out of play. 101 at exactly 20 km/h (not high-speed: must be over 20), 201 at 26 km/h (sprint).
        frame("half_1", "00:00.080000", "out of play", [player(101, "Aland", 10.0, kmh(20)), player(201, "Bland", 10.0, kmh(26))]),
        # Half 2: Bland on the ball; 101 has no speed this frame and is imputed.
        frame("half_2", "00:00.000000", "Bland", [player(101, "Aland", 10.0, None, imputed=True), player(201, "Bland", 10.0, kmh(5))]),
    ]
    running, ball = running_player_match(make_match_zip(tmp_path / "t.zip", frames))
    totals = player_totals(running).set_index("person_id")

    # Player 101: two frames at 18 km/h + one at 20 km/h; the missing-speed frame adds nothing but is counted.
    p = totals.loc[101]
    assert p["frames"] == 4 and p["speed_missing_frames"] == 1 and p["imputed_frames"] == 1
    assert p["distance_m"] == pytest.approx((2 * kmh(18) + kmh(20)) / 25)
    assert p["hsr_m"] == 0 and p["sprint_m"] == 0
    assert p["dist_15_20_kmh_m"] == pytest.approx(p["distance_m"])
    assert p["minutes"] == pytest.approx(4 / 25 / 60)
    assert p["position"] == "M" and p["tracking_match_id"] == 7

    # Player 201: 22 km/h twice (high-speed), 26 km/h once (high-speed and sprint), 5 km/h once.
    p = totals.loc[201]
    assert p["hsr_m"] == pytest.approx((2 * kmh(22) + kmh(26)) / 25)
    assert p["sprint_m"] == pytest.approx(kmh(26) / 25)
    assert p["dist_20_25_kmh_m"] + p["dist_25_plus_kmh_m"] == pytest.approx(p["hsr_m"])
    band_sum = sum(p[c] for c in ["dist_1_7_kmh_m", "dist_7_15_kmh_m", "dist_15_20_kmh_m",
                                  "dist_20_25_kmh_m", "dist_25_plus_kmh_m"])
    assert band_sum == pytest.approx(p["distance_m"])

    # States and thirds for player 201 (Bland): attacks right to left in half 1, so x = 10 is
    # its attacking third; in half 2 it attacks left to right, so x = 10 is its defensive third.
    rows = running[running["person_id"] == 201].set_index(["state", "third"])["frames"].to_dict()
    assert rows == {("without_ball", "attacking"): 2, ("out_of_play", "attacking"): 1, ("with_ball", "defensive"): 1}

    # No player names anywhere in the output.
    assert "person_name" not in running.columns

    # Ball position per period: (clock seconds, x, y) for every frame.
    assert sorted(ball) == [1, 2]
    assert ball[1].shape == (3, 3)
    assert ball[1][:, 0] == pytest.approx([0.0, 0.04, 0.08])
    assert ball[2][0].tolist() == [0.0, 50.0, 30.0]


# ---------------------------------------------------------------------------
# Check against FIFA's physical file (running.py)
# ---------------------------------------------------------------------------

def fake_running(players: list[tuple[str, int, float, float, float]]) -> pd.DataFrame:
    """A fake running table: one row per player (team, shirt number, distance, high-speed, sprint).
    10% of each player's frames are imputed."""
    rows = []
    for i, (team, jersey, dist, hsr, sprint) in enumerate(players):
        rows.append({
            "tracking_match_id": 7, "team": team, "person_id": i, "jersey_number": jersey,
            "category_name": "player", "position": "M", "state": "with_ball", "third": "middle",
            "minutes": 90.0, "frames": 1000, "imputed_frames": 100, "speed_missing_frames": 0,
            "distance_m": dist, "hsr_m": hsr, "sprint_m": sprint,
            "dist_1_7_kmh_m": 0.0, "dist_7_15_kmh_m": 0.0, "dist_15_20_kmh_m": 0.0,
            "dist_20_25_kmh_m": hsr - sprint, "dist_25_plus_kmh_m": sprint,
        })
    return pd.DataFrame(rows)


def fake_physical(players: list[tuple[str, int, float, float, float]]) -> pd.DataFrame:
    """A fake FIFA physical table, as read_physical returns it (team, shirt, distance, high-speed, sprint)."""
    return pd.DataFrame([
        {"fifa_team_name": team, "jersey_number": jersey, "total_distance_m": dist,
         "dist_20_25_kmh_m": hsr - sprint, "dist_25_plus_kmh_m": sprint}
        for team, jersey, dist, hsr, sprint in players
    ])


# FIFA's figures for four players; team names in FIFA's upper case, one without its accent.
FIFA_PLAYERS = [
    ("AUSTRIA", 7, 10000, 800, 200), ("AUSTRIA", 9, 9000, 600, 100),
    ("COTE D'IVOIRE", 4, 11000, 900, 300), ("COTE D'IVOIRE", 1, 5000, 50, 10),
]


def test_fifa_check_passes_when_tracking_is_close() -> None:
    """Tracking 1% above FIFA for everyone: every row matches (ignoring case and accents), every check passes."""
    tracking = [(t.title().replace("Cote", "Côte").replace("D'I", "d'I"), j, d * 1.01, h * 1.01, s * 1.01)
                for t, j, d, h, s in FIFA_PLAYERS]
    checks = check_against_fifa(fake_running(tracking), fake_physical(FIFA_PLAYERS)).set_index("check")

    assert checks.loc["physical rows matched", "value"] == 4
    assert checks.loc["distance: median ratio tracking / FIFA", "value"] == pytest.approx(1.01)
    assert checks.loc["sprinting: correlation", "value"] == pytest.approx(1.0)
    assert checks.loc["team distance km, tracking (Austria)", "value"] == pytest.approx(19.19)
    assert checks.loc["imputed player-frames share", "value"] == pytest.approx(0.1)
    assert checks["passed"].all()


def test_fifa_check_flags_unmatched_rows_and_big_differences() -> None:
    """A FIFA row with no tracked player is counted as unmatched; tracking 5% above FIFA fails the 2% test."""
    tracking = [("Austria", 7, 10500, 800, 200), ("Austria", 9, 9450, 600, 100)]
    physical = fake_physical(FIFA_PLAYERS[:2] + [("AUSTRIA", 99, 100, 0, 0)])

    joined = match_to_fifa(fake_running(tracking), physical)
    assert joined["matched"].tolist() == [True, True, False]

    checks = check_against_fifa(fake_running(tracking), physical).set_index("check")
    assert checks.loc["physical rows matched", "value"] == 2
    assert not checks.loc["physical rows matched", "passed"]
    assert not checks.loc["team distance km, tracking (Austria)", "passed"]


# ---------------------------------------------------------------------------
# The tracking stage, run(cfg), on a fake data folder
# ---------------------------------------------------------------------------

def test_tracking_stage_runs_and_skips_done_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One fake match folder: the stage writes running, ball and checks tables, then skips it on a rerun."""
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")

    # A StatsBomb matches CSV holding the fake match.
    cfg.statsbomb["men"].matches.parent.mkdir(parents=True)
    pd.DataFrame({"match_id": [555], "match_date": ["2025-11-03"], "home_team": ["Aland U17"],
                  "away_team": ["Bland U17"]}).to_csv(cfg.statsbomb["men"].matches, index=False)

    # A match folder with a tracking zip (two frames) and a physical CSV.
    folder = cfg.match_folders_men / "M1 Aland - Bland"
    folder.mkdir(parents=True)
    frames = [frame("half_1", f"00:00.{i * 4:02d}0000", "Aland",
                    [player(101, "Aland", 10.0, kmh(18)), player(201, "Bland", 90.0, kmh(22))]) for i in range(2)]
    make_match_zip(folder / "20251103_FU17WC_01_ALA-BLA_RSV_Broadcast_Tracking_Data_Tactical_Wide.zip", frames)
    fifa_row = {c: 0 for c in PHYSICAL_COLUMNS}
    pd.DataFrame([{**fifa_row, "Team Name": "ALAND", "Jersey #": 1, "Total Distance (m)": 2 * kmh(18) / 25},
                  {**fifa_row, "Team Name": "BLAND", "Jersey #": 1, "Total Distance (m)": 2 * kmh(22) / 25}]
                 ).to_csv(folder / "20251103_FU17WC_01_ALA-BLA_FIFA_Player_Physical_Metrics_Data.csv", index=False)

    stream.run(cfg)

    # Per-match tables, keyed by the StatsBomb match_id, and the joined table.
    running = read_table(cfg, "tracking/555/running")
    assert set(running["match_id"]) == {555} and set(running["tournament"]) == {"men"}
    assert read_table(cfg, "tracking/555/ball")[["period", "clock_s"]].values.tolist() == [[1, 0.0], [1, 0.04]]
    assert len(read_table(cfg, "running_player_match")) == len(running)

    # Checks: the folder matched, one match processed, and FIFA's rows found for both players.
    checks = read_table(cfg, "checks_tracking").set_index("check")
    assert checks.loc["men: folders matched to a StatsBomb match_id", "passed"]
    assert checks.loc["men: matches processed this run", "value"] == 1
    assert checks.loc["men: M1 Aland - Bland: physical rows matched", "value"] == 2

    # Information rows (no pass or fail) stay empty, not failed.
    out_of_play = checks.loc["men: out of play: median share, tracking ball_possession", "passed"]
    assert out_of_play is None or pd.isna(out_of_play)

    # No StatsBomb events table yet (before H1): sync is skipped and the check says so.
    assert not checks.loc["men: sync: StatsBomb events table built (H1)", "passed"]
    assert checks.loc["men: matches with sync offsets", "value"] == 0

    # A second run skips the match because its tables already exist.
    stream.run(cfg)
    checks = read_table(cfg, "checks_tracking").set_index("check")
    assert checks.loc["men: matches processed this run", "value"] == 0
    assert checks.loc["men: matches skipped (already done)", "value"] == 1

    # Once the events table exists, the next run adds the sync for the match (without re-reading the tracking).
    write_table(cfg, "events", pd.DataFrame({
        "tournament": ["men"], "match_id": [555], "id": ["e1"], "period": [1], "period_seconds": [0.0], "type": ["Pass"], "duel_type": [None],
        "team": ["Aland"], "location_x": [50 * 120 / 105], "location_y": [30 * 80 / 68]}))
    stream.run(cfg)
    assert read_table(cfg, "tracking/555/sync")[["match_id", "period", "n_events"]].values.tolist() == [[555, 1, 1]]
    assert len(read_table(cfg, "sync_offsets")) == 1
    checks = read_table(cfg, "checks_tracking").set_index("check")
    assert "men: sync: StatsBomb events table built (H1)" not in checks.index
    assert checks.loc["men: matches with sync offsets", "passed"]


# ---------------------------------------------------------------------------
# Sync with StatsBomb (sync.py)
# ---------------------------------------------------------------------------

def test_to_statsbomb_with_and_without_mirror() -> None:
    """Tracking metres -> StatsBomb yards; mirrored for the team attacking right to left."""
    x, y = to_statsbomb(np.array([0.0, 105.0, 52.5]), np.array([0.0, 68.0, 34.0]), mirror=False)
    assert x.tolist() == [0.0, 120.0, 60.0] and y.tolist() == [0.0, 80.0, 40.0]
    x, y = to_statsbomb(np.array([10.5]), np.array([6.8]), mirror=True)  # 105 - 10.5 = 94.5 m -> 108 yd
    assert x == pytest.approx([108.0]) and y == pytest.approx([72.0])


def test_left_teams_from_metadata() -> None:
    """Phase<n>HomeGKLeft true = home team attacks left to right in that period; unused periods left out."""
    meta = {"HomeTeam": {"Name": "Aland"}, "AwayTeam": {"Name": "Bland"},
            "Phase1HomeGKLeft": True, "Phase2HomeGKLeft": False, "Phase3HomeGKLeft": False, "Phase4HomeGKLeft": False,
            "Phase1StartFrame": 100, "Phase2StartFrame": 80, "Phase3StartFrame": 0, "Phase4StartFrame": 0}
    assert left_teams(meta) == {1: "Aland", 2: "Bland"}


def fake_ball_path(period: int) -> pd.DataFrame:
    """A fake ball moving smoothly around the pitch for 100 s, at 25 frames a second."""
    t = np.arange(0, 100, 0.04)
    return pd.DataFrame({"period": period, "clock_s": t,
                         "ball_x": 52.5 + 40 * np.sin(t / 5), "ball_y": 34 + 25 * np.cos(t / 7)})


def fake_events(ball: pd.DataFrame, offset: float, teams: list[str], left_team: str) -> pd.DataFrame:
    """Passes placed exactly where the fake ball is at event time + offset, from each event team's view."""
    times = np.round(np.linspace(20, 80, len(teams)) / 0.04) * 0.04  # on the 25 fps frame grid
    b = ball.set_index(np.round(ball["clock_s"], 2))
    bx = b.loc[np.round(times + offset, 2), "ball_x"].to_numpy()
    by = b.loc[np.round(times + offset, 2), "ball_y"].to_numpy()
    x, y = to_statsbomb(bx, by, mirror=np.array([t != left_team for t in teams]))
    return pd.DataFrame({"match_id": 9, "period": int(ball["period"].iloc[0]), "period_seconds": times,
                         "type": "Pass", "team": teams, "location_x": x, "location_y": y})


def test_grid_search_finds_known_offset() -> None:
    """Events made with a 2.4 s offset (both teams, so with and without the mirror): the search finds 2.4 s."""
    ball = fake_ball_path(period=1)
    events = fake_events(ball, 2.4, ["Aland", "Bland"] * 30, left_team="Aland")
    out = sync_match(events, ball, {1: "Aland"})

    assert out["offset_s"].tolist() == [2.4]
    assert out["median_error_yd"].iloc[0] == pytest.approx(0, abs=1e-6)
    assert out["n_events"].iloc[0] == 60


def test_sync_ignores_other_event_types_and_periods() -> None:
    """Only passes and shots with a location are used; each period gets its own offset."""
    b1, b2 = fake_ball_path(1), fake_ball_path(2)
    e1 = fake_events(b1, -1.0, ["Aland"] * 20, left_team="Aland")
    e2 = fake_events(b2, -5.0, ["Aland"] * 20, left_team="Bland")  # ends swapped: Aland now mirrored
    noise = e1.assign(type="Pressure", location_x=0.0)              # not used
    out = sync_match(pd.concat([e1, e2, noise]), pd.concat([b1, b2]), {1: "Aland", 2: "Bland"})

    assert out["offset_s"].tolist() == [-1.0, -5.0]
    assert out["n_events"].tolist() == [20, 20]


def test_grid_search_finds_offset_beyond_15_seconds() -> None:
    """Some real halves are about 30 s out (M37 Germany - Korea DPR); the wider search finds them."""
    t = np.arange(0, 200, 0.04)  # a 200 s fake ball path, so event time + 29.4 s stays on it
    ball = pd.DataFrame({"period": 1, "clock_s": t, "ball_x": 52.5 + 40 * np.sin(t / 5), "ball_y": 34 + 25 * np.cos(t / 7)})
    events = fake_events(ball, 29.4, ["Aland", "Bland"] * 20, left_team="Aland")
    out = sync_match(events, ball, {1: "Aland"})
    assert out["offset_s"].tolist() == [29.4]
    assert out["median_error_yd"].iloc[0] == pytest.approx(0, abs=1e-6)


def test_excluded_matches_by_filled_in_share() -> None:
    from trivium.tracking.running import EXCLUDE_FILLED_IN_SHARE, excluded_matches, filled_in_share

    running = pd.DataFrame({"tournament": "men", "match_id": [1, 1, 2, 2],
                            "frames": [100, 100, 100, 300], "imputed_frames": [80, 60, 10, 20]})
    shares = filled_in_share(running).set_index("match_id")["filled_in_share"]
    assert shares[1] == pytest.approx(0.7) and shares[2] == pytest.approx(0.075)
    out = excluded_matches(running)
    assert out["match_id"].tolist() == [1] and EXCLUDE_FILLED_IN_SHARE == 0.30
    assert excluded_matches(None).empty


def test_positions_window_sampled_and_turned_to_attacking_direction(tmp_path: Path) -> None:
    from trivium.tracking.positions import match_positions

    frames = []
    for n in range(50):  # two sampled seconds per half at every=25
        frames.append(frame("half_1", f"00:{n // 25:02d}.000000", "Aland",
                            [player(101, "Aland", 30.0, 1.0), player(201, "Bland", 30.0, 1.0)]))
    for n in range(50):
        frames.append(frame("half_2", f"05:{n // 25:02d}.000000", "Bland",
                            [player(101, "Aland", 30.0, 1.0), player(201, "Bland", 30.0, 1.0)]))
    out = match_positions(make_match_zip(tmp_path / "t.zip", frames)).set_index(["team", "period"])
    assert out.loc[("Aland", 1), "x_m"] == 30.0 and out.loc[("Aland", 1), "state"] == "with_ball"
    assert out.loc[("Aland", 2), "x_m"] == 75.0 and out.loc[("Aland", 2), "y_m"] == 38.0  # ends swapped
    assert out.loc[("Bland", 1), "x_m"] == 75.0 and out.loc[("Bland", 1), "state"] == "without_ball"
    assert out.loc[("Aland", 2), "window"] == 1 and out.loc[("Aland", 2), "window_start_min"] == 50.0
    assert out["seconds"].eq(2).all() and out.loc[("Bland", 1), "position_group"] == "Defender"
    assert "person_name" not in out
