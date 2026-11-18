"""Tests for team shape (trivium.tracking.shape), FIFA phases (tracking.phases) and the
10-second windows before defensive actions (tracking.windows). All data is made up in the tests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trivium.tracking.outofplay import fifa_ball_alive, out_of_play_shares
from trivium.tracking.phases import phase_shares, read_phase_clips, running_by_phase, to_clock
from trivium.tracking.shape import shape_means, team_shape
from trivium.tracking.windows import defensive_actions, event_windows


# ---------------------------------------------------------------------------
# Team shape
# ---------------------------------------------------------------------------

def test_team_shape_on_a_known_formation() -> None:
    """A 4-4-2 laid out by hand: back four at 20-26 m, front two at 50 m, 10 m to 60 m wide."""
    x = np.array([20, 22, 24, 26, 35, 35, 35, 35, 50, 50])
    y = np.array([10, 25, 43, 58, 12, 28, 40, 56, 30, 38])
    line, length, width = team_shape(x, y)
    assert line == pytest.approx(23.0)   # mean of the four deepest: (20 + 22 + 24 + 26) / 4
    assert length == pytest.approx(30.0)  # 50 - 20
    assert width == pytest.approx(48.0)   # 58 - 10


def test_team_shape_needs_four_outfield_players() -> None:
    assert all(np.isnan(v) for v in team_shape(np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])))


def test_shape_means_use_seconds_without_the_ball() -> None:
    shape = pd.DataFrame({"match_id": 1, "team": "Aland", "state": ["without_ball", "without_ball", "with_ball"],
                          "line_height_m": [30.0, 40.0, 60.0], "team_length_m": 30.0, "team_width_m": 40.0})
    out = shape_means(shape, ["match_id", "team"])
    assert out["line_height_m"].tolist() == [35.0]


# ---------------------------------------------------------------------------
# FIFA phases
# ---------------------------------------------------------------------------

def write_enhanced(path: Path) -> Path:
    """A fake Enhanced file: kick-offs at 600 s and 4000 s of video, two phase rows, one player row."""
    def clip(start: float, end: float, game_time: str | None = None) -> dict:
        q = [{"category": "GAME TIME", "name": game_time}] if game_time else []
        return {"time_start": start, "time_end": end, "qualifiers": {"qualifiers_array": q}}
    rows = [
        {"row_name": "KICK OFF", "clips": [clip(600.0, 608.0, "First Half"), clip(4000.0, 4008.0, "Second Half")]},
        {"row_name": "ALAND_BUILD UP", "clips": [clip(610.0, 620.0), clip(4010.0, 4015.0)]},
        {"row_name": "ALAND_PRESS", "clips": [clip(615.0, 625.0)]},
        {"row_name": "ALAND_7 SOME PLAYER", "clips": [clip(610.0, 700.0)]},  # player row: never read
        {"row_name": "BLAND_BLOCK", "clips": [clip(590.0, 595.0), clip(630.0, 640.0)]},  # first clip before kick-off
    ]
    path.write_text(json.dumps({"row_count": len(rows), "rows": rows}))
    return path


def test_clip_to_clock_uses_each_halfs_kick_off() -> None:
    kicks = {1: 600.0, 2: 4000.0}
    assert to_clock(610.0, kicks) == (1, 10.0)
    assert to_clock(4010.0, kicks) == (2, 10.0)
    assert to_clock(590.0, kicks)[0] is None  # before the first kick-off


def test_read_phase_clips(tmp_path: Path) -> None:
    clips = read_phase_clips(write_enhanced(tmp_path / "e.json"))
    assert set(clips["phase"]) == {"build_up", "press", "block"}  # the player row is not a phase
    bu = clips[clips["phase"] == "build_up"]
    assert bu[["period", "start_s", "end_s"]].values.tolist() == [[1, 10.0, 20.0], [2, 10.0, 15.0]]
    assert len(clips[clips["fifa_team"] == "BLAND"]) == 1  # the clip before kick-off is left out


def test_phase_per_second_counts_overlaps_in_both_phases(tmp_path: Path) -> None:
    """Seconds 15-19 of half 1 are in both Build up (10-20) and Press (15-25)."""
    clips = read_phase_clips(write_enhanced(tmp_path / "e.json")).assign(team=lambda d: d["fifa_team"])
    seconds = pd.DataFrame({"team": "ALAND", "position_group": "Defender", "period": 1, "second": range(0, 60),
                            "distance_m": 5.0, "hsr_m": 1.0, "sprint_m": 0.0})
    out = running_by_phase(clips, seconds).set_index("phase")
    assert out.loc["build_up", "seconds"] == 10 and out.loc["press", "seconds"] == 10
    assert out.loc["build_up", "distance_m"] == pytest.approx(50.0)


def test_phase_shares_sum_to_one(tmp_path: Path) -> None:
    clips = read_phase_clips(write_enhanced(tmp_path / "e.json")).assign(team=lambda d: d["fifa_team"])
    shares = phase_shares(clips).set_index("team")
    aland = shares.loc["ALAND"]
    assert aland.sum() == pytest.approx(1.0)
    assert aland["fifa_build_up_share"] == pytest.approx(15 / 25)  # 10 + 5 s of build-up, 10 s of press


# ---------------------------------------------------------------------------
# Running in the 10 seconds before a defensive action
# ---------------------------------------------------------------------------

def test_ten_second_window_uses_the_sync_offset() -> None:
    """Team runs 1 m of HSR in each second 100-109 of the tracking clock; with a +5 s offset, an event at
    StatsBomb time 105 s happens at tracking second 110, so its window (100-109) holds all 10 m."""
    seconds = pd.DataFrame({"team": "Aland", "position_group": "Midfielder", "period": 1, "second": range(0, 200),
                            "hsr_m": [1.0 if 100 <= s < 110 else 0.0 for s in range(200)], "distance_m": 2.0,
                            "sprint_m": 0.0})
    events = pd.DataFrame({"id": ["a", "b", "c"], "match_id": 1, "period": 1, "period_seconds": [105.0, 105.0, 50.0],
                           "type": ["Pressure", "Duel", "Duel"], "duel_type": [None, "Tackle", "Aerial Lost"],
                           "team": "Aland", "location_x": 60.0, "location_y": 40.0})
    actions = defensive_actions(events)
    assert actions["id"].tolist() == ["a", "b"]  # an aerial duel is not a defensive action here
    out = event_windows(actions, seconds, {1: 5.0})
    assert out["team_hsr_10s_m"].tolist() == [10.0, 10.0]
    assert out["team_distance_10s_m"].tolist() == [20.0, 20.0]
    assert event_windows(actions, seconds, {2: 5.0}).empty  # no offset for that half: left out


# ---------------------------------------------------------------------------
# Out of play: our tracking against FIFA's ball_alive
# ---------------------------------------------------------------------------

def test_out_of_play_lines_up_on_frame_count(tmp_path: Path) -> None:
    """FIFA's timestamp runs on into the second half, but frame_count restarts: frames line up on it."""
    lines = []
    for period, offset in ((1, 0.0), (2, 3000.0)):  # half 2's timestamp carries on from half 1
        for second in range(4):
            lines.append(json.dumps({"match_period": period, "frame_count": second * 25,
                                     "timestamp": f"{int((offset + second) // 60):02d}:{(offset + second) % 60:09.6f}",
                                     "ball_alive": second != 3}))
    zip_path = tmp_path / "t.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("tracking_data_fifa_ifes.jsonl", "\n".join(lines))
    alive = fifa_ball_alive(zip_path)
    assert len(alive) == 8 and set(alive["frame"]) == {0, 25, 50, 75}

    # Our shape seconds: out of play in second 3 of each half (as FIFA) and also second 2 of half 2.
    shape = pd.DataFrame({"period": [1] * 4 + [2] * 4, "clock_s": [0.0, 1.0, 2.0, 3.0] * 2, "team": "Aland",
                          "state": ["with_ball", "with_ball", "without_ball", "out_of_play",
                                    "with_ball", "with_ball", "out_of_play", "out_of_play"]})
    shares = out_of_play_shares(shape, alive)
    assert shares["seconds"] == 8
    assert shares["fifa_out_of_play_share"] == pytest.approx(2 / 8)
    assert shares["tracking_out_of_play_share"] == pytest.approx(3 / 8)
    assert shares["agreement_share"] == pytest.approx(7 / 8)


def test_out_of_play_without_fifa_file_is_empty(tmp_path: Path) -> None:
    zip_path = tmp_path / "t.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("tracking_data.jsonl", "")
    assert fifa_ball_alive(zip_path).empty
