from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from three_trees.charts.formations import average_shape, formation_results, formation_timeline
from three_trees.features.formations import listed_formations
from three_trees.synthetic import demo_positions_window
from three_trees.tracking import formations


@pytest.mark.parametrize(
    ("depths", "expected"),
    [
        ([20, 21, 22, 23, 40, 41, 42, 43, 60, 61], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]),
        ([20, 21, 22, 40, 41, 42, 43, 44, 60, 61], [0, 0, 0, 1, 1, 1, 1, 1, 2, 2]),
    ],
)
def test_jenks_labels_and_formation_counts(depths: list[float], expected: list[int]) -> None:
    labels = formations.jenks_labels(depths, 3)
    assert labels.tolist() == expected
    assert formations.formation_string(depths) == ("4-4-2" if expected.count(0) == 4 else "3-5-2")


def test_gvf_matches_within_and_total_variance() -> None:
    assert formations.gvf([0, 0, 10, 10], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert formations.gvf([0, 0, 10, 10], [0, 0, 0, 0]) == pytest.approx(0.0)
    assert formations.gvf([2, 2, 2], [0, 1, 2]) == 0.0


def test_best_line_count_notebook_small_input_rules() -> None:
    assert formations.best_line_count([]) == 0
    assert formations.best_line_count([4, 4, 4]) == 1
    assert formations.best_line_count([1, 3, 1, 3]) == 2
    assert formations.best_line_count([1, 2, 3, 1, 2]) == 3
    assert formations.best_line_count([1, 2, 3, 4, 5]) == 3


def test_best_line_count_accepts_each_gvf_step_of_004(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = {3: 0.0, 4: 0.04, 5: 0.08}
    monkeypatch.setattr(formations, "jenks_labels", lambda values, count: np.full(len(values), count))
    monkeypatch.setattr(formations, "gvf", lambda values, labels: scores[int(labels[0])])
    assert formations.best_line_count(range(10)) == 5

    scores.update({3: 0.0, 4: 0.04, 5: 0.07})
    assert formations.best_line_count(range(10)) == 4


def _positions_for_window(window: int, n_players: int = 10) -> pd.DataFrame:
    rows = []
    for player in range(n_players):
        rows.append({
            "tournament": "men",
            "match_id": 1,
            "team": "North",
            "period": 1,
            "window": window,
            "window_start_min": window * 5,
            "state": "without_ball",
            "person_id": player,
            "position_group": "Defender",
            "x_m": float(player * 5),
            "y_m": float(player * 4),
            "seconds": 59.0 if window == 1 and player == 0 else 300.0,
        })
    return pd.DataFrame(rows)


def test_window_formation_filters_short_players_and_understrength_windows() -> None:
    positions = pd.concat([
        _positions_for_window(0),
        _positions_for_window(1),
        _positions_for_window(2, n_players=9),
    ])
    result = formations.team_window_formations(positions, min_players=10, min_seconds=60)
    assert result[["window", "players"]].to_dict("records") == [{"window": 0, "players": 10}]


def test_synthetic_positions_form_shapes_and_switch_in_some_windows() -> None:
    matches = pd.DataFrame({
        "tournament": ["men"],
        "match_id": [1],
        "home_team": ["Team A1"],
        "away_team": ["Team A2"],
    })
    positions = demo_positions_window(matches)
    detected = formations.team_window_formations(positions)
    assert {"4-4-2", "4-3-3"} <= set(detected["formation"])
    assert detected.groupby("team")["formation"].nunique().gt(1).any()


def test_team_match_mode_tie_uses_earlier_formation_and_counts_period_changes() -> None:
    rows = []
    for period, window, shape in [
        (1, 0, "4-4-2"),
        (1, 1, "3-5-2"),
        (1, 2, "3-5-2"),
        (2, 0, "4-4-2"),
    ]:
        rows.append({
            "tournament": "men",
            "match_id": 1,
            "team": "North",
            "period": period,
            "window": window,
            "window_start_min": (period - 1) * 45 + window * 5,
            "state": "without_ball",
            "formation": shape,
            "back_line_m": 20.0 + window,
            "depth_m": 40.0,
        })
    result = formations.team_match_formations(pd.DataFrame(rows)).iloc[0]
    assert result["main_formation"] == "4-4-2"
    assert result["main_share"] == pytest.approx(0.5)
    assert result["formations_used"] == 2
    assert result["changes"] == 1
    assert result["back_line_m"] == pytest.approx(20.75)


def test_listed_formations_read_starting_and_tactical_shift_literals() -> None:
    events = pd.DataFrame([
        {"tournament": "men", "match_id": 1, "team": "North", "period": 1, "period_seconds": 0,
         "type": "Starting XI", "tactics": "{'formation': 433}"},
        {"tournament": "men", "match_id": 1, "team": "North", "period": 1, "period_seconds": 900,
         "type": "Tactical Shift", "tactics": "{'formation': 4231}"},
        {"tournament": "men", "match_id": 1, "team": "North", "period": 1, "period_seconds": 2700,
         "type": "Half End", "tactics": None},
        {"tournament": "men", "match_id": 1, "team": "North", "period": 2, "period_seconds": 2700,
         "type": "Half End", "tactics": None},
    ])
    result = listed_formations(events).iloc[0]
    assert result["starting_formation"] == "4-3-3"
    assert result["main_formation"] == "4-2-3-1"
    assert result["formation_changes"] == 1
    assert result["back_line"] == "back 4"


def test_formation_charts_have_window_depths_shape_lines_and_result_intervals() -> None:
    window_rows = pd.DataFrame([
        {"team": "North", "match_id": 1, "state": state, "window_start_min": 0,
         "formation": "4-4-2", "back_line_m": 22.0, "front_line_m": 62.0}
        for state in ("with_ball", "without_ball")
    ])
    timeline = formation_timeline(window_rows, "North", 1)
    assert len(timeline.data) == 2
    assert timeline.layout.shapes[0].x0 == 45
    assert "Back line" in timeline.data[0].hovertemplate

    positions = pd.DataFrame([
        {"team": "North", "match_id": 1, "state": "without_ball", "period": 1, "window": 0,
         "window_start_min": 0, "person_id": player, "position_group": group,
         "x_m": depth, "y_m": width, "seconds": 60.0}
        for player, (depth, width, group) in enumerate([
            (20, 10, "Defender"), (21, 24, "Defender"), (22, 44, "Defender"), (23, 58, "Defender"),
            (40, 12, "Midfielder"), (41, 26, "Midfielder"), (42, 42, "Midfielder"), (43, 56, "Midfielder"),
            (60, 30, "Forward"), (61, 38, "Forward"),
        ])
    ])
    shape = average_shape(positions, "North", 1, "without_ball")
    assert len(shape.axes[0].collections) == 3
    assert any(text.get_text() == "4-4-2" for text in shape.axes[0].texts)
    plt.close(shape)

    formation_table = pd.DataFrame([
        {"tournament": "men", "team": team, "match_id": match_id, "main_formation": formation}
        for team, formation, match_id in [
            ("A", "4-4-2", 1), ("B", "4-4-2", 2), ("C", "4-3-3", 3), ("D", "4-3-3", 4),
        ]
    ])
    summary = pd.DataFrame({
        "tournament": ["men"] * 4,
        "team": ["A", "B", "C", "D"],
        "overperformance_per_match": [1.0, 3.0, -1.0, -3.0],
    })
    results = formation_results(formation_table, summary, "main_formation")
    assert list(results.data[0].x) == pytest.approx([-2.0, 2.0])
    assert list(results.data[0].error_x.array) == pytest.approx([1.96, 1.96])
    assert "(n=2)" in results.data[0].y[0]


def test_window_with_a_substitution_keeps_the_ten_longest_players() -> None:
    from three_trees.tracking.formations import team_window_formations

    depths = [20, 21, 22, 23, 45, 46, 47, 48, 70, 71]
    rows = [{"tournament": "men", "match_id": 1, "team": "A", "period": 1, "window": 0, "window_start_min": 0.0,
             "state": "without_ball", "person_id": i, "x_m": float(d), "seconds": 300} for i, d in enumerate(depths)]
    rows.append(rows[-1] | {"person_id": 99, "x_m": 72.0, "seconds": 120})  # substitute on for the last 2 minutes
    out = team_window_formations(pd.DataFrame(rows))
    assert out["players"].iloc[0] == 10 and out["formation"].iloc[0] == "4-4-2"
