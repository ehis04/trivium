from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trivium.charts.running import (
    cost_of_dominance,
    fifa_phase_profile,
    running_heatmap,
    shape_profile,
)
from trivium.tracking.running import filled_in_share


def test_cost_of_dominance_uses_medians_and_marks_quadrants() -> None:
    teams = pd.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "matches": [3, 3, 3, 3],
            "dominance": [0.4, 0.6, 0.8, 0.2],
            "hsr_without_ball_km": [4.0, 6.0, 8.0, 2.0],
        }
    )

    figure = cost_of_dominance(teams, "A")

    assert figure.layout.shapes[0].x0 == pytest.approx(50)
    assert figure.layout.shapes[1].y0 == pytest.approx(5)
    assert {annotation.text for annotation in figure.layout.annotations} == {
        "Dominant at low cost",
        "Dominant at high cost",
        "Less dominant, low cost",
        "Chasing the game",
    }
    selected = next(trace for trace in figure.data if trace.name == "A")
    assert list(selected.x) == pytest.approx([40])
    assert list(selected.y) == pytest.approx([4])
    assert list(selected.customdata[0]) == ["A", 3]


def test_running_heatmap_calculates_group_running_per_90() -> None:
    running = pd.DataFrame(
        {
            "team": ["A", "A", "B"],
            "position_group": ["Defender", "Defender", "Defender"],
            "state": ["without_ball"] * 3,
            "third": ["defensive"] * 3,
            "hsr_m": [100, 300, 600],
            "minutes": [30, 60, 90],
        }
    )

    figure = running_heatmap(running, "A")

    defender = figure.data[0]
    assert defender.z[1][0] == pytest.approx(400)
    assert defender.customdata[1][0] == pytest.approx(500)
    assert len(figure.data) == 3


def test_fifa_phase_profile_keeps_order_and_skips_missing_phase() -> None:
    table = pd.DataFrame(
        {
            "team": ["A", "A", "B"],
            "fifa_build_up_share": [0.2, 0.4, 0.3],
            "fifa_progression_share": [0.1, 0.3, 0.2],
            "fifa_final_third_share": [0.4, 0.2, 0.1],
            "fifa_press_share": [0.3, 0.1, 0.4],
            "fifa_block_share": [0.0, 0.0, 0.0],
            "fifa_recovery_share": [np.nan, np.nan, np.nan],
        }
    )

    figure = fifa_phase_profile(table, "A")

    assert list(figure.data[0].y) == [
        "FIFA phase: build-up",
        "FIFA phase: progression",
        "FIFA phase: final third",
        "FIFA phase: press",
        "FIFA phase: block",
    ]
    assert list(figure.data[1].x) == pytest.approx([30, 20, 30, 20, 0])


def test_shape_profile_shows_team_mean_and_tournament_range() -> None:
    table = pd.DataFrame(
        {
            "team": ["A", "A", "B"],
            "line_height_m": [30, 50, 40],
            "team_length_m": [20, 30, 40],
            "team_width_m": [30, 50, 40],
        }
    )

    figure = shape_profile(table, "A")

    range_trace = figure.data[0]
    assert list(range_trace.x[:3]) == [40.0, 40.0, None]
    assert list(range_trace.y[:3]) == [
        "Defensive line height (metres)",
        "Defensive line height (metres)",
        None,
    ]
    assert list(figure.data[1].x) == pytest.approx([40, 32.5, 40])
    assert list(figure.data[2].x) == pytest.approx([40, 25, 40])


def test_filled_in_share_aggregates_frames_per_match() -> None:
    running = pd.DataFrame(
        {
            "tournament": ["men", "men", "men"],
            "match_id": [1, 1, 2],
            "frames": [100, 100, 200],
            "imputed_frames": [60, 80, 10],
        }
    )

    shares = filled_in_share(running).set_index("match_id")["filled_in_share"]

    assert shares[1] == pytest.approx(0.7)
    assert shares[2] == pytest.approx(0.05)
