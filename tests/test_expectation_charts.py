from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from trivium.charts import expectation as charts


def _team_table(tournament: str = "men", teams: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tournament": tournament,
            "team": list(teams),
            "overperformance": [2.0, -1.0, 0.5][: len(teams)],
            "expected_points": [3.0, 4.0, 5.0][: len(teams)],
            "actual_points": [5.0, 3.0, 5.5][: len(teams)],
        }
    )


def _expected(tournament: str, teams: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    has_round_32 = tournament == "men"
    actual_stages = [3, 1, 2][: len(teams)] if has_round_32 else [2, 0, 1][: len(teams)]
    labels = (
        ["Quarter-final", "Group", "Round of 16"][: len(teams)]
        if has_round_32
        else ["Quarter-final", "Group", "Round of 16"][: len(teams)]
    )
    return pd.DataFrame(
        {
            "tournament": tournament,
            "team": list(teams),
            "actual_stage": actual_stages,
            "actual_stage_name": labels,
            "expected_stage": [2.2, 1.1, 1.8][: len(teams)],
            "progress_over_expectation": [0.8, -1.1, 0.2][: len(teams)],
            "p_reach_round_of_32": [0.9, 0.5, 0.8][: len(teams)] if has_round_32 else np.nan,
        }
    )


def test_overperformance_chart_has_one_marker_per_team_with_optional_bands() -> None:
    teams = _team_table()
    figure = charts.overperformance_chart(teams, "men")
    bands = pd.DataFrame(
        {
            "tournament": ["men"] * 3,
            "team": ["A", "B", "C"],
            "band_low": [-1.0, np.nan, -2.0],
            "band_high": [1.0, np.nan, 2.0],
            "beyond_luck": ["above", "", ""],
        }
    )
    with_bands = charts.overperformance_chart(teams, "men", bands=bands)

    assert isinstance(figure, go.Figure)
    assert len(figure.data[0].x) == len(teams)
    assert len(with_bands.data[0].x) == len(teams)
    horizontal_bands = [shape for shape in with_bands.layout.shapes if shape.x0 != shape.x1]
    assert len(horizontal_bands) == 2
    assert with_bands.data[0].marker.symbol[0] == "diamond"
    assert charts.overperformance_chart(_team_table(teams=("Solo",)), "men").data[0].x.tolist() == [2.0]


def test_expected_actual_points_labels_and_one_team() -> None:
    teams = _team_table()
    figure = charts.expected_vs_actual_points(teams, "men", label_n=1)

    assert isinstance(figure, go.Figure)
    assert len(figure.data[0].x) == len(teams)
    assert sum(bool(label) for label in figure.data[0].text) == 2
    single = charts.expected_vs_actual_points(_team_table(teams=("Solo",)), "men")
    assert len(single.data[0].x) == 1


def test_expected_actual_stage_handles_men_women_and_single_team() -> None:
    men = charts.expected_vs_actual_stage(_expected("men"), "men")
    women = charts.expected_vs_actual_stage(_expected("women"), "women")
    single = charts.expected_vs_actual_stage(_expected("women", ("Solo",)), "women")

    assert isinstance(men, go.Figure) and len(men.data[0].x) == 3
    assert list(men.layout.xaxis.ticktext) == [
        "Group", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"
    ]
    assert list(women.layout.xaxis.ticktext) == [
        "Group", "Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"
    ]
    assert len(single.data[0].x) == 1
    assert [trace.name for trace in men.data] == ["Expected", "Actual"]
    assert list(men.data[0].x) == [1.1, 1.8, 2.2] and len(men.data[1].x) == 3
    stage_names_only = _expected("men").drop(columns="p_reach_round_of_32").assign(actual_stage_name="round_of_32")
    assert "Round of 32" in charts.expected_vs_actual_stage(stage_names_only, "men").layout.xaxis.ticktext


def test_calibration_chart_has_one_marker_per_bin_and_single_bin_works() -> None:
    calibration = pd.DataFrame(
        {
            "tournament": ["men"] * 3,
            "bin": [1, 2, 3],
            "p_win_mean": [0.2, 0.5, 0.8],
            "win_share": [0.1, 0.6, 0.9],
        }
    )
    figure = charts.calibration_chart(calibration, "men")
    single = charts.calibration_chart(calibration.head(1), "men")

    assert isinstance(figure, go.Figure)
    assert len(figure.data[0].x) == 3
    assert len(single.data[0].x) == 1
