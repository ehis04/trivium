from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from trivium.charts.team import phase_dot_plot, tournament_lines


def test_tournament_lines_selected_team_last_and_thick() -> None:
    rows = pd.DataFrame(
        {
            "team": ["Team A1", "Team A1", "Team A1", "Team A2", "Team A2", "Team A2", "Team A3", "Team A3"],
            "match_number": [1, 2, 3, 1, 2, 3, 1, 2],
            "dominance": [0.5, 0.7, 0.8, 0.6, 0.4, 0.5, 0.2, 0.3],
        }
    )

    figure = tournament_lines(rows, "dominance", "Overall dominance", "Team A2", {2: "Round of 16"})

    assert len(figure.data) == 3
    assert figure.data[-1].name == "Team A2"
    assert figure.data[-1].line.width == 3
    assert figure.data[0].line.width == 1.5
    assert any(shape["x0"] == 1.5 for shape in figure.layout.shapes if shape["type"] == "line")  # before match 2


def test_phase_dot_plot_uses_order_and_skips_missing_phases() -> None:
    team_phase = pd.DataFrame(
        {
            "team": ["Team A1", "Team A1", "Team A1", "Team A1"],
            "phase": ["settled", "transition", "counter_attack", "set_piece"],
            "possessions": [10, 8, 5, 3],
            "shots": [6, 2, 1, 0],
            "xg": [0.8, 0.4, 0.2, 0.1],
            "xg_per_possession": [0.08, 0.05, 0.04, 0.03],
            "share_of_possessions": [0.4, 0.2, 0.1, 0.07],
        }
    )
    tournament_phase = pd.DataFrame(
        {
            "team": ["Team A1", "Team A2", "Team A3", "Team A1", "Team A2", "Team A3", "Team A1", "Team A2", "Team A3", "Team A1"],
            "phase": ["settled", "settled", "settled", "transition", "transition", "transition", "counter_attack", "counter_attack", "counter_attack", "set_piece"],
            "xg_per_possession": [0.07, 0.06, 0.08, 0.04, 0.05, 0.04, 0.03, 0.02, 0.04, 0.02],
            "share_of_possessions": [0.35, 0.32, 0.37, 0.18, 0.16, 0.22, 0.09, 0.08, 0.12, 0.06],
        }
    )

    figure = phase_dot_plot(team_phase, tournament_phase, "Team A1")

    assert list(figure.data[0].x) == ["Settled", "Transition", "Counter-attack", "Set piece"]
    assert list(figure.data[1].x) == ["Settled", "Transition", "Counter-attack", "Set piece"]
    assert figure.data[1].y[0] == 0.08
    assert figure.data[0].y[0] == 0.07


def test_team_report_includes_total_tournament_and_phase_sections() -> None:
    page = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "2_Team_report.py")
    app = AppTest.from_file(page, default_timeout=60).run()

    assert not app.exception
    assert any(item.value == "How the team changed through the tournament" for item in app.subheader)
    assert any(item.value == "Chances and possession by phase" for item in app.subheader)
