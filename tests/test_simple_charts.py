from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from three_trees.charts import simple


def _teams() -> pd.DataFrame:
    return pd.DataFrame({"team": ["A", "B", "C", "D"], "x": [1.0, 2.0, 3.0, 4.0], "y": [0.1, 0.3, 0.5, 0.7]})


def test_ranked_bars_highlight_and_reference() -> None:
    fig = simple.ranked_bars(_teams().assign(v=[-1.0, 0.5, 2.0, 1.0]), "v", "Value", highlight="B")
    bar = fig.data[0]
    assert list(bar.y) == ["A", "B", "D", "C"] and bar.marker.color[1] == simple.ACCENT
    assert fig.layout.shapes[0]["x0"] == 0.0


def test_scatter_fit_line_and_numbers() -> None:
    fig, fit = simple.scatter_fit(_teams(), "x", "y", "X", "Y", highlight="C")
    assert fit["slope"] == pytest.approx(0.2) and fit["r"] == pytest.approx(1.0) and fit["n"] == 4
    assert len(fig.data) == 2 and fig.data[1].line.color == simple.ABOVE
    assert np.isnan(simple.best_fit(pd.Series([1.0, 1.0, 1.0]), pd.Series([1.0, 2.0, 3.0]))["slope"])


def test_profile_and_evidence_bars() -> None:
    summary = pd.DataFrame({"tournament": "men", "profile": ["P1", "P2"], "overperformance_per_match": [0.3, -0.4],
                            "n_teams": [5, 7]})
    fig = simple.profile_bars(summary, "men", "P1")
    assert list(fig.data[0].text) == ["7 teams", "5 teams"]
    evidence = pd.DataFrame({"tournament": "men", "point": ["Dominance", "Dominance"], "label": ["Chances", "Duels"],
                             "adjusted_difference_z": [1.0, 0.3], "adjusted_excludes_zero": [True, False]})
    bars = simple.evidence_bars(evidence, "men", "Dominance").data[0]
    assert list(bars.marker.color) == [simple.LIGHT_GREY, simple.ABOVE]


def test_team_vs_average_and_match_bars() -> None:
    fig = simple.team_vs_average(pd.Series({"a": 2.0, "b": 1.0}), pd.Series({"a": 1.0, "b": 2.0}),
                                 {"a": "A", "b": "B"})
    assert list(fig.data[0].x) == [200.0, 50.0]
    rows = pd.DataFrame({"team": ["T", "O"], "np_xg": [2.0, 1.0]})
    fig = simple.match_bars(rows, "T", "O", {"np_xg": "xG"})
    assert [t.name for t in fig.data] == ["T", "O"]


def test_thirds_shares_and_pitch() -> None:
    zones = pd.DataFrame({"zone_x": [0, 1, 2, 3, 4, 5], "team_actions": [9, 9, 5, 5, 1, 1],
                          "opponent_actions": [1, 1, 5, 5, 9, 9]})
    shares = simple.thirds_shares(zones)
    assert shares["Own third"] == pytest.approx(0.9) and shares["Final third"] == pytest.approx(0.1)
    figure = simple.thirds_pitch(zones, title="T")
    assert any("Final third" in t.get_text() for t in figure.axes[0].texts)
    plt.close(figure)
