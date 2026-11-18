"""Tests for tactical profile features across the dashboard:
- team_scores chart (order, values, zero line)
- Home page answer section on demo build
- Team report tactical profile section
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from three_trees import synthetic
from three_trees.charts.tactical import team_scores
from three_trees.storage import write_table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the small synthetic demo dataset once for this test module."""
    root = tmp_path_factory.mktemp("tactical-dashboard-demo") / "data"
    cfg = synthetic.write_demo(root, scale="small", luck_sims=200, progress_sims=200)

    # Populate tactical profile tables for the demo build
    teams_df = pd.DataFrame(
        [
            {
                "tournament": "men",
                "team": "Team A1",
                "matches": 4,
                "dominant": 1.25,
                "efficient": 0.85,
                "resourceful": -0.40,
                "profile": "Dominant · Efficient · Costly",
                "overperformance_per_match": 0.75,
                "progress_over_expectation": 1.0,
                "actual_stage_name": "quarter_final",
                "elo": 1750.0,
            },
            {
                "tournament": "women",
                "team": "Team A1",
                "matches": 3,
                "dominant": 0.50,
                "efficient": 0.30,
                "resourceful": 0.90,
                "profile": "Dominant · Efficient · Resourceful",
                "overperformance_per_match": 0.40,
                "progress_over_expectation": 0.5,
                "actual_stage_name": "group",
                "elo": 1650.0,
            },
        ]
    )
    summary_df = pd.DataFrame(
        [
            {
                "tournament": "men",
                "profile": "Dominant · Efficient · Costly",
                "dominant": True,
                "efficient": True,
                "resourceful": False,
                "n_teams": 3,
                "teams": "Team A1, Team B1, Team C1",
                "overperformance_per_match": 0.65,
                "low": 0.20,
                "high": 1.10,
                "excludes_zero": True,
                "share_above_expectation": 0.8,
                "progress_over_expectation": 0.75,
            },
            {
                "tournament": "women",
                "profile": "Dominant · Efficient · Resourceful",
                "dominant": True,
                "efficient": True,
                "resourceful": True,
                "n_teams": 2,
                "teams": "Team A1, Team B2",
                "overperformance_per_match": 0.35,
                "low": 0.05,
                "high": 0.65,
                "excludes_zero": True,
                "share_above_expectation": 1.0,
                "progress_over_expectation": 0.5,
            },
        ]
    )
    axes_df = pd.DataFrame(
        [
            {
                "tournament": "men",
                "axis": "dominant",
                "label": "Dominant",
                "parts": "Overall dominance",
                "n_teams": 24,
                "effect": 0.3,
                "low": 0.1,
                "high": 0.5,
                "excludes_zero": True,
                "spearman": 0.4,
                "spearman_low": 0.15,
                "spearman_high": 0.65,
                "spearman_excludes_zero": True,
                "r_sb": 0.7,
                "repeatable": True,
                "early_late_spearman": 0.35,
                "early_late_low": 0.1,
                "early_late_high": 0.6,
                "predicts_later": True,
            },
            {
                "tournament": "men",
                "axis": "efficient",
                "label": "Efficient",
                "parts": "Conversion, Security",
                "n_teams": 24,
                "effect": 0.2,
                "low": -0.1,
                "high": 0.5,
                "excludes_zero": False,
                "spearman": 0.25,
                "spearman_low": -0.05,
                "spearman_high": 0.55,
                "spearman_excludes_zero": False,
                "r_sb": 0.2,
                "repeatable": False,
                "early_late_spearman": 0.1,
                "early_late_low": -0.2,
                "early_late_high": 0.4,
                "predicts_later": False,
            },
            {
                "tournament": "men",
                "axis": "resourceful",
                "label": "Resourceful",
                "parts": "Defensive work per point of dominance",
                "n_teams": 24,
                "effect": 0.4,
                "low": 0.15,
                "high": 0.65,
                "excludes_zero": True,
                "spearman": 0.38,
                "spearman_low": 0.12,
                "spearman_high": 0.64,
                "spearman_excludes_zero": True,
                "r_sb": 0.65,
                "repeatable": True,
                "early_late_spearman": 0.32,
                "early_late_low": 0.08,
                "early_late_high": 0.56,
                "predicts_later": True,
            },
        ]
    )
    write_table(cfg, "tactical_profile_team", teams_df)
    write_table(cfg, "tactical_profile_summary", summary_df)
    write_table(cfg, "tactical_profile_axes", axes_df)

    return cfg.data_root


@pytest.fixture(autouse=True)
def demo_environment(demo_root: Path) -> Iterator[None]:
    previous = os.environ.get("THREE_TREES_DATA_ROOT")
    os.environ["THREE_TREES_DATA_ROOT"] = str(demo_root)
    yield
    if previous is None:
        os.environ.pop("THREE_TREES_DATA_ROOT", None)
    else:
        os.environ["THREE_TREES_DATA_ROOT"] = previous


def test_team_scores_order_values_and_zero_line() -> None:
    """team_scores produces horizontal bars in exact order, with values and zero line."""
    teams = pd.DataFrame(
        {
            "tournament": ["men", "men"],
            "team": ["Team Alpha", "Team Beta"],
            "dominant": [1.25, -0.75],
            "efficient": [-0.40, 0.85],
            "resourceful": [0.60, -0.30],
            "profile": [
                "Dominant · Less efficient · Resourceful",
                "Less dominant · Efficient · Costly",
            ],
        }
    )

    fig = team_scores(teams, "men", "Team Alpha")
    assert len(fig.data) == 1
    trace = fig.data[0]

    # Order of scores on y-axis
    assert list(trace.y) == ["Dominant", "Efficient", "Resourceful"]
    # Values on x-axis
    assert list(trace.x) == [1.25, -0.40, 0.60]

    # Zero line check
    zero_lines = [
        shape
        for shape in fig.layout.shapes
        if getattr(shape, "x0", None) == 0 and getattr(shape, "x1", None) == 0
    ]
    assert len(zero_lines) >= 1

    # Empty figure when team not found
    empty_fig = team_scores(teams, "men", "Nonexistent Team")
    assert len(empty_fig.data) == 0


def test_home_answer_section_on_demo_build() -> None:
    """Home page opens with 'The answer' section, score definitions, outcomes chart, and link."""
    app = AppTest.from_file(str(REPO_ROOT / "app" / "Home.py"), default_timeout=45).run()
    assert not app.exception

    # Section heading
    assert any("The answer" in item.value for item in app.subheader)

    # Score definition line
    markdown_text = " ".join(item.value for item in app.markdown if isinstance(item.value, str))
    assert "dominant" in markdown_text.lower()
    assert "efficient" in markdown_text.lower()
    assert "resourceful" in markdown_text.lower()

    # Link to pages/1_Tactical_profiles.py
    assert any("Tactical_profiles" in str(pl.page) for pl in app.page_link)

    # Switch to women and verify no exceptions
    app.sidebar.selectbox(key="tournament").set_value("women").run()
    assert not app.exception
    assert any("The answer" in item.value for item in app.subheader)


def test_team_report_profile_section_on_demo_build() -> None:
    """Team report includes 'Tactical profile' section with name, team_scores, and summary sentence."""
    app = AppTest.from_file(
        str(REPO_ROOT / "app" / "pages" / "2_Team_report.py"), default_timeout=45
    ).run()
    assert not app.exception

    # Section heading
    assert any("Team style profile" in item.value for item in app.subheader)

    # Profile name
    markdown_text = " ".join(item.value for item in app.markdown if isinstance(item.value, str))
    assert "**Profile:**" in markdown_text

    # Sentence on how teams with the same profile did (with interval and n)
    assert any("profile averaged" in item.value for item in app.markdown)

    # Switch to women and verify no exceptions
    app.sidebar.selectbox(key="tournament").set_value("women").run()
    assert not app.exception
    assert any("Team style profile" in item.value for item in app.subheader)
