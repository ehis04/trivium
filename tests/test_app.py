from __future__ import annotations

import os
import re
import shutil
import sys
import time
from statistics import median
from pathlib import Path
from collections.abc import Iterator

import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest

from three_trees import synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))
PAGES = [
    REPO_ROOT / "app" / "Home.py",
    REPO_ROOT / "app" / "pages" / "2_Team_report.py",
    REPO_ROOT / "app" / "pages" / "3_Match_review.py",
    REPO_ROOT / "app" / "pages" / "4_Style_explorer.py",
    REPO_ROOT / "app" / "pages" / "5_Running_and_load.py",
    REPO_ROOT / "app" / "pages" / "7_Method_and_limits.py",
    REPO_ROOT / "app" / "pages" / "1_Tactical_profiles.py",
]


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the small synthetic dashboard dataset once for this test module."""
    root = tmp_path_factory.mktemp("dashboard-demo") / "data"
    cfg = synthetic.write_demo(root, scale="small", luck_sims=200, progress_sims=200)
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


def _assert_no_player_detail(app: AppTest) -> None:
    restricted = re.compile(r"(person_id|jersey_number|player name|player_id)", re.IGNORECASE)
    for element in app.dataframe:
        frame = element.value
        if isinstance(frame, pd.DataFrame):
            assert not any(restricted.search(str(column)) for column in frame.columns)
            for column in frame:
                assert not frame[column].astype(str).str.fullmatch(r"\d{6,}").any()
    text = "\n".join(element.value for element in app.markdown if isinstance(element.value, str))
    assert not restricted.search(text)


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
def test_owned_pages_run_for_men_and_women_with_synthetic_banner(
    page: Path,
    demo_root: Path,
) -> None:
    app = AppTest.from_file(str(page), default_timeout=60).run()
    assert not app.exception
    assert any("Synthetic demo data" in warning.value for warning in app.warning)
    _assert_no_player_detail(app)

    app.sidebar.selectbox(key="tournament").set_value("women").run()
    assert not app.exception
    assert any("Synthetic demo data" in warning.value for warning in app.warning)
    _assert_no_player_detail(app)
    if page.name == "5_Running_and_load.py":
        assert any(
            "Men only: FIFA physical and broadcast tracking data cover the men's tournament." in item.value
            for item in app.info
        )


def test_missing_table_shows_pipeline_notice(demo_root: Path, tmp_path: Path) -> None:
    copied_root = tmp_path / "copy"
    shutil.copytree(demo_root / "build", copied_root / "build")
    (copied_root / "build" / "team_match.parquet").unlink()
    os.environ["THREE_TREES_DATA_ROOT"] = str(copied_root)

    app = AppTest.from_file(str(PAGES[1]), default_timeout=60).run()

    assert not app.exception
    assert any(
        "Not built yet: needs table `team_match` from the `features` stage." in item.value
        for item in app.info
    )


def test_presenter_mode_hides_tables_and_keeps_team_export(demo_root: Path) -> None:
    page = REPO_ROOT / "app" / "pages" / "2_Team_report.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()

    assert not app.exception
    assert app.dataframe
    assert len(app.get("download_button")) == 1

    app.sidebar.checkbox(key="presenter_mode").check().run()

    assert not app.exception
    assert not app.dataframe
    assert len(app.get("download_button")) == 1


def test_method_check_tables_are_arrow_serializable(demo_root: Path) -> None:
    page = REPO_ROOT / "app" / "pages" / "7_Method_and_limits.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()

    assert not app.exception
    assert app.dataframe
    stage_summary = next(
        element.value
        for element in app.dataframe
        if {"Stage", "Checks", "Passed", "Failed"}.issubset(element.value.columns)
    )
    assert stage_summary["Checks"].sum() == stage_summary["Passed"].sum() + stage_summary["Failed"].sum()
    for element in app.dataframe:
        pa.Table.from_pandas(element.value)


def test_style_comparison_has_registry_labels_and_tournament_ranks(demo_root: Path) -> None:
    page = REPO_ROOT / "app" / "pages" / "4_Style_explorer.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()

    assert not app.exception
    comparison = next(
        element.value
        for element in app.dataframe
        if {"Measure", "Team A1 rank", "Team A2 rank"}.issubset(element.value.columns)
    )
    assert "Contests won share" in comparison["Measure"].values
    assert comparison[["Team A1 rank", "Team A2 rank"]].notna().any().any()


def test_profile_comparison_displays_intervals_and_survivor_markers() -> None:
    from components.charts import profile_comparison

    table = pd.DataFrame(
        {
            "label": ["Signifier 2", "Signifier 3"],
            "raw_difference_z": [0.4, -0.2],
            "raw_low": [0.1, -0.5],
            "raw_high": [0.7, 0.1],
            "adjusted_difference_z": [0.3, -0.1],
            "adjusted_low": [0.0, -0.4],
            "adjusted_high": [0.6, 0.2],
            "survives_adjustment": [True, False],
        }
    )

    figure = profile_comparison(table)

    assert list(figure.data[0].error_y.array) == pytest.approx([0.3, 0.3])
    assert list(figure.data[0].error_y.arrayminus) == pytest.approx([0.3, 0.3])
    assert list(figure.data[1].error_y.array) == pytest.approx([0.3, 0.3])
    assert list(figure.data[2].marker.symbol) == ["diamond", "circle-open"]


def test_findings_use_direction_ranks_and_no_causal_language() -> None:
    from components.findings import build_findings

    team_match = pd.DataFrame(
        {
            "tournament": ["men"] * 4,
            "team": ["Team A1", "Team A1", "Team B1", "Team B1"],
            "territory_share": [0.8, 0.8, 0.4, 0.4],
            "security": [0.1, 0.1, 0.3, 0.3],
            "possession_share": [0.4, 0.4, 0.6, 0.6],
        }
    )
    findings = build_findings("Team A1", team_match, "men")

    assert len(findings) == 2
    assert any(
        "higher than the tournament average" in finding and "rank 1 of 2" in finding
        for finding in findings
    )
    assert any("lower than the tournament average" in finding for finding in findings)
    causal = re.compile(r"\b(causes?|leads to|results in)\b", re.IGNORECASE)
    assert not any(causal.search(finding) for finding in findings)


def test_pages_reload_from_cache_under_one_second(demo_root: Path) -> None:
    timings = {}
    for page in PAGES:
        app = AppTest.from_file(str(page), default_timeout=60).run()
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            app.run()
            samples.append(time.perf_counter() - started)
        timings[page.name] = median(samples)
    assert all(seconds < 1.0 for seconds in timings.values()), timings
