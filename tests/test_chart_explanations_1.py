from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from trivium import synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    REPO_ROOT / "app" / "pages" / "1_Tactical_profiles.py",
    REPO_ROOT / "app" / "pages" / "2_Team_report.py",
    REPO_ROOT / "app" / "pages" / "3_Match_review.py",
]


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("chart-explanations-demo") / "data"
    return synthetic.write_demo(
        root,
        scale="small",
        luck_sims=200,
        progress_sims=200,
        run_models=True,
    ).data_root


@pytest.fixture(autouse=True)
def demo_environment(demo_root: Path) -> Iterator[None]:
    previous = os.environ.get("TRIVIUM_DATA_ROOT")
    os.environ["TRIVIUM_DATA_ROOT"] = str(demo_root)
    yield
    if previous is None:
        os.environ.pop("TRIVIUM_DATA_ROOT", None)
    else:
        os.environ["TRIVIUM_DATA_ROOT"] = previous


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.name)
@pytest.mark.parametrize("tournament", ["men", "women"])
def test_each_chart_has_a_caption_and_page_runs(
    page: Path,
    tournament: str,
    demo_root: Path,
) -> None:
    app = AppTest.from_file(str(page), default_timeout=60).run()
    if tournament == "women":
        app.sidebar.selectbox(key="tournament").set_value("women").run()

    assert not app.exception
    chart_count = sum(
        element.type in {"plotly_chart", "imgs"}
        for element in app.main
    )
    assert chart_count > 0
    assert len(app.caption) >= chart_count
