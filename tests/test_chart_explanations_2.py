from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from three_trees import synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNED_PAGES = [
    REPO_ROOT / "app" / "pages" / "4_Style_explorer.py",
    REPO_ROOT / "app" / "pages" / "5_Running_and_load.py",
    REPO_ROOT / "app" / "pages" / "6_Ask.py",
    REPO_ROOT / "app" / "pages" / "7_Method_and_limits.py",
]


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("chart-explanations-demo") / "data"
    cfg = synthetic.write_demo(
        root, scale="small", luck_sims=100, progress_sims=100, run_models=True,
    )
    return cfg.data_root


@pytest.fixture(autouse=True)
def demo_environment(demo_root: Path) -> Iterator[None]:
    previous_root = os.environ.get("THREE_TREES_DATA_ROOT")
    previous_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ["THREE_TREES_DATA_ROOT"] = str(demo_root)
    yield
    if previous_root is None:
        os.environ.pop("THREE_TREES_DATA_ROOT", None)
    else:
        os.environ["THREE_TREES_DATA_ROOT"] = previous_root
    if previous_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = previous_key


@pytest.mark.parametrize("page", OWNED_PAGES, ids=lambda path: path.name)
@pytest.mark.parametrize("tournament", ["men", "women"])
def test_owned_page_explains_every_chart(page: Path, tournament: str) -> None:
    app = AppTest.from_file(str(page), default_timeout=60).run()
    app.sidebar.selectbox(key="tournament").set_value(tournament).run()

    assert not app.exception
    assert len(app.caption) >= len(app.get("plotly_chart")), (
        f"{page.name} ({tournament}) has fewer captions than charts"
    )


@pytest.mark.parametrize("tournament,question", [
    ("men", "Top 5 teams by field tilt"),
    ("women", "Top 5 women teams by field tilt"),
])
def test_ask_chart_has_a_plain_explanation(tournament: str, question: str) -> None:
    page = REPO_ROOT / "app" / "pages" / "6_Ask.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()
    app.sidebar.selectbox(key="tournament").set_value(tournament).run()
    app.text_input[0].input(question).run()

    assert not app.exception
    assert app.get("plotly_chart")
    assert any("shows the numbers behind the answer above" in item.value for item in app.caption)
