from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from trivium import synthetic
from trivium.charts.answer import point_evidence_chart
from trivium.definitions import MEASURE_POINTS, POINTS
from trivium.models.answer import point_evidence
from trivium.storage import write_table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))


def _profile_comparison() -> pd.DataFrame:
    labels = {
        "dominance": "Overall control",
        "contests_share": "Contests won share",
        "duels_share": "Tackles won share",
        "aerials_share": "Aerial duels won share",
        "fifty_fifties_share": "50/50s won share",
        "second_balls_share": "Second balls won share",
        "territory_share": "Field tilt",
        "final_third_actions": "Passes and carries in the final third",
        "line_height_m": "Defensive line height",
        "chances_share": "Chances share",
        "pressures": "Pressures",
        "pressures_per_opp_possession_min": "Pressures per minute of opponent possession",
        "defensive_actions": "Defensive actions",
        "possession_share": "Possession share",
        "possession_minutes": "Possession time",
        "distance_km": "Total distance",
        "hsr_km": "High-speed running",
        "sprint_km": "Sprinting",
        "hsr_without_ball_km": "High-speed running without the ball",
        "players_used": "Players used",
        "minutes_outside_top11_share": "Minutes outside the most-used 11",
        "starters_changed": "Starters changed",
        "top3_hsr_share": "Top-3 players' share of high-speed running",
        "defensive_work_per_dominance": "Defensive work per point of dominance",
        "hsr_without_ball_per_dominance": "High-speed running without the ball per point of dominance",
        "conversion": "Conversion",
        "security": "Security",
        "top2_xg_xa_share": "Top-2 players' share of xG + xA",
        "chance_zone_spread": "Spread of chance creation across zones",
    }
    rows = []
    for tournament in ("men", "women"):
        for index, (measure, (point, part)) in enumerate(MEASURE_POINTS.items()):
            if tournament == "women" and measure in {
                "hsr_without_ball_km",
                "top3_hsr_share",
                "hsr_without_ball_per_dominance",
                "distance_km",
                "hsr_km",
                "sprint_km",
            }:
                continue
            estimate = 0.35 if index % 2 == 0 else -0.25
            clear = index % 2 == 0
            low, high = (estimate - 0.1, estimate + 0.1) if clear else (-0.3, 0.3)
            rows.append(
                {
                    "tournament": tournament,
                    "measure": measure,
                    "label": labels.get(measure, measure.replace("_", " ").title()),
                    "adjusted_difference_z": estimate,
                    "adjusted_low": low,
                    "adjusted_high": high,
                    "adjusted_excludes_zero": clear,
                    "raw_difference_z": estimate / 2,
                    "raw_low": estimate / 2 - 0.1,
                    "raw_high": estimate / 2 + 0.1,
                    "survives_adjustment": clear,
                    "verdict": "survives adjustment" if clear else "no clear difference",
                }
            )
    return pd.DataFrame(rows)


def _write_research_question_tables(cfg: object) -> None:
    comparison = _profile_comparison()
    write_table(cfg, "profile_comparison", comparison)
    axes = []
    for tournament in ("men", "women"):
        for axis, label in (("dominant", "Dominant"), ("resourceful", "Resourceful"), ("efficient", "Efficient")):
            axes.append(
                {
                    "tournament": tournament,
                    "axis": axis,
                    "label": label,
                    "parts": "Team-level measures",
                    "n_teams": 20,
                    "effect": 0.2,
                    "low": -0.1,
                    "high": 0.5,
                    "spearman": 0.3,
                    "spearman_low": -0.05,
                    "spearman_high": 0.6,
                    "spearman_excludes_zero": False,
                    "repeatable": axis != "efficient",
                    "predicts_later": axis == "dominant",
                }
            )
    write_table(cfg, "tactical_profile_axes", pd.DataFrame(axes))

    summaries = []
    for tournament in ("men", "women"):
        for profile, value in (("Dominant · Efficient · Resourceful", 0.3), ("Less dominant · Costly · Less efficient", -0.1)):
            summaries.append(
                {
                    "tournament": tournament,
                    "profile": profile,
                    "n_teams": 8,
                    "teams": "Team A1, Team B1",
                    "overperformance_per_match": value,
                    "low": value - 0.2,
                    "high": value + 0.2,
                }
            )
    write_table(cfg, "tactical_profile_summary", pd.DataFrame(summaries))

    sensitivity = []
    for tournament in ("men", "women"):
        for score in ("dominant", "resourceful", "efficient"):
            for variant, same_sign in (("main", True), ("alternate definition", score != "efficient")):
                sensitivity.append(
                    {
                        "tournament": tournament,
                        "score": score,
                        "variant": variant,
                        "n_teams": 20,
                        "spearman": 0.3,
                        "low": -0.05,
                        "high": 0.6,
                        "excludes_zero": False,
                        "same_sign_as_main": same_sign,
                        "median_threshold_share": 0.8,
                        "leave_one_out_share": 0.9,
                    }
                )
    write_table(cfg, "tactical_profile_sensitivity", pd.DataFrame(sensitivity))

    gate = []
    for tournament in ("men", "women"):
        for axis, label in (("dominant", "Dominant score"), ("resourceful", "Resourceful score"), ("efficient", "Efficient score")):
            passed = axis == "dominant" and tournament == "men"
            gate.append(
                {
                    "finding_type": "tactical profile",
                    "tournament": tournament,
                    "measure": f"profile_{axis}",
                    "label": label,
                    "wording": f"The {label.lower()} is associated with points above expectation.",
                    "passes_gate": passed,
                    "test3_repeatable": True,
                    "test4_not_just_strength": True,
                    "test5_out_of_sample": passed,
                    "test6_interval_excludes_zero": passed,
                }
            )
    write_table(cfg, "finding_gate", pd.DataFrame(gate))


@pytest.fixture(scope="module")
def demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a small synthetic dataset and add the evidence tables used by this page."""
    root = tmp_path_factory.mktemp("research-question-demo") / "data"
    cfg = synthetic.write_demo(root, scale="small", luck_sims=50, progress_sims=50)
    _write_research_question_tables(cfg)
    return cfg.data_root


@pytest.fixture(autouse=True)
def demo_environment(demo_root: Path) -> Iterator[None]:
    previous = os.environ.get("TRIVIUM_DATA_ROOT")
    os.environ["TRIVIUM_DATA_ROOT"] = str(demo_root)
    yield
    if previous is None:
        os.environ.pop("TRIVIUM_DATA_ROOT", None)
    else:
        os.environ["TRIVIUM_DATA_ROOT"] = previous


def test_point_evidence_chart_groups_measures_and_marks_both_tournaments() -> None:
    evidence = point_evidence(_profile_comparison())

    fig = point_evidence_chart(evidence)

    assert len(fig.data) == 2
    assert [trace.name for trace in fig.data] == ["Men", "Women"]
    assert len(fig.data[0].x) > 0 and len(fig.data[1].x) > 0
    assert "<b>Dominance</b>" in fig.layout.yaxis.ticktext
    assert "<b>Resourcefulness</b>" in fig.layout.yaxis.ticktext
    assert "<b>Efficiency</b>" in fig.layout.yaxis.ticktext
    assert "<b>Contests</b>" in fig.layout.yaxis.ticktext
    assert "Teams above minus below their Elo expectation (standard deviations, 95% interval)" == fig.layout.xaxis.title.text
    assert "circle" in list(fig.data[0].marker.symbol)
    assert "circle-open" in list(fig.data[0].marker.symbol)

    dominance = point_evidence_chart(evidence, "Dominance")
    assert len(dominance.data) == 2
    assert all(len(trace.x) == evidence.loc[evidence["point"].eq("Dominance") & evidence["tournament"].eq(t)].shape[0]
               for trace, t in zip(dominance.data, ("men", "women")))


def test_research_question_page_shows_question_and_all_three_points() -> None:
    page = REPO_ROOT / "app" / "pages" / "0_Research_question.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()

    assert not app.exception
    assert any("Can we define dominant, resourceful and efficient tactical profiles" in item.value for item in app.markdown)
    assert all(any(item.value == point for item in app.subheader) for point in POINTS)
    assert any(item.value == "The short answer" for item in app.subheader)
    assert any(item.value == "How sure" for item in app.subheader)
    assert any(item.value == "What we cannot say" for item in app.subheader)
    assert len(app.get("plotly_chart")) >= 5
    score_results = [
        item.value for item in app.markdown
        if isinstance(item.value, str) and "the Dominant score: its rank correlation was" in item.value
    ]
    assert len(score_results) == 2
    assert any("For men, the Dominant score" in result for result in score_results)
    assert any("For women, the Dominant score" in result for result in score_results)


def test_slide_figure_manifest_lists_three_points_and_each_point(tmp_path: Path) -> None:
    from trivium.charts.export import write_figures

    cfg = synthetic.write_demo(tmp_path / "figure-demo", scale="small", luck_sims=50, progress_sims=50)
    write_table(cfg, "profile_comparison", _profile_comparison())
    try:
        import kaleido

        chrome = kaleido.get_chrome_sync()
        has_chrome = bool(chrome and Path(chrome).is_file())
    except Exception:
        has_chrome = False
    if not has_chrome:
        pytest.skip("PNG export requires Kaleido and Chrome.")

    output = tmp_path / "slide-figures"
    manifest = write_figures(cfg, output)
    expected = {
        "three_points.png",
        "point_dominance.png",
        "point_resourcefulness.png",
        "point_efficiency.png",
    }
    rows = manifest.loc[manifest["file"].isin(expected)]
    assert set(rows["file"]) == expected
    assert rows["chart"].eq("point_evidence_chart").all()
    assert rows["source tables"].eq("profile_comparison").all()
    assert rows["tournament"].eq("both").all()
    for filename in expected:
        image_path = output / filename
        assert image_path.is_file() and image_path.stat().st_size > 0
        with Image.open(image_path) as image:
            assert image.size == (1600, 900)
