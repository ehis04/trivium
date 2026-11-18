"""Tests for pitch zone maps, slide figures export, and efficiency ranking chart."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from components.charts import ACCENT, LIGHT_GREY, efficiency_rank
from trivium import synthetic
from trivium.charts.export import write_figures
from trivium.charts.pitch import zone_figure
from trivium.config import repo_root
from trivium.storage import write_table


def test_zone_figure_axes_and_skipping() -> None:
    """zone_figure creates one axis per usable panel and skips panels without shares."""
    usable_table_1 = pd.DataFrame(
        {
            "zone_x": [0, 1, 2],
            "zone_y": [0, 1, 2],
            "territory_share": [0.6, 0.4, 0.5],
            "team_actions": [10, 8, 5],
            "opponent_actions": [7, 12, 5],
        }
    )
    unusable_table = pd.DataFrame(
        {
            "zone_x": [0, 1],
            "zone_y": [0, 1],
            "territory_share": [np.nan, np.nan],
        }
    )
    usable_table_2 = pd.DataFrame(
        {
            "zone_x": [0, 5],
            "zone_y": [0, 2],
            "territory_share": [0.75, 0.25],
            "team_actions": [15, 5],
            "opponent_actions": [5, 15],
        }
    )

    panels = [
        ("Panel 1", usable_table_1),
        ("Panel 2 (empty shares)", unusable_table),
        ("Panel 3", usable_table_2),
    ]

    fig = zone_figure(panels, measure="territory_share")
    assert len(fig.axes) == 2
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["Panel 1", "Panel 3"]
    plt.close(fig)

    # All unusable panels returns figure with zero axes
    empty_fig = zone_figure([("No shares", unusable_table)], measure="territory_share")
    assert len(empty_fig.axes) == 0
    plt.close(empty_fig)


def test_efficiency_rank_colours_and_hover() -> None:
    """efficiency_rank colours bars by expectation and includes component metrics in tooltip."""
    table = pd.DataFrame(
        {
            "tournament": ["men", "men"],
            "team": ["Team Alpha", "Team Beta"],
            "efficiency_index": [1.45, -0.85],
            "cost_measure": ["defensive_work_per_dominance", "defensive_work_per_dominance"],
            "cost_value": [12.4, 25.1],
            "cost_z": [-0.8, 1.2],
            "conversion_value": [0.35, 0.12],
            "conversion_z": [0.9, -0.5],
        }
    )
    summary = pd.DataFrame(
        {
            "tournament": ["men", "men"],
            "team": ["Team Alpha", "Team Beta"],
            "overperformance": [1.8, -0.6],
        }
    )

    fig = efficiency_rank(table, summary=summary)
    assert len(fig.data) == 1
    trace = fig.data[0]

    # Best at the top in horizontal bar (Team Alpha sorted last on y-axis)
    assert list(trace.y) == ["Team Beta", "Team Alpha"]
    assert list(trace.x) == [-0.85, 1.45]

    # Colours: Team Alpha is above expectation (ACCENT), Team Beta is not (LIGHT_GREY)
    assert list(trace.marker.color) == [LIGHT_GREY, ACCENT]

    # Hovertemplate contains cost measure, cost value, cost z, conversion value, conversion z
    template = trace.hovertemplate
    assert "Cost measure" in template
    assert "Cost value" in template
    assert "Cost z" in template
    assert "Conversion value" in template
    assert "Conversion z" in template

    # Customdata checks for Team Alpha
    alpha_idx = list(trace.y).index("Team Alpha")
    alpha_custom = trace.customdata[alpha_idx]
    assert alpha_custom[0] == "defensive_work_per_dominance"
    assert alpha_custom[1] == 12.4
    assert alpha_custom[2] == -0.8
    assert alpha_custom[3] == 0.35
    assert alpha_custom[4] == 0.9


def test_write_figures_refuses_repo_folder(tmp_path: Path) -> None:
    """write_figures refuses an output directory located inside the git repository."""
    root = repo_root()
    if root is None:
        pytest.skip("repo_root not found")
    inside_dir = root / "temp_figures_inside_repo"
    cfg = synthetic.demo_config(tmp_path)
    with pytest.raises(ValueError, match="Output folder must be outside the repo"):
        write_figures(cfg, inside_dir)


def test_write_figures_on_small_demo(tmp_path: Path) -> None:
    """write_figures exports 1600x900 PNG slide figures and manifest from small demo build."""
    demo_root = tmp_path / "demo"
    cfg = synthetic.write_demo(demo_root, scale="small", luck_sims=50, progress_sims=50)

    # Provide sample profile_comparison so profiles figures are generated
    profiles_df = pd.DataFrame(
        [
            {
                "tournament": t,
                "measure": "territory_share",
                "label": "Field tilt",
                "raw_difference_z": 0.45,
                "raw_low": 0.1,
                "raw_high": 0.8,
                "adjusted_difference_z": 0.35,
                "adjusted_low": 0.05,
                "adjusted_high": 0.65,
                "survives_adjustment": True,
                "adjusted_excludes_zero": True,
            }
            for t in ("men", "women")
        ]
    )
    write_table(cfg, "profile_comparison", profiles_df)

    out_dir = tmp_path / "slide_figures"

    # Check Chrome availability
    try:
        import kaleido

        chrome = kaleido.get_chrome_sync()
        has_chrome = bool(chrome and Path(chrome).is_file())
    except Exception:
        has_chrome = False

    if not has_chrome:
        with pytest.raises(RuntimeError, match="Chrome binary required by Kaleido was not found"):
            write_figures(cfg, out_dir)
        return

    manifest = write_figures(cfg, out_dir)
    assert not manifest.empty
    expected_cols = ["file", "chart", "tournament", "team", "match_id", "source tables"]
    assert list(manifest.columns) == expected_cols

    manifest_file = out_dir / "figures_manifest.csv"
    assert manifest_file.is_file()

    expected_files = {
        "dumbbell_men.png",
        "dumbbell_women.png",
        "profiles_men.png",
        "profiles_women.png",
        "match_dominance.png",
        "match_xg.png",
        "match_zones.png",
    }
    assert expected_files.issubset(set(manifest["file"]))

    # Verify PNG images and their 1600 x 900 dimensions
    for fname in expected_files:
        png_path = out_dir / fname
        assert png_path.is_file(), f"Missing file: {fname}"
        assert png_path.stat().st_size > 0
        im = Image.open(png_path)
        assert im.size == (1600, 900), f"Incorrect dimensions for {fname}: {im.size}"
