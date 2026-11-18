from __future__ import annotations

import pandas as pd

from trivium.models import answer


def _comparison() -> pd.DataFrame:
    rows = [("men", "chances_share", "Chances share", 1.0, True), ("women", "chances_share", "Chances share", 0.9, True),
            ("men", "contests_share", "Contests won share", 0.3, False),
            ("men", "minutes_outside_top11_share", "Minutes outside the most-used 11", 0.1, False),
            ("men", "security", "Security", -0.9, True), ("women", "security", "Security", -0.2, False),
            ("men", "fifa_press_share", "FIFA phase: press", 0.5, True)]
    return pd.DataFrame(rows, columns=["tournament", "measure", "label", "adjusted_difference_z",
                                       "adjusted_excludes_zero"]).assign(adjusted_low=-1.0, adjusted_high=1.0)


def test_point_evidence_maps_points_and_direction() -> None:
    e = answer.point_evidence(_comparison())
    assert "fifa_press_share" not in set(e["measure"])
    assert list(e["point"].drop_duplicates()) == ["Dominance", "Resourcefulness", "Efficiency"]
    assert e.set_index(["tournament", "measure"]).loc[("men", "security"), "direction"] == "lower"


def test_point_verdicts() -> None:
    v = answer.point_verdicts(_comparison()).set_index("point")
    assert v.loc["Dominance", "verdict"] == "clear in both tournaments"
    assert v.loc["Resourcefulness", "verdict"] == "no clear difference"
    assert v.loc["Efficiency", "verdict"] == "clear in the men's tournament"
    assert v.loc["Dominance", "men_clear"] == "Chances share (higher)"
    assert "Contests won share (higher)" in v.loc["Dominance", "men_not_clear"]
