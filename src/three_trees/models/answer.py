"""The answer to the research question point by point: the profile comparison (teams above against below
their Elo expectation, after accounting for strength; models/profiles.py) grouped under dominance,
resourcefulness and efficiency (three_trees.definitions), with a verdict per point and tournament.

A measure is clear when its strength-adjusted 95% interval excludes zero (adjusted_excludes_zero).
Verdict per point: "clear in both tournaments", "clear in the men's tournament", "clear in the women's
tournament" or "no clear difference".
"""

from __future__ import annotations

import pandas as pd

from three_trees.definitions import MEASURE_POINTS, POINTS

EVIDENCE_COLUMNS = ["point", "part", "tournament", "measure", "label", "adjusted_difference_z", "adjusted_low",
                    "adjusted_high", "adjusted_excludes_zero", "direction", "verdict"]
VERDICT_COLUMNS = ["point", "verdict", "men_clear", "women_clear", "men_not_clear", "women_not_clear"]


def point_evidence(comparison: pd.DataFrame) -> pd.DataFrame:
    """profile_comparison rows of the mapped measures with point, part and direction ("higher" or "lower" for
    teams above their expectation), in point, part and label order."""
    rows = comparison[comparison["measure"].isin(MEASURE_POINTS)].copy()
    rows["point"] = rows["measure"].map(lambda m: MEASURE_POINTS[m][0])
    rows["part"] = rows["measure"].map(lambda m: MEASURE_POINTS[m][1])
    rows["direction"] = rows["adjusted_difference_z"].map(lambda v: "higher" if v > 0 else "lower")
    rows["adjusted_excludes_zero"] = rows["adjusted_excludes_zero"].fillna(False).astype(bool)
    rows["point"] = pd.Categorical(rows["point"], categories=POINTS, ordered=True)
    rows = rows.sort_values(["point", "part", "label", "tournament"], kind="stable")
    rows["point"] = rows["point"].astype(str)
    return rows.reindex(columns=EVIDENCE_COLUMNS).reset_index(drop=True)


def _labels(rows: pd.DataFrame) -> str:
    return ", ".join(f"{r.label} ({r.direction})" for r in rows.itertuples(index=False))


def point_verdicts(comparison: pd.DataFrame) -> pd.DataFrame:
    """One row per point: the verdict and, per tournament, the clear and not-clear measures in words."""
    evidence = point_evidence(comparison)
    rows = []
    for point in POINTS:
        e = evidence[evidence["point"] == point]
        clear = {t: e[(e["tournament"] == t) & e["adjusted_excludes_zero"]] for t in ("men", "women")}
        unclear = {t: e[(e["tournament"] == t) & ~e["adjusted_excludes_zero"]] for t in ("men", "women")}
        men, women = not clear["men"].empty, not clear["women"].empty
        verdict = ("clear in both tournaments" if men and women else "clear in the men's tournament" if men
                   else "clear in the women's tournament" if women else "no clear difference")
        rows.append({"point": point, "verdict": verdict,
                     "men_clear": _labels(clear["men"]), "women_clear": _labels(clear["women"]),
                     "men_not_clear": _labels(unclear["men"]), "women_not_clear": _labels(unclear["women"])})
    return pd.DataFrame(rows, columns=VERDICT_COLUMNS)
