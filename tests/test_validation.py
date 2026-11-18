from __future__ import annotations

import numpy as np
import pandas as pd

from trivium import validation as v


def _checks() -> dict[str, pd.DataFrame]:
    text_checks = pd.DataFrame(
        {
            "check": ["men: rows (events + shootout events)", "men: Ireland v Panama (FIFA deck target): pressures"],
            "target": ["307,480", "198 v 143"],
            "value": ["307,480", "122 v 117"],
            "passed": pd.array([True, pd.NA], dtype="boolean"),
        }
    )
    number_checks = pd.DataFrame(
        {
            "check": ["men teams beyond luck", "men host_100: Spearman with main (summed over-performance)",
                      "men: probabilities sum to 1 (largest error)", "men: ranked probability score, ordered logit vs base rates"],
            "target": [2.4, 0.8, 0.0, 0.24],
            "value": [1.0, 0.65, 0.0, 0.22],
            "passed": [True, False, True, True],
        }
    )
    return {"statsbomb": text_checks, "expectation": number_checks}


def test_collect_checks_maps_tests_and_mixed_types() -> None:
    report = v.collect_checks(_checks()).set_index("check")
    assert report.loc["men: rows (events + shootout events)", "test"] == 1
    assert report.loc["men: Ireland v Panama (FIFA deck target): pressures", "test"] == 11
    assert report.loc["men: Ireland v Panama (FIFA deck target): pressures", "status"] == "information"
    assert report.loc["men teams beyond luck", "test"] == 6
    assert report.loc["men host_100: Spearman with main (summed over-performance)", "status"] == "failed"
    assert report.loc["men host_100: Spearman with main (summed over-performance)", "test"] == 9
    assert report.loc["men: probabilities sum to 1 (largest error)", "test"] == 2
    assert report.loc["men: ranked probability score, ordered logit vs base rates", "test"] == 5
    assert report.loc["men teams beyond luck", "target"] == "2.4"


def test_summary_lists_every_test() -> None:
    by_test = v.summary(v.collect_checks(_checks())).set_index("test")
    assert list(by_test.index) == list(range(1, 12))
    assert by_test.loc[9, "status"] == "has failures"
    assert by_test.loc[3, "status"] == "not run yet"
    assert by_test.loc[11, "information"] == 1


def test_finding_gate() -> None:
    reliability = pd.DataFrame({"tournament": ["men", "men", "women"], "measure": ["a", "b", "a"],
                                "passed": [True, False, True]})
    drivers = pd.DataFrame({"tournament": ["men", "men", "women"], "measure": ["a", "b", "a"],
                            "label": ["A", "B", "A"], "coef": [0.3, 0.2, 0.1], "coef_low": [0.1, 0.05, -0.1],
                            "coef_high": [0.5, 0.4, 0.3], "excludes_zero": [True, True, False]})
    summary = pd.DataFrame({"tournament": ["men", "women"], "beats_baseline": [True, True]})
    comparison = pd.DataFrame({"tournament": ["men"], "measure": ["a"], "label": ["A"],
                               "adjusted_difference_z": [0.8], "adjusted_low": [0.2], "adjusted_high": [1.4],
                               "survives_adjustment": [True], "adjusted_excludes_zero": [True],
                               "verdict": ["survives adjustment"]})
    gate = v.finding_gate({"reliability": reliability, "drivers": drivers, "drivers_summary": summary,
                           "profile_comparison": comparison})
    key = gate.set_index(["finding_type", "tournament", "measure"])
    assert key.loc[("driver", "men", "a"), "passes_gate"]
    assert not key.loc[("driver", "men", "b"), "passes_gate"]          # not repeatable
    assert not key.loc[("driver", "women", "a"), "passes_gate"]        # interval includes zero
    assert key.loc[("driver", "men", "a"), "men_women"] == "same direction"
    assert key.loc[("profile", "men", "a"), "passes_gate"]
    assert key.loc[("profile", "men", "a"), "men_women"] == "one tournament only"
    assert "+0.30 (95% interval +0.10 to +0.50)" in key.loc[("driver", "men", "a"), "wording"]
    assert v.finding_gate({}).empty

    checks = v.men_women_checks(gate)
    assert checks["passed"].iloc[0] and checks["passed"].iloc[1:].isna().all()
    assert "same direction" in checks["value"].tolist() and "one tournament only" in checks["value"].tolist()
    report = v.collect_checks({"men_women": checks})
    assert set(report["test"]) == {7} and v.men_women_checks(v.finding_gate({})).empty


def _team(n: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({"tournament": "men", "team": [f"T{i}" for i in range(n)]}).assign(
        overperformance_per_match=rng.normal(0, 0.5, n))


def test_gate_styles_efficiency_sustainability_and_headline() -> None:
    rng = np.random.default_rng(3)
    teams = _team(40, rng)
    efficiency = teams[["tournament", "team"]].assign(
        efficiency_index=teams["overperformance_per_match"] * 2 + rng.normal(0, 0.1, 40),
        cost_measure="defensive_work_per_dominance")
    noise_eff = efficiency.assign(efficiency_index=rng.normal(0, 1, 40), tournament="women")
    styles = pd.DataFrame({"tournament": "men", "style_id": [1, 2], "style_label": ["high press", "low block"],
                           "n_teams": [20, 20], "overperformance_per_match": [0.3, -0.1], "low": [0.1, -0.4],
                           "high": [0.5, 0.2], "excludes_zero": [True, False]})
    sustainability = pd.DataFrame({"tournament": "men", "outcome": ["dominance", "dominance"],
                                   "term": ["match_number", "rest_days"], "coef": [-0.02, 0.01],
                                   "ci_low": [-0.05, -0.01], "ci_high": [0.01, 0.03]})
    luck = pd.DataFrame({"tournament": ["men", "women"], "teams": [48, 24], "above": [6, 0], "below": [2, 1],
                         "expected_by_chance": [2.4, 1.2]})
    sens = pd.DataFrame({"tournament": ["men", "men"], "spearman_total": [0.95, 0.9]})
    gate = v.finding_gate({
        "style_overperformance": styles,
        "efficiency_team_tournament": pd.concat([efficiency, noise_eff], ignore_index=True),
        "overperformance_team": pd.concat([teams, teams.assign(tournament="women")], ignore_index=True),
        "sustainability": sustainability, "luck_summary": luck, "sensitivity_comparison": sens,
    }).set_index(["finding_type", "tournament", "measure"])
    assert gate.loc[("style", "men", "style_1"), "passes_gate"]
    assert not gate.loc[("style", "men", "style_2"), "passes_gate"]
    assert "within what luck allows" in gate.loc[("style", "men", "style_2"), "wording"]
    assert gate.loc[("efficiency", "men", "efficiency_index"), "passes_gate"]
    assert not gate.loc[("efficiency", "women", "efficiency_index"), "passes_gate"]
    assert not gate.loc[("sustainability", "men", "dominance"), "passes_gate"]
    assert "survivor effect" in gate.loc[("sustainability", "men", "dominance"), "wording"]
    assert len(gate.xs("sustainability", level="finding_type")) == 1
    assert gate.loc[("headline", "men", "luck_band"), "passes_gate"]          # 8 of 48 vs 2.4 by chance
    assert not gate.loc[("headline", "women", "luck_band"), "passes_gate"]
    assert "robust to model choices" in gate.loc[("headline", "men", "luck_band"), "wording"]
    for text in gate["wording"]:
        assert not any(w in text.lower() for w in ("causes", "leads to", "because"))


def test_fixed_stages_and_new_keywords() -> None:
    assert v.test_for("senior_signifiers", "senior x: conversion repeatable (split-half after Spearman-Brown)") == 10
    assert v.test_for("contracts", "team_match: required columns present") == 1
    assert v.test_for("reports", "men: possession provider check, StatsBomb vs FIFA") == 11
    assert v.test_for("tracking", "M1 Qatar - Italy: sync median error yd, period 1") == 1
    assert v.test_for("reliability", "men: split-half reliability of conversion") == 3
    assert v.test_for("sustainability", "men: dominance survivor wording") == 8
    assert v.test_for("drivers", "men_women coefficient agreement") == 7


def test_markdown_report_lists_failures_and_gate() -> None:
    report = v.collect_checks(_checks())
    gate = pd.DataFrame({"finding_type": ["profile"], "tournament": ["men"], "label": ["Conversion"],
                         "wording": ["Conversion (men): clear of zero."], "passes_gate": [True]})
    coverage = pd.DataFrame({"tournament": ["men", "men", "women"], "label": ["Field tilt", "Security", "Field tilt"],
                             "status": ["live", "missing", "partial"]})
    text = v.markdown_report(report, v.summary(report), gate, synthetic=True, measure_coverage=coverage)
    assert "Synthetic demo build" in text and "host_100" in text and "men: Conversion (profile)" in text
    assert "## What we can say" in text and "- Conversion (men): clear of zero." in text
    assert "## What we cannot say yet" in text and "Test 9 (Sensitivity): failed checks in expectation" in text
    assert "| men | 1 | 0 | 1 |" in text and "- men, missing: Security" in text
