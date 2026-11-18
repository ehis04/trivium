"""Validation: collects every checks_<stage> table in the build, maps each check to the
tests fixed in advance (build context section 10) and writes the check report, plus the
slide gate: a finding goes on the slides only if it passes tests 3-6.

Tests: 1 data checks, 2 baseline, 3 repeatability, 4 not just strength, 5 out-of-sample,
6 uncertainty, 7 men and women separately, 8 opponent and survivor effects, 9 sensitivity,
10 professional game, 11 provider check. A check maps to its stage's default test unless its
wording names another (for example luck bands -> 6, variants vs main -> 9); contract and
senior checks always keep their stage's test.

Checks tables come in two shapes (numeric target/value, or text with passed empty for
information rows); both are reported as text, and an empty passed is "information".

Slide gate rows: drivers, profile traits, styles, the efficiency index, sustainability
(match number) and the headline over-performance, each with its numbers and one plain
sentence that carries the hedge the tests require.

Output tables: check_report, finding_gate (and, through contracts.run, checks_contracts,
measure_coverage, build_status); check_report.xlsx (summary, one tab per stage, the gate,
coverage); check_report.md (plain-English summary).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from three_trees import contracts, db
from three_trees.config import Config
from three_trees.measures import load_measures, measures
from three_trees.storage import read_table, table_exists, write_table

TESTS = {
    1: "Data checks", 2: "Baseline", 3: "Repeatability", 4: "Not just strength", 5: "Out-of-sample",
    6: "Uncertainty", 7: "Men and women separately", 8: "Opponent and survivor effects", 9: "Sensitivity",
    10: "Professional game", 11: "Provider check",
}
STAGE_TESTS = {
    "statsbomb": 1, "features": 1, "match_folders": 1, "tracking": 1, "reports": 1, "contracts": 1,
    "external": 1, "expectation": 2, "reliability": 3, "drivers": 5, "styles": 6, "profiles": 4,
    "sustainability": 8, "senior": 10, "senior_signifiers": 10, "men_women": 7,
}
FIXED_STAGES = frozenset({"contracts", "senior", "senior_signifiers", "men_women"})
KEYWORD_TESTS = (
    (("fifa deck", "provider", "fifa vs statsbomb", "statsbomb vs fifa"), 11),
    (("spearman with main", "sensitivity:"), 9),
    (("men and women", "men_women", "both tournaments"), 7),
    (("survivor", "match number"), 8),
    (("luck", "beyond", "interval", "bootstrap", "excludes zero"), 6),
    (("beats baseline", "base rates", "out-of-sample"), 5),
    (("split-half", "repeatable", "spearman-brown"), 3),
    (("higher-rated", "brier", "binomial"), 2),
    (("survive adjustment", "corr_with_elo"), 4),
    (("sync", "within 2%", "matched"), 1),
)
REPORT_COLUMNS = ["stage", "test", "test_name", "check", "target", "value", "passed", "status"]
GATE_TESTS = ["test3_repeatable", "test4_not_just_strength", "test5_out_of_sample", "test6_interval_excludes_zero"]
GATE_COLUMNS = ["finding_type", "tournament", "measure", "label", "estimate", "low", "high", *GATE_TESTS,
                "men_women", "note", "wording"]
N_BOOT = 2_000
SEED = 20251103
ROBUST_SPEARMAN = 0.8


def _text(value: object) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def test_for(stage: str, check: str) -> int:
    if stage in FIXED_STAGES:
        return STAGE_TESTS[stage]
    text = check.lower()
    for words, test in KEYWORD_TESTS:
        if any(w in text for w in words):
            return test
    return STAGE_TESTS.get(stage, 1)


def collect_checks(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One report row per check from {stage: checks table}."""
    rows = []
    for stage, df in tables.items():
        for r in df.itertuples(index=False):
            passed = r.passed
            passed = None if passed is None or passed is pd.NA or (isinstance(passed, float) and np.isnan(passed)) \
                else bool(passed)
            test = test_for(stage, str(r.check))
            rows.append({
                "stage": stage, "test": test, "test_name": TESTS[test], "check": str(r.check),
                "target": _text(r.target), "value": _text(r.value), "passed": passed,
                "status": "information" if passed is None else "passed" if passed else "failed",
            })
    report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    report["passed"] = report["passed"].astype("boolean")
    return report.sort_values(["test", "stage"], kind="stable", ignore_index=True)


def summary(report: pd.DataFrame) -> pd.DataFrame:
    counts = report.pivot_table(index=["test", "test_name"], columns="status", values="check", aggfunc="count",
                                fill_value=0)
    counts = counts.reindex(columns=["passed", "failed", "information"], fill_value=0).reset_index()
    full = pd.DataFrame({"test": list(TESTS), "test_name": list(TESTS.values())})
    out = full.merge(counts, on=["test", "test_name"], how="left").fillna(0)
    out[["passed", "failed", "information"]] = out[["passed", "failed", "information"]].astype(int)
    out["status"] = np.where(out[["passed", "failed", "information"]].sum(axis=1) == 0, "not run yet",
                             np.where(out["failed"] > 0, "has failures", "all passed"))
    return out


def coverage(cfg: Config) -> pd.DataFrame:
    """Registry coverage per measure and tournament (contracts.measure_coverage)."""
    return contracts.measure_coverage(cfg)


# ---------------------------------------------------------------- slide gate


def _repeatable(reliability: pd.DataFrame | None, tournament: str, names: list[str]) -> object:
    """True if every named measure with a reliability row passed test 3; NA when none has a row."""
    if reliability is None or reliability.empty:
        return pd.NA
    rows = reliability[(reliability["tournament"] == tournament) & reliability["measure"].isin(names)]
    return pd.NA if rows.empty else bool(rows["passed"].astype(bool).all())


def _interval(estimate: float, low: float, high: float) -> str:
    return f"{estimate:+.2f} (95% interval {low:+.2f} to {high:+.2f})"


def _clear(row: dict[str, object]) -> str:
    return "clear of zero" if row["test6_interval_excludes_zero"] is True else "within what luck allows"


def _both(men_women: str) -> str:
    return {"agree": "; seen in both tournaments", "differ": "; the men's and women's results differ",
            "same direction": "; same direction in both tournaments, clear in one",
            "one tournament only": "; seen in one tournament only"}.get(men_women, "")


def wording(row: dict[str, object]) -> str:
    """One plain sentence from the row's numbers, with the hedge its tests require."""
    t, label, kind = row["tournament"], row["label"], row["finding_type"]
    est, low, high = (float(row[k]) if pd.notna(row[k]) else np.nan for k in ("estimate", "low", "high"))
    if kind == "driver":
        return (f"{label} ({t}): {_interval(est, low, high)} points per match per standard deviation, beyond the "
                f"Elo expectation; {_clear(row)}{_both(row['men_women'])}; an association, not cause.")
    if kind == "profile":
        return (f"{label} ({t}): teams above their Elo expectation differ from teams below by "
                f"{_interval(est, low, high)} standard deviations ({row['note']}); {_clear(row)}"
                f"{_both(row['men_women'])}.")
    if kind == "style":
        return (f"{label} ({t}): {_interval(est, low, high)} points per match above expectation; {_clear(row)}; "
                "an association, not cause.")
    if kind == "tactical profile":
        later = {True: "; measured in a team's first matches it predicts the rest",
                 False: "; measured in a team's first matches it does not clearly predict the rest"}.get(
            row["test5_out_of_sample"] if row["test5_out_of_sample"] is not pd.NA else None, "")
        return (f"{label} ({t}; {row['note']}): rank correlation with points above Elo expectation per match "
                f"{_interval(est, low, high)}; {_clear(row)}{later}{_both(row['men_women'])}.")
    if kind == "efficiency":
        return (f"Efficiency index ({t}): correlation with points above expectation per match "
                f"{_interval(est, low, high)}; {_clear(row)}; an association, not cause.")
    if kind == "sustainability":
        return (f"{label} ({t}): {_interval(est, low, high)} per standard deviation of match number; "
                f"{_clear(row)}; later matches only include teams still in the tournament, so part of any change "
                "can be a survivor effect.")
    if kind == "headline":
        return (f"{t}: {int(est)} teams finished beyond their luck band against {low:.1f} expected by chance; "
                + ("more than chance alone would give" if row["test6_interval_excludes_zero"] is True
                   else "single-tournament totals are mostly within what luck allows")
                + f"{row['note']}.")
    return f"{label} ({t})."


def _men_women(table: pd.DataFrame, key: str, sign_col: str, clear_col: str) -> dict[str, str]:
    """Per measure: agree (same sign, same clear / not clear), same direction (same sign, clear in one),
    differ (opposite signs), or one tournament only."""
    out = {}
    for name, rows in table.groupby(key):
        by_t = rows.set_index("tournament")
        if not {"men", "women"} <= set(by_t.index):
            out[name] = "one tournament only"
            continue
        same_sign = np.sign(by_t.loc["men", sign_col]) == np.sign(by_t.loc["women", sign_col])
        same_clear = bool(by_t.loc["men", clear_col]) == bool(by_t.loc["women", clear_col])
        out[name] = ("agree" if same_clear else "same direction") if same_sign else "differ"
    return out


def _gate_row(kind: str, tournament: str, measure: str, label: str, estimate: float, low: float, high: float,
              t3: object, t4: object, t5: object, t6: object, men_women: str, note: str) -> dict[str, object]:
    row = {"finding_type": kind, "tournament": tournament, "measure": measure, "label": label, "estimate": estimate,
           "low": low, "high": high, "test3_repeatable": t3, "test4_not_just_strength": t4,
           "test5_out_of_sample": t5, "test6_interval_excludes_zero": t6, "men_women": men_women, "note": note}
    row["wording"] = wording(row)
    return row


def finding_gate(tables: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Tests 3-6 and a plain sentence per candidate finding."""
    reliability = tables.get("reliability")
    rows = []

    drivers, drv_summary = tables.get("drivers"), tables.get("drivers_summary")
    if drivers is not None and not drivers.empty:
        beats = drv_summary.set_index("tournament")["beats_baseline"] if drv_summary is not None else pd.Series()
        agree = _men_women(drivers.assign(clear=drivers["excludes_zero"]), "measure", "coef", "clear")
        for r in drivers.itertuples(index=False):
            rows.append(_gate_row("driver", r.tournament, r.measure, r.label, r.coef, r.coef_low, r.coef_high,
                                  _repeatable(reliability, r.tournament, [r.measure]), True,
                                  bool(beats.get(r.tournament, False)), bool(r.excludes_zero), agree[r.measure],
                                  "target is over-performance against Elo expectation"))

    comparison = tables.get("profile_comparison")
    if comparison is not None and not comparison.empty:
        agree = _men_women(comparison, "measure", "adjusted_difference_z", "adjusted_excludes_zero")
        for r in comparison.itertuples(index=False):
            rows.append(_gate_row("profile", r.tournament, r.measure, r.label, r.adjusted_difference_z,
                                  r.adjusted_low, r.adjusted_high, _repeatable(reliability, r.tournament, [r.measure]),
                                  True, pd.NA, bool(r.adjusted_excludes_zero), agree[r.measure],
                                  getattr(r, "verdict", "")))

    styles = tables.get("style_overperformance")
    if styles is not None and not styles.empty:
        for r in styles.itertuples(index=False):
            style_measures = measures(group="style", tournament=r.tournament)["measure"].tolist() + ["territory_share"]
            rows.append(_gate_row("style", r.tournament, f"style_{r.style_id}", r.style_label,
                                  r.overperformance_per_match, r.low, r.high,
                                  _repeatable(reliability, r.tournament, style_measures), True, pd.NA,
                                  bool(r.excludes_zero), "", f"{r.n_teams} teams"))

    axes = tables.get("tactical_profile_axes")
    if axes is not None and not axes.empty:
        agree = _men_women(axes, "axis", "spearman", "spearman_excludes_zero")
        for r in axes.itertuples(index=False):
            repeatable = pd.NA if pd.isna(r.repeatable) else bool(r.repeatable)
            later = pd.NA if pd.isna(r.predicts_later) else bool(r.predicts_later)
            note = (f"{r.parts}; joint effect {_interval(r.effect, r.low, r.high)} points per match per standard "
                    "deviation")
            rows.append(_gate_row("tactical profile", r.tournament, f"profile_{r.axis}", f"{r.label} score",
                                  r.spearman, r.spearman_low, r.spearman_high, repeatable, True, later,
                                  bool(r.spearman_excludes_zero), agree[r.axis], note))

    efficiency, team_table = tables.get("efficiency_team_tournament"), tables.get("overperformance_team")
    if efficiency is not None and not efficiency.empty and team_table is not None and not team_table.empty:
        from three_trees.models.uncertainty import bootstrap_by_team

        for tournament, e in efficiency.groupby("tournament", sort=False):
            merged = e[["tournament", "team", "efficiency_index", "cost_measure"]].merge(
                team_table[["tournament", "team", "overperformance_per_match"]], on=["tournament", "team"]
            ).dropna(subset=["efficiency_index", "overperformance_per_match"])
            if len(merged) < 4:
                continue
            boot = bootstrap_by_team(
                merged, lambda d: float(d["efficiency_index"].corr(d["overperformance_per_match"])),
                n_boot=N_BOOT, seed=SEED, strata=None)
            inputs = [str(merged["cost_measure"].iloc[0]), "np_xg", "dominance"]
            rows.append(_gate_row("efficiency", tournament, "efficiency_index", "Efficiency index", boot["estimate"],
                                  boot["low"], boot["high"], _repeatable(reliability, tournament, inputs), True,
                                  pd.NA, bool(boot["excludes_zero"]), "", "team-level correlation"))

    sustainability = tables.get("sustainability")
    if sustainability is not None and not sustainability.empty:
        labels = load_measures().set_index("measure")["label"]
        for r in sustainability[sustainability["term"] == "match_number"].itertuples(index=False):
            rows.append(_gate_row("sustainability", r.tournament, r.outcome, str(labels.get(r.outcome, r.outcome)),
                                  r.coef, r.ci_low, r.ci_high, pd.NA, pd.NA, pd.NA,
                                  bool(r.ci_low > 0 or r.ci_high < 0), "", "match number term"))

    luck = tables.get("luck_summary")
    if luck is not None and not luck.empty:
        comparison_s = tables.get("sensitivity_comparison")
        for r in luck.itertuples(index=False):
            beyond = int(r.above + r.below)
            # Each team is beyond its band with probability expected_by_chance / teams under the model.
            p_value = binomtest(beyond, int(r.teams), r.expected_by_chance / r.teams, "greater").pvalue \
                if r.teams else 1.0
            robust = ""
            if comparison_s is not None and not comparison_s.empty:
                s = comparison_s[comparison_s["tournament"] == r.tournament]["spearman_total"]
                if len(s):
                    robust = ("; the team ranking is robust to model choices" if bool((s >= ROBUST_SPEARMAN).all())
                              else "; the team ranking changes under some model choices")
            rows.append(_gate_row("headline", r.tournament, "luck_band", "Points above expectation", beyond,
                                  r.expected_by_chance, np.nan, pd.NA, True, pd.NA, bool(p_value < 0.05), "", robust))

    gate = pd.DataFrame(rows, columns=GATE_COLUMNS)
    if gate.empty:
        return gate.assign(passes_gate=pd.Series(dtype=bool))
    gate[GATE_TESTS] = gate[GATE_TESTS].astype("boolean")
    # A missing test (not run yet, or not applicable) does not block; a failed one does.
    gate["passes_gate"] = ~gate[GATE_TESTS].eq(False).any(axis=1) & gate["test6_interval_excludes_zero"].fillna(False)
    return gate


# ---------------------------------------------------------------- report


def men_women_checks(gate: pd.DataFrame) -> pd.DataFrame:
    """Test 7: every finding judged in each tournament separately, and how the men's and women's results
    compare for the findings that pass the slide gate (information)."""
    rows: list[dict[str, object]] = []
    if gate.empty:
        return pd.DataFrame(rows, columns=["check", "target", "value", "passed"])
    tournaments = sorted(gate["tournament"].unique())
    rows.append({"check": "men and women: findings judged in each tournament separately", "target": "men, women",
                 "value": ", ".join(tournaments), "passed": tournaments == ["men", "women"]})
    passing = gate[gate["passes_gate"]]
    for r in passing.itertuples(index=False):
        rows.append({"check": f"men and women: {r.label} ({r.finding_type}, passes in {r.tournament})",
                     "target": "agree / same direction / differ / one tournament only", "value": r.men_women,
                     "passed": None})
    counts = passing["men_women"].value_counts()
    rows.append({"check": "men and women: findings passing the slide gate, by agreement", "target": "listed",
                 "value": ", ".join(f"{k} {v}" for k, v in counts.items()) or "none", "passed": None})
    return pd.DataFrame(rows, columns=["check", "target", "value", "passed"])


def markdown_report(report: pd.DataFrame, by_test: pd.DataFrame, gate: pd.DataFrame, synthetic: bool,
                    measure_coverage: pd.DataFrame | None = None) -> str:
    lines = ["# Check report", ""]
    if synthetic:
        lines += ["**Synthetic demo build: fake teams and random values, not real results.**", ""]
    lines += ["| Test | Passed | Failed | Information | Status |", "|---|---|---|---|---|"]
    lines += [f"| {r.test}. {r.test_name} | {r.passed} | {r.failed} | {r.information} | {r.status} |"
              for r in by_test.itertuples(index=False)]

    passing = gate[gate["passes_gate"]] if not gate.empty else gate
    lines += ["", "## What we can say", ""]
    lines += [f"- {r.wording}" for r in passing.itertuples(index=False)] or ["- nothing yet passes tests 3-6"]

    lines += ["", "## What we cannot say yet", ""]
    blocked = by_test[by_test["status"] != "all passed"]
    for r in blocked.itertuples(index=False):
        stages = sorted(report.loc[(report["test"] == r.test) & (report["status"] == "failed"), "stage"].unique())
        reason = f"failed checks in {', '.join(stages)}" if stages else "not run yet"
        lines.append(f"- Test {r.test} ({r.test_name}): {reason}")
    if blocked.empty:
        lines.append("- every test has passed checks only")

    if measure_coverage is not None and not measure_coverage.empty:
        lines += ["", "## Measure coverage", "", "| Tournament | Live | Partial | Missing |", "|---|---|---|---|"]
        for t, rows in measure_coverage.groupby("tournament", sort=False):
            counts = rows["status"].value_counts()
            lines.append(f"| {t} | {counts.get('live', 0)} | {counts.get('partial', 0)} | {counts.get('missing', 0)} |")
        for t, rows in measure_coverage.groupby("tournament", sort=False):
            for status in ("partial", "missing"):
                names = rows.loc[rows["status"] == status, "label"].tolist()
                if names:
                    lines.append(f"- {t}, {status}: {', '.join(names)}")

    failed = report[report["status"] == "failed"]
    lines += ["", f"## Failed checks ({len(failed)})", ""]
    lines += [f"- [{r.stage}] {r.check}: value {r.value}, target {r.target}" for r in failed.itertuples(index=False)]
    lines += ["", "## Findings passing tests 3-6 (slide gate)", ""]
    lines += [f"- {r.tournament}: {r.label} ({r.finding_type})" for r in passing.itertuples(index=False)] or ["- none"]
    return "\n".join(lines) + "\n"


def run(cfg: Config) -> None:
    from three_trees.models.expectation import write_excel

    contracts.run(cfg)
    names = [n for n in db.tables(cfg) if n.startswith("checks_")]
    checks = {n.removeprefix("checks_"): read_table(cfg, n) for n in names}
    sources = {n: read_table(cfg, n) if table_exists(cfg, n) else None
               for n in ("reliability", "drivers", "drivers_summary", "profile_comparison", "style_overperformance",
                         "efficiency_team_tournament", "overperformance_team", "sustainability", "luck_summary",
                         "sensitivity_comparison", "tactical_profile_axes")}
    gate = finding_gate(sources)
    report = collect_checks({**checks, "men_women": men_women_checks(gate)})
    by_test = summary(report)
    measure_coverage = read_table(cfg, "measure_coverage")
    synthetic = table_exists(cfg, "build_info") and read_table(cfg, "build_info")["source"].eq("synthetic").all()

    write_table(cfg, "check_report", report)
    write_table(cfg, "finding_gate", gate)
    sheets = {"summary": by_test, **{s: report[report["stage"] == s] for s in checks}, "finding_gate": gate,
              "coverage": measure_coverage}
    write_excel(cfg.build_dir / "check_report.xlsx", sheets)
    (cfg.build_dir / "check_report.md").write_text(
        markdown_report(report, by_test, gate, synthetic, measure_coverage), encoding="utf-8")
    with pd.option_context("display.width", 160):
        print(by_test.to_string(index=False))
        failed = report[report["status"] == "failed"]
        if not failed.empty:
            print(failed[["stage", "check", "target", "value"]].to_string(index=False))
