"""Tactical profiles (the answer to the tournament pathways question): how dominant, how efficient and how
resourceful each team was, and which profiles did better than the team's strength predicts.

- Scores per team (team-tournament means of team_match, z within tournament, oriented with
  the registry's higher_is_better so that higher is always the better side of the definition):
  dominant = dominance; efficient = mean of conversion and security; resourceful = mean of the
  costs of dominance (defensive work per point of dominance, and for the men high-speed running
  without the ball per point of dominance). A score uses the parts the tournament has.
- Profile: each score at or above the tournament average is "high" ("Dominant", "Efficient",
  "Resourceful"), below is "low" ("Less dominant", "Less efficient", "Costly"); eight profiles.
- Outcome: points above Elo expectation per match (overperformance_team), and stages beyond the
  expected stage (team_summary.progress_over_expectation) where built.
- Per profile: teams, mean points above expectation per match with a bootstrap interval by team,
  share of teams above expectation, mean stages beyond expectation.
- Per score: the Spearman correlation with points above expectation per match (bootstrap interval
  by team; ranks of the full sample), the joint least-squares effect of the three scores on points above expectation per
  match (per standard deviation, bootstrap interval by team) and the Spearman correlation;
  repeatability of the match-level score (split-half, Spearman-Brown, models.reliability); and out
  of sample, the score from the first half of each team's matches (by date) against its points
  above expectation per match in the rest (Spearman, bootstrap interval by team).

Output tables: tactical_profile_team, tactical_profile_summary, tactical_profile_axes,
tactical_profile_sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from three_trees.measures import measure

SEED = 20251103
N_BOOT = 2_000
LEVEL = 0.95
AXES = {
    "dominant": ("dominance",),
    "efficient": ("conversion", "security"),
    "resourceful": ("defensive_work_per_dominance", "hsr_without_ball_per_dominance"),
}
AXIS_LABELS = {"dominant": ("Dominant", "Less dominant"), "efficient": ("Efficient", "Less efficient"),
               "resourceful": ("Resourceful", "Costly")}
TEAM_COLUMNS = ["tournament", "team", "matches", "dominant", "efficient", "resourceful", "profile",
                "overperformance_per_match", "progress_over_expectation", "actual_stage_name", "elo"]
SUMMARY_COLUMNS = ["tournament", "profile", "dominant", "efficient", "resourceful", "n_teams", "teams",
                   "overperformance_per_match", "low", "high", "excludes_zero", "share_above_expectation",
                   "progress_over_expectation"]
AXES_COLUMNS = ["tournament", "axis", "label", "parts", "n_teams", "effect", "low", "high", "excludes_zero",
                "spearman", "spearman_low", "spearman_high", "spearman_excludes_zero", "r_sb", "repeatable", "early_late_spearman", "early_late_low", "early_late_high",
                "predicts_later"]


def _z(values: pd.Series) -> pd.Series:
    sd = values.std()
    return (values - values.mean()) / sd if sd and np.isfinite(sd) else values * 0.0


def _oriented_parts(rows: pd.DataFrame, tournament: str, names: tuple[str, ...]) -> list[pd.Series]:
    parts = []
    for name in names:
        m = measure(name)
        column = str(m["column"])
        if m["tier"] == "men" and tournament != "men" or column not in rows or rows[column].isna().all():
            continue
        sign = 1.0 if str(m["higher_is_better"]).lower() == "true" else -1.0
        parts.append(sign * _z(rows[column].astype(float)))
    return parts


def match_scores(team_match: pd.DataFrame) -> pd.DataFrame:
    """tournament, match_id, team and score_<axis> per team-match (oriented z of the parts within
    tournament, averaged)."""
    out = []
    for tournament, rows in team_match.groupby("tournament", sort=True):
        scores = rows[["tournament", "match_id", "team"]].copy()
        for axis, names in AXES.items():
            parts = _oriented_parts(rows, tournament, names)
            scores[f"score_{axis}"] = pd.concat(parts, axis=1).mean(axis=1) if parts else np.nan
        out.append(scores)
    return pd.concat(out, ignore_index=True)


def axis_scores(team_tournament: pd.DataFrame) -> pd.DataFrame:
    """tournament, team, matches and the three scores (z within tournament, higher = better)."""
    out = []
    for tournament, rows in team_tournament.groupby("tournament", sort=True):
        scores = rows[["tournament", "team", "matches"]].copy()
        for axis, names in AXES.items():
            parts = _oriented_parts(rows, tournament, names)
            scores[axis] = _z(pd.concat(parts, axis=1).mean(axis=1)) if parts else np.nan
        out.append(scores)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["tournament", "team", "matches",
                                                                             *AXES])


def profile_name(row: pd.Series) -> str:
    return " · ".join(AXIS_LABELS[a][0] if row[a] >= 0 else AXIS_LABELS[a][1] for a in AXES)


def team_profiles(team_tournament: pd.DataFrame, team_table: pd.DataFrame,
                  summary: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team: scores, profile and outcomes."""
    scores = axis_scores(team_tournament).dropna(subset=list(AXES))
    scores["profile"] = scores.apply(profile_name, axis=1)
    out = scores.merge(team_table[["tournament", "team", "overperformance_per_match"]],
                       on=["tournament", "team"], how="left", validate="one_to_one")
    extra = ["progress_over_expectation", "actual_stage_name", "elo"]
    if summary is not None:
        out = out.merge(summary[["tournament", "team", *[c for c in extra if c in summary]]],
                        on=["tournament", "team"], how="left", validate="one_to_one")
    out = out.reindex(columns=TEAM_COLUMNS)
    return out.sort_values(["tournament", "overperformance_per_match"], ascending=[True, False], ignore_index=True)


def _interval(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [(1 - LEVEL) / 2, (1 + LEVEL) / 2]))


def profile_summary(teams: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Per tournament and profile: teams and how they did against their Elo expectation."""
    rng = np.random.default_rng(seed)
    rows = []
    for (tournament, profile), group in teams.groupby(["tournament", "profile"], sort=True):
        values = group["overperformance_per_match"].dropna().to_numpy(float)
        low, high = _interval(values, rng, n_boot)
        first = group.iloc[0]
        rows.append({
            "tournament": tournament, "profile": profile,
            **{a: bool(first[a] >= 0) for a in AXES},
            "n_teams": len(group), "teams": ", ".join(sorted(group["team"])),
            "overperformance_per_match": float(values.mean()) if len(values) else np.nan,
            "low": low, "high": high, "excludes_zero": bool(low > 0 or high < 0) if np.isfinite(low) else False,
            "share_above_expectation": float((values > 0).mean()) if len(values) else np.nan,
            "progress_over_expectation": float(group["progress_over_expectation"].mean()),
        })
    out = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return out.sort_values(["tournament", "overperformance_per_match"], ascending=[True, False], ignore_index=True)


def _effects(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0][1:]


def axis_effects(teams: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """Per tournament and score: joint effect on points above expectation per match per standard
    deviation, with a bootstrap interval by team, and the Spearman correlation."""
    rng = np.random.default_rng(seed)
    rows = []
    for tournament, group in teams.groupby("tournament", sort=True):
        data = group.dropna(subset=[*AXES, "overperformance_per_match"])
        x, y = data[list(AXES)].to_numpy(float), data["overperformance_per_match"].to_numpy(float)
        if len(data) <= len(AXES) + 1:
            continue
        effect = _effects(x, y)
        idx = rng.integers(0, len(data), size=(n_boot, len(data)))
        boot = np.array([_effects(x[i], y[i]) for i in idx])
        low, high = np.quantile(boot, [(1 - LEVEL) / 2, (1 + LEVEL) / 2], axis=0)
        ranks = pd.DataFrame(np.column_stack([x, y])).rank().to_numpy()
        rho_boot = np.array([np.corrcoef(ranks[i].T)[-1, :-1] for i in idx])
        rho_low, rho_high = np.nanquantile(rho_boot, [(1 - LEVEL) / 2, (1 + LEVEL) / 2], axis=0)
        for k, axis in enumerate(AXES):
            parts = [n for n in AXES[axis] if not (measure(n)["tier"] == "men" and tournament != "men")]
            rows.append({
                "tournament": tournament, "axis": axis, "label": AXIS_LABELS[axis][0],
                "parts": ", ".join(str(measure(n)["label"]) for n in parts), "n_teams": len(data),
                "effect": float(effect[k]), "low": float(low[k]), "high": float(high[k]),
                "excludes_zero": bool(low[k] > 0 or high[k] < 0),
                "spearman": float(data[axis].corr(data["overperformance_per_match"], method="spearman")),
                "spearman_low": float(rho_low[k]), "spearman_high": float(rho_high[k]),
                "spearman_excludes_zero": bool(rho_low[k] > 0 or rho_high[k] < 0),
            })
    return pd.DataFrame(rows, columns=AXES_COLUMNS)


def axis_reliability(scores: pd.DataFrame) -> pd.DataFrame:
    """tournament, axis, r_sb, repeatable: split-half reliability of each match-level score."""
    from three_trees.models import reliability as rel

    rows = []
    for tournament in sorted(scores["tournament"].unique()):
        table = rel.split_half(scores, [f"score_{a}" for a in AXES], tournament)
        for r in table.itertuples(index=False):
            rows.append({"tournament": tournament, "axis": r.measure.removeprefix("score_"), "r_sb": r.r_sb,
                         "repeatable": bool(r.passed)})
    return pd.DataFrame(rows, columns=["tournament", "axis", "r_sb", "repeatable"])


def early_late(scores: pd.DataFrame, overperformance: pd.DataFrame, n_boot: int = N_BOOT,
               seed: int = SEED) -> pd.DataFrame:
    """tournament, axis and the Spearman correlation (bootstrap interval by team) between the score
    over the first half of each team's matches and its points above expectation per match in the rest."""
    rng = np.random.default_rng(seed)
    keys = ["tournament", "match_id", "team"]
    data = scores.merge(overperformance[[*keys, "match_date", "overperformance"]], on=keys, how="inner")
    data = data.sort_values(["tournament", "team", "match_date", "match_id"], ignore_index=True)
    n = data.groupby(["tournament", "team"])["match_id"].transform("size")
    data["early"] = data.groupby(["tournament", "team"]).cumcount() < np.ceil(n / 2)
    rows = []
    for tournament, group in data.groupby("tournament", sort=True):
        early = group[group["early"]].groupby("team")[[f"score_{a}" for a in AXES]].mean()
        late = group[~group["early"]].groupby("team")["overperformance"].mean().rename("late")
        both = early.join(late, how="inner").dropna()
        if len(both) < 5:
            continue
        idx = rng.integers(0, len(both), size=(n_boot, len(both)))
        for axis in AXES:
            x, y = both[f"score_{axis}"].to_numpy(float), both["late"].to_numpy(float)
            rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            boot = [pd.Series(x[i]).corr(pd.Series(y[i]), method="spearman") for i in idx]
            low, high = np.nanquantile(boot, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
            rows.append({"tournament": tournament, "axis": axis, "early_late_spearman": rho,
                         "early_late_low": float(low), "early_late_high": float(high),
                         "predicts_later": bool(low > 0)})
    return pd.DataFrame(rows, columns=["tournament", "axis", "early_late_spearman", "early_late_low",
                                       "early_late_high", "predicts_later"])


def tactical_profile_tables(team_tournament: pd.DataFrame, team_table: pd.DataFrame,
                            summary: pd.DataFrame | None = None, team_match: pd.DataFrame | None = None,
                            overperformance: pd.DataFrame | None = None,
                            n_boot: int = N_BOOT) -> dict[str, pd.DataFrame]:
    teams = team_profiles(team_tournament, team_table, summary)
    axes = axis_effects(teams, n_boot)
    if team_match is not None:
        scores = match_scores(team_match)
        axes = axes.drop(columns=["r_sb", "repeatable"]).merge(axis_reliability(scores), on=["tournament", "axis"],
                                                               how="left")
        if overperformance is not None:
            axes = axes.drop(columns=["early_late_spearman", "early_late_low", "early_late_high", "predicts_later"]
                             ).merge(early_late(scores, overperformance, n_boot), on=["tournament", "axis"],
                                     how="left")
    tables = {
        "tactical_profile_team": teams,
        "tactical_profile_summary": profile_summary(teams, n_boot),
        "tactical_profile_axes": axes.reindex(columns=AXES_COLUMNS),
    }
    sensitivity_columns = ["tournament", "score", "variant", "n_teams", "spearman", "low", "high",
                           "excludes_zero", "same_sign_as_main"]
    if team_match is not None and overperformance is not None and summary is not None:
        sensitivity_teams = team_table
        if "elo" not in sensitivity_teams:
            sensitivity_teams = sensitivity_teams.merge(
                summary[["tournament", "team", "elo"]], on=["tournament", "team"], how="left",
                validate="one_to_one")
        sensitivity = profile_sensitivity(team_tournament, sensitivity_teams, team_match, overperformance)
        stability = profile_stability(team_tournament)
        sensitivity = sensitivity.merge(stability, on="tournament", how="left", validate="many_to_one")
        tables["tactical_profile_sensitivity"] = sensitivity
    else:
        tables["tactical_profile_sensitivity"] = pd.DataFrame(columns=[
            *sensitivity_columns, "median_threshold_share", "leave_one_out_share",
        ])
    return tables


def _correlation_interval(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Spearman correlation and its 95% team-bootstrap interval."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan
    estimate = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    indices = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    bx, by = rankdata(x[indices], axis=1), rankdata(y[indices], axis=1)
    bx -= bx.mean(axis=1, keepdims=True)
    by -= by.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", bx, by)
    denominator = np.sqrt(np.einsum("ij,ij->i", bx, bx) * np.einsum("ij,ij->i", by, by))
    valid = numerator[denominator > 0] / denominator[denominator > 0]
    if not len(valid):
        return estimate, np.nan, np.nan
    low, high = np.quantile(valid, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
    return estimate, float(low), float(high)


def profile_sensitivity(team_tournament: pd.DataFrame, team_table: pd.DataFrame, team_match: pd.DataFrame,
                        overperformance: pd.DataFrame) -> pd.DataFrame:
    """Per tournament and score: rank correlation and team-bootstrap interval for each definition variant."""
    variant_parts = {
        "dominant": ("contests_share", "territory_share", "chances_share"),
        "efficient": AXES["efficient"],
        "resourceful": AXES["resourceful"],
    }
    required_parts = {name for names in (*AXES.values(), *variant_parts.values()) for name in names}
    required = {
        "team_tournament": (team_tournament, ["tournament", "team", "matches", *required_parts]),
        "team_table": (team_table, ["tournament", "team", "overperformance_per_match", "elo"]),
        "team_match": (team_match, ["tournament", "match_id", "team"]),
        "overperformance": (overperformance, ["tournament", "match_id", "team", "competition_stage"]),
    }
    for name, (table, columns) in required.items():
        missing = [column for column in columns if column not in table]
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}.")
    if team_table[["overperformance_per_match", "elo"]].isna().any().any():
        raise ValueError("team_table must have over-performance and pre-tournament Elo for every team.")

    rows = []
    rng = np.random.default_rng(SEED)
    for tournament, outcomes in team_table.groupby("tournament", sort=True):
        teams = team_tournament[team_tournament["tournament"] == tournament].copy()
        main_scores = axis_scores(teams).set_index("team")
        score_variants: dict[str, dict[str, pd.Series]] = {
            axis: {"main": main_scores[axis]} for axis in AXES
        }

        for axis, names in variant_parts.items():
            for name in names:
                if measure(name)["tier"] == "men" and tournament != "men":
                    continue
                part = _oriented_parts(teams, tournament, (name,))
                if part:
                    score_variants[axis][name] = part[0].set_axis(teams["team"].to_numpy())
                else:
                    score_variants[axis][name] = pd.Series(dtype=float, name=name)

        stages = overperformance.loc[
            overperformance["tournament"] == tournament, ["match_id", "team", "competition_stage"]
        ]
        group_matches = team_match.loc[team_match["tournament"] == tournament].merge(
            stages, on=["match_id", "team"], how="inner", validate="one_to_one"
        )
        group_matches = group_matches[
            group_matches["competition_stage"].astype(str).str.contains("group", case=False, na=False)
        ]
        if group_matches.empty:
            raise ValueError(f"{tournament} has no group-stage matches for tactical sensitivity.")
        group_columns = [name for names in AXES.values() for name in names if name in group_matches]
        grouped = group_matches.groupby("team", as_index=False)[group_columns].mean()
        grouped.insert(0, "tournament", tournament)
        for axis, names in AXES.items():
            parts = _oriented_parts(grouped, tournament, names)
            score_variants[axis]["group_stage_only"] = (
                _z(pd.concat(parts, axis=1).mean(axis=1)).set_axis(grouped["team"].to_numpy())
                if parts else pd.Series(dtype=float, name="group_stage_only")
            )

        n_exclude = max(1, int(np.ceil(len(outcomes) * 0.25)))
        strongest = set(outcomes.nlargest(n_exclude, "elo")["team"])
        retained = set(outcomes["team"]) - strongest
        for axis in AXES:
            score_variants[axis]["without_strongest"] = main_scores.loc[
                main_scores.index.isin(retained), axis
            ]

        for axis, variants in score_variants.items():
            correlations: dict[str, tuple[float, float, float, int]] = {}
            for variant, scores in variants.items():
                data = outcomes[["team", "overperformance_per_match"]].merge(
                    scores.rename("score_value"), left_on="team", right_index=True, how="inner",
                    validate="one_to_one",
                ).dropna(subset=["score_value", "overperformance_per_match"])
                x = data["score_value"].to_numpy(float)
                y = data["overperformance_per_match"].to_numpy(float)
                estimate, low, high = _correlation_interval(x, y, rng)
                correlations[variant] = estimate, low, high, len(data)
            main_sign = np.sign(correlations["main"][0])
            for variant, (estimate, low, high, n_teams) in correlations.items():
                rows.append({
                    "tournament": tournament, "score": axis, "variant": variant, "n_teams": n_teams,
                    "spearman": estimate, "low": low, "high": high,
                    "excludes_zero": bool(low > 0 or high < 0)
                    if np.isfinite(low) and np.isfinite(high) else False,
                    "same_sign_as_main": bool(np.isfinite(estimate) and np.isfinite(main_sign)
                                              and np.sign(estimate) == main_sign),
                })
    return pd.DataFrame(rows, columns=["tournament", "score", "variant", "n_teams", "spearman", "low", "high",
                                       "excludes_zero", "same_sign_as_main"])


def profile_stability(team_tournament: pd.DataFrame) -> pd.DataFrame:
    """Per tournament: share of team profiles unchanged at median thresholds and across leave-one-out fits."""
    rows = []
    for tournament, teams in team_tournament.groupby("tournament", sort=True):
        scores = axis_scores(teams)
        axes = list(AXES)
        baseline = scores.set_index("team")
        baseline_profiles = baseline.apply(profile_name, axis=1)
        median_profiles = (baseline[axes].ge(baseline[axes].median()).replace({False: -1.0, True: 1.0})
                           .apply(profile_name, axis=1))
        median_share = float(median_profiles.eq(baseline_profiles).mean()) if len(baseline) else np.nan

        stable = 0
        comparisons = 0
        if len(teams) > 1:
            for omitted in teams["team"]:
                reduced = teams[teams["team"] != omitted]
                loo = axis_scores(reduced).set_index("team")
                common = baseline_profiles.index.intersection(loo.index)
                stable += int(baseline_profiles.loc[common].eq(loo.apply(profile_name, axis=1).loc[common]).sum())
                comparisons += len(common)
        rows.append({
            "tournament": tournament, "n_teams": len(teams), "median_threshold_share": median_share,
            "leave_one_out_share": stable / comparisons if comparisons else np.nan,
        })
    return pd.DataFrame(rows, columns=["tournament", "median_threshold_share", "leave_one_out_share"])
