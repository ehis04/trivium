"""Sensitivity of over-performance and team rankings to changes in the expectation model.

Evaluates whether teams' over-performance survives changes to the baseline:
host advantage (+100 Elo points), counting shootouts as wins/losses, excluding
shootouts, unconstrained Elo formula probabilities, group-stage only fitting,
and alternative youth record baselines (FIFA TDS best finish and campaign count).

These checks decide slide wording, not pipeline validity.

Returns:
    variant_rows: Filtered match rows and feature list for one variant.
    variant_overperformance: Team-match over-performance with leading variant column.
    run_variants: (team_results, scores) across all requested variants.
    knockout_teams: (tournament, team) pairs that played a knockout match.
    compare_to_main: Summary comparison of non-main variants against the main model.
    checks_sensitivity: Standard checks schema evaluating rank correlation with main.
    scores_summary: Table of ranked probability scores across variants per tournament.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.stats

from three_trees.ingest.external import expected_score
from three_trees.models.expectation import (
    DRAW,
    LOSS,
    PROB_COLUMNS,
    WIN,
    FitPredict,
    leave_team_out,
    ordered_logit,
    overperformance,
    score_predictions,
    team_match_rows,
    team_overperformance,
)

VARIANTS = (
    "main",
    "host_100",
    "shootout_as_result",
    "no_shootout_matches",
    "tds_best_finish",
    "tds_campaigns",
    "elo_plus_campaigns",
    "elo_formula",
    "group_stage_fit",
)
TDS_VARIANTS = frozenset({"tds_best_finish", "tds_campaigns", "elo_plus_campaigns"})
ROBUST_SPEARMAN = 0.8


def elo_formula_fit(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Probabilities from Elo expected score and training draw share (leave-team-out)."""
    d = float(np.mean(train["result"] == DRAW))
    gap = test["elo_gap"].to_numpy(dtype=float)
    we = np.asarray(expected_score(gap), dtype=float)
    p_draw = np.full_like(we, d)
    p_win = we - d / 2.0
    p_loss = 1.0 - we - d / 2.0
    probs = np.column_stack([p_loss, p_draw, p_win])
    probs = np.clip(probs, 0.0, None)
    total = probs.sum(axis=1, keepdims=True)
    return probs / np.where(total > 0, total, 1.0)


def group_stage_ordered_logit(features: Sequence[str] = ("elo_gap",)) -> FitPredict:
    """Ordered logit fitted on group stage rows only, then predicting all rows."""
    base_fit = ordered_logit(features)

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        group_mask = train["competition_stage"].str.contains("group", case=False, na=False)
        group_train = train[group_mask]
        effective_train = group_train if not group_train.empty else train
        return base_fit(effective_train, test)

    return fit_predict


def variant_rows(
    matches: pd.DataFrame,
    elo_pre: pd.DataFrame,
    tds_team: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Rows for the tournaments the variant applies to, and its feature list."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")

    if variant in TDS_VARIANTS:
        men_matches = matches[matches["tournament"] == "men"]
        if men_matches.empty or tds_team.empty:
            empty_main = team_match_rows(
                matches.iloc[0:0], elo_pre.iloc[0:0], host_advantage=0.0
            )
            empty_main["tds_best_finish_gap"] = pd.Series(dtype=float)
            empty_main["campaigns_gap"] = pd.Series(dtype=float)
            if variant == "tds_best_finish":
                return empty_main, ["tds_best_finish_gap"]
            if variant == "tds_campaigns":
                return empty_main, ["campaigns_gap"]
            return empty_main, ["elo_gap", "campaigns_gap"]

        men_teams = set(men_matches["home_team"]).union(men_matches["away_team"])
        valid_tds = set(
            tds_team.dropna(subset=["best_finish_rank", "campaigns_incl_2025"])["team"]
        )
        missing = sorted(men_teams - valid_tds)
        if missing:
            raise ValueError(f"Teams in TDS tournament missing TDS record: {', '.join(missing)}")

        men_elo = elo_pre[elo_pre["tournament"] == "men"]
        rows = team_match_rows(men_matches, men_elo, host_advantage=0.0)

        tds_cols = ["team", "best_finish_rank", "campaigns_incl_2025"]
        t_tds = tds_team[tds_cols].rename(
            columns={
                "best_finish_rank": "team_best_finish_rank",
                "campaigns_incl_2025": "team_campaigns",
            }
        )
        rows = rows.merge(t_tds, on="team", how="left")
        o_tds = tds_team[tds_cols].rename(
            columns={
                "team": "opponent",
                "best_finish_rank": "opponent_best_finish_rank",
                "campaigns_incl_2025": "opponent_campaigns",
            }
        )
        rows = rows.merge(o_tds, on="opponent", how="left")
        rows["tds_best_finish_gap"] = rows["opponent_best_finish_rank"] - rows["team_best_finish_rank"]
        rows["campaigns_gap"] = rows["team_campaigns"] - rows["opponent_campaigns"]
        rows = rows.drop(
            columns=[
                "team_best_finish_rank",
                "team_campaigns",
                "opponent_best_finish_rank",
                "opponent_campaigns",
            ]
        )

        if variant == "tds_best_finish":
            return rows, ["tds_best_finish_gap"]
        if variant == "tds_campaigns":
            return rows, ["campaigns_gap"]
        return rows, ["elo_gap", "campaigns_gap"]

    if variant in ("main", "elo_formula", "group_stage_fit"):
        return team_match_rows(matches, elo_pre, host_advantage=0.0), ["elo_gap"]

    if variant == "host_100":
        return team_match_rows(matches, elo_pre, host_advantage=100.0), ["elo_gap"]

    if variant == "shootout_as_result":
        rows = team_match_rows(matches, elo_pre, host_advantage=0.0)
        so = rows["shootout"].astype(bool)
        rows.loc[so, "result"] = np.where(rows.loc[so, "won_shootout"], WIN, LOSS).astype(int)
        rows.loc[so, "actual_points"] = np.where(rows.loc[so, "won_shootout"], 3.0, 0.0)
        return rows, ["elo_gap"]

    if variant == "no_shootout_matches":
        rows = team_match_rows(matches, elo_pre, host_advantage=0.0)
        rows = rows[~rows["shootout"].astype(bool)].copy().reset_index(drop=True)
        return rows, ["elo_gap"]

    raise ValueError(f"Unhandled variant: {variant}")


def variant_overperformance(
    matches: pd.DataFrame,
    elo_pre: pd.DataFrame,
    tds_team: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    """Overperformance table (per team-match) with a leading variant column."""
    rows, features = variant_rows(matches, elo_pre, tds_team, variant)
    if rows.empty:
        empty = rows.copy()
        for col in PROB_COLUMNS:
            empty[col] = pd.Series(dtype=float)
        empty["expected_points"] = pd.Series(dtype=float)
        empty["overperformance"] = pd.Series(dtype=float)
        empty.insert(0, "variant", pd.Series(dtype=str))
        return empty

    if variant == "elo_formula":
        fit_fn = elo_formula_fit
    elif variant == "group_stage_fit":
        fit_fn = group_stage_ordered_logit(features)
    else:
        fit_fn = ordered_logit(features)

    parts = []
    for _tourney, t_rows in rows.groupby("tournament", sort=False):
        t_rows = t_rows.reset_index(drop=True)
        probs = leave_team_out(t_rows, fit_fn)
        t_over = overperformance(t_rows, probs)
        parts.append(t_over)

    combined = pd.concat(parts, ignore_index=True)
    combined.insert(0, "variant", variant)
    return combined


def run_variants(
    matches: pd.DataFrame,
    elo_pre: pd.DataFrame,
    tds_team: pd.DataFrame,
    variants: Sequence[str] = VARIANTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run requested variants, returning (team_results, scores)."""
    all_results = []
    all_scores = []

    for v in variants:
        v_over = variant_overperformance(matches, elo_pre, tds_team, v)
        if v_over.empty:
            continue

        t_op = team_overperformance(v_over)
        t_op.insert(0, "variant", v)
        keep_cols = [
            "variant",
            "tournament",
            "team",
            "matches",
            "actual_points",
            "expected_points",
            "overperformance",
            "overperformance_per_match",
        ]
        all_results.append(t_op[keep_cols])

        for tourney, t_rows in v_over.groupby("tournament", sort=False):
            s = score_predictions(t_rows["result"].to_numpy(), t_rows[PROB_COLUMNS].to_numpy())
            all_scores.append({"variant": v, "tournament": tourney, **s})

    result_cols = [
        "variant",
        "tournament",
        "team",
        "matches",
        "actual_points",
        "expected_points",
        "overperformance",
        "overperformance_per_match",
    ]
    team_results = (
        pd.concat(all_results, ignore_index=True)
        if all_results
        else pd.DataFrame(columns=result_cols)
    )

    score_cols = ["variant", "tournament", "log_loss", "brier", "rps", "accuracy"]
    scores = pd.DataFrame(all_scores, columns=score_cols) if all_scores else pd.DataFrame(columns=score_cols)
    return team_results, scores


def knockout_teams(matches: pd.DataFrame) -> set[tuple[str, str]]:
    """(tournament, team) for every team that played a knockout match; team names repeat across tournaments."""
    ko = matches[~matches["competition_stage"].str.contains("group", case=False, na=False)]
    sides = pd.concat(
        [ko[["tournament", "home_team"]].set_axis(["tournament", "team"], axis=1),
         ko[["tournament", "away_team"]].set_axis(["tournament", "team"], axis=1)],
        ignore_index=True,
    )
    return set(sides.itertuples(index=False, name=None))


def compare_to_main(
    team_results: pd.DataFrame,
    knockout_teams: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Compare each non-main variant against main on teams present in both.

    knockout_teams: (tournament, team) pairs from knockout_teams(matches); without it,
    spearman_knockout_teams is NaN.
    """
    if "main" not in team_results["variant"].values:
        raise ValueError("team_results must contain a 'main' variant to compare against.")

    ko_set = set(knockout_teams) if knockout_teams is not None else None

    main_df = team_results[team_results["variant"] == "main"]
    other_variants = [v for v in team_results["variant"].unique() if v != "main"]

    rows = []
    for v in other_variants:
        v_df = team_results[team_results["variant"] == v]
        for tourney in v_df["tournament"].unique():
            m = main_df[main_df["tournament"] == tourney]
            curr = v_df[v_df["tournament"] == tourney]
            merged = m.merge(curr, on="team", suffixes=("_main", "_var"))
            if merged.empty:
                continue
            n_teams = len(merged)

            sp_total = scipy.stats.spearmanr(
                merged["overperformance_main"], merged["overperformance_var"]
            ).statistic
            sp_match = scipy.stats.spearmanr(
                merged["overperformance_per_match_main"], merged["overperformance_per_match_var"]
            ).statistic

            k = min(10, n_teams)
            top_main = set(m.sort_values("overperformance", ascending=False)["team"].iloc[:k])
            top_var = set(curr.sort_values("overperformance", ascending=False)["team"].iloc[:k])
            top10_overlap = int(len(top_main.intersection(top_var)))

            op_main = merged["overperformance_main"].to_numpy()
            op_var = merged["overperformance_var"].to_numpy()
            sign_changes = int(np.sum(((op_main > 0) & (op_var < 0)) | ((op_main < 0) & (op_var > 0))))

            diff = np.abs(op_var - op_main)
            max_idx = int(np.argmax(diff))
            max_abs_change = float(diff[max_idx])
            largest_change_team = str(merged["team"].iloc[max_idx])

            # Knockout teams Spearman correlation
            if ko_set is not None:
                ko_merged = merged[merged["team"].isin({team for t, team in ko_set if t == tourney})]
                if len(ko_merged) >= 2:
                    sp_ko = scipy.stats.spearmanr(
                        ko_merged["overperformance_main"], ko_merged["overperformance_var"]
                    ).statistic
                    spearman_ko = float(sp_ko) if not np.isnan(sp_ko) else np.nan
                else:
                    spearman_ko = np.nan
            else:
                spearman_ko = np.nan

            rows.append(
                {
                    "variant": v,
                    "tournament": tourney,
                    "n_teams": n_teams,
                    "spearman_total": float(sp_total) if not np.isnan(sp_total) else np.nan,
                    "spearman_per_match": float(sp_match) if not np.isnan(sp_match) else np.nan,
                    "top10_overlap": top10_overlap,
                    "sign_changes": sign_changes,
                    "max_abs_change": max_abs_change,
                    "largest_change_team": largest_change_team,
                    "spearman_knockout_teams": spearman_ko,
                }
            )

    cols = [
        "variant",
        "tournament",
        "n_teams",
        "spearman_total",
        "spearman_per_match",
        "top10_overlap",
        "sign_changes",
        "max_abs_change",
        "largest_change_team",
        "spearman_knockout_teams",
    ]
    return pd.DataFrame(rows, columns=cols)


def checks_sensitivity(comparison: pd.DataFrame) -> pd.DataFrame:
    """Shared checks schema evaluating rank correlation with main for each variant."""
    checks = []
    for _, row in comparison.iterrows():
        sp = row["spearman_total"]
        passed = bool(sp >= ROBUST_SPEARMAN) if not pd.isna(sp) else False
        checks.append(
            {
                "check": f"{row['tournament']} {row['variant']}: Spearman with main (summed over-performance)",
                "target": float(ROBUST_SPEARMAN),
                "value": float(sp) if not pd.isna(sp) else float("nan"),
                "passed": passed,
            }
        )
    return pd.DataFrame(checks, columns=["check", "target", "value", "passed"])


def scores_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summary of ranked probability score per tournament across variants with best_variant."""
    if scores.empty:
        return pd.DataFrame(columns=["tournament", "best_variant"])

    pivoted = scores.pivot(index="tournament", columns="variant", values="rps")
    pivoted["best_variant"] = pivoted.idxmin(axis=1)
    summary = pivoted.reset_index()
    summary.columns.name = None
    return summary

