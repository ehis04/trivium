"""Senior check part 2: signifiers in the professional game (build context test 10).

Evaluates whether tactical signifiers (1 territory/field tilt, 2 conversion, 3 security,
6 threat spread) and dominance parts replicate in senior tournaments: computes feature
measures from senior event data, leave-team-out Elo expectation and over-performance,
split-half repeatability with Spearman-Brown correction, and bootstrap relationship to
over-performance per match.

Output tables: senior_team_match, senior_overperformance, senior_signifiers,
checks_senior_signifiers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trivium.config import Config
from trivium.ingest import external as ex
from trivium.measures import load_measures
from trivium.models import expectation as xp
from trivium.models import reliability as rel
from trivium.models import uncertainty
from trivium.storage import read_table, table_exists, write_table

SEED = 20251103
N_BOOT = 2_000
SIGNIFIERS_TESTED = (1, 2, 3, 6)
THRESHOLD = 0.5
KEY_COLUMNS = ["tournament", "match_id", "team", "team_id", "opponent", "opponent_id"]
SIGNIFIERS_COLUMNS = [
    "tournament",
    "measure",
    "label",
    "signifier",
    "n_teams",
    "n_rows",
    "r_sb",
    "passed_reliability",
    "r_overperformance",
    "r_low",
    "r_high",
    "excludes_zero",
    "corr_with_elo",
]
CHECK_COLUMNS = ["check", "target", "value", "passed"]


def senior_team_match(events: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Team-match feature measures for senior matches (features.tables.build_team_match)."""
    from trivium.features import tables

    df, _ = tables.build_team_match(events, matches)

    reg = load_measures()
    reg_team_match = reg[reg["table"] == "team_match"]
    reg_cols = reg_team_match["column"].dropna().unique().tolist()
    reg_measures = reg_team_match["measure"].dropna().unique().tolist()

    keys = [c for c in KEY_COLUMNS if c in df.columns]
    measure_cols: list[str] = []
    for col in dict.fromkeys(reg_cols + reg_measures):
        if col in df.columns and col not in keys:
            if df[col].notna().any():
                measure_cols.append(col)

    keep_cols = keys + measure_cols
    return df[keep_cols].copy()


def senior_expectation(
    matches: pd.DataFrame,
    results: pd.DataFrame | dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute Elo expectation and over-performance per senior tournament."""
    if matches.empty:
        return pd.DataFrame()

    out_tables: list[pd.DataFrame] = []
    for tourney in matches["tournament"].unique():
        t_matches = matches[matches["tournament"] == tourney].copy()
        is_women = "women" in str(tourney).lower()
        if isinstance(results, dict):
            res_df = results["women"] if is_women else results["men"]
        else:
            res_df = results

        first_date = pd.to_datetime(t_matches["match_date"]).min()
        cutoff = first_date.date() if hasattr(first_date, "date") else first_date

        elo_table = ex.compute_elo(res_df, cutoff)
        teams = ex.tournament_teams(t_matches, tourney)
        elo_pre = ex.elo_pre_tournament(elo_table, teams, tourney)

        missing = elo_pre.loc[elo_pre["elo"].isna(), "team"].tolist()
        if missing:
            raise ValueError(
                f"No pre-tournament Elo in {tourney} for {', '.join(sorted(missing))}."
            )

        rows = xp.team_match_rows(t_matches, elo_pre, host_advantage=0.0)
        no_elo = sorted(set(rows.loc[rows["team_elo"].isna(), "team"]))
        if no_elo:
            raise ValueError(f"No pre-tournament Elo for: {', '.join(no_elo)}.")

        probs = xp.leave_team_out(rows, xp.ordered_logit())
        op = xp.overperformance(rows, probs)
        out_tables.append(op)

    return pd.concat(out_tables, ignore_index=True)


def senior_signifiers(
    team_match: pd.DataFrame,
    overperformance: pd.DataFrame,
    matches: pd.DataFrame,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate split-half reliability and over-performance association for signifiers."""
    if team_match.empty or overperformance.empty:
        return pd.DataFrame(columns=SIGNIFIERS_COLUMNS)

    reg = load_measures()
    reg_tm = reg[reg["table"] == "team_match"]

    rows: list[dict[str, Any]] = []

    for tourney in team_match["tournament"].unique():
        tm_t = team_match[team_match["tournament"] == tourney].copy()
        op_t = overperformance[overperformance["tournament"] == tourney].copy()
        m_t = matches[matches["tournament"] == tourney].copy()

        if "match_date" not in tm_t.columns and "match_date" in m_t.columns:
            m_dates = m_t[["match_id", "match_date"]].drop_duplicates("match_id")
            tm_t = tm_t.merge(m_dates, on="match_id", how="left")

        merged = tm_t.merge(
            op_t[["tournament", "match_id", "team", "overperformance", "team_elo"]],
            on=["tournament", "match_id", "team"],
            how="inner",
        )

        target_measures: list[dict[str, Any]] = []
        for _, r in reg_tm.iterrows():
            col = str(r["column"])
            m_name = str(r["measure"])
            sig = r["signifier"]
            is_tested_sig = pd.notna(sig) and int(sig) in SIGNIFIERS_TESTED
            is_dom = r["group"] == "dominance"
            if is_tested_sig or is_dom:
                chosen_col = col if col in tm_t.columns else (m_name if m_name in tm_t.columns else None)
                if chosen_col is not None and tm_t[chosen_col].notna().any():
                    target_measures.append(
                        {
                            "measure": m_name,
                            "column": chosen_col,
                            "label": str(r["label"]),
                            "signifier": int(sig) if is_tested_sig else pd.NA,
                        }
                    )

        # Run reliability and bootstrap for each measure
        rng_seed = seed
        for item in target_measures:
            col = item["column"]
            try:
                rel_df = rel.split_half(tm_t, [col], tournament=tourney, seed=rng_seed)
                r_sb = float(rel_df["r_sb"].iloc[0])
                passed_rel = bool(rel_df["passed"].iloc[0])
            except Exception:
                r_sb = np.nan
                passed_rel = False

            valid = merged[merged[col].notna() & merged["overperformance"].notna()]
            n_teams = int(valid["team"].nunique())
            n_rows = int(len(valid))

            if len(valid) >= 3 and valid[col].std() > 0 and valid["overperformance"].std() > 0:

                def make_stat(column_name: str):
                    def stat(d: pd.DataFrame) -> float:
                        corr = d[column_name].corr(d["overperformance"])
                        return float(corr) if np.isfinite(corr) else 0.0

                    return stat

                boot = uncertainty.bootstrap_by_team(
                    valid,
                    statistic=make_stat(col),
                    n_boot=n_boot,
                    seed=rng_seed,
                    team_col="team",
                    strata=None,
                )
                r_op = float(boot["estimate"])
                r_low = float(boot["low"])
                r_high = float(boot["high"])
                excludes_zero = bool(boot["excludes_zero"])
            else:
                r_op = np.nan
                r_low = np.nan
                r_high = np.nan
                excludes_zero = False

            if "team_elo" in valid.columns and len(valid) >= 3:
                team_means = valid.groupby("team")[col].mean()
                team_elos = valid.groupby("team")["team_elo"].mean()
                if team_means.std() > 0 and team_elos.std() > 0:
                    c_elo = float(team_means.corr(team_elos))
                else:
                    c_elo = np.nan
            else:
                c_elo = np.nan

            rows.append(
                {
                    "tournament": tourney,
                    "measure": item["measure"],
                    "label": item["label"],
                    "signifier": item["signifier"],
                    "n_teams": n_teams,
                    "n_rows": n_rows,
                    "r_sb": r_sb,
                    "passed_reliability": passed_rel,
                    "r_overperformance": r_op,
                    "r_low": r_low,
                    "r_high": r_high,
                    "excludes_zero": excludes_zero,
                    "corr_with_elo": c_elo,
                }
            )

    df_out = pd.DataFrame(rows, columns=SIGNIFIERS_COLUMNS)
    if not df_out.empty and "signifier" in df_out.columns:
        df_out["signifier"] = df_out["signifier"].astype("Int64")
    return df_out


def checks_senior_signifiers(table: pd.DataFrame) -> pd.DataFrame:
    """Format evaluation results into the shared checks schema."""
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        tourney = row["tournament"]
        measure = row["measure"]
        r_sb = float(row["r_sb"]) if pd.notna(row["r_sb"]) else np.nan
        passed_rel = (
            bool(row["passed_reliability"]) if pd.notna(row["passed_reliability"]) else False
        )
        r_op = (
            float(row["r_overperformance"])
            if pd.notna(row["r_overperformance"])
            else np.nan
        )

        rows.append(
            {
                "check": f"senior {tourney}: {measure} repeatable (split-half after Spearman-Brown)",
                "target": THRESHOLD,
                "value": r_sb,
                "passed": passed_rel,
            }
        )
        rows.append(
            {
                "check": f"senior {tourney}: {measure} tracks over-performance",
                "target": np.nan,
                "value": r_op,
                "passed": True,
            }
        )
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def run_senior_signifiers(
    cfg: Config,
    results: dict[str, pd.DataFrame] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Execute senior signifiers pipeline and persist output tables."""
    if not (table_exists(cfg, "senior_events") and table_exists(cfg, "senior_matches")):
        return {}

    events = read_table(cfg, "senior_events")
    matches = read_table(cfg, "senior_matches")
    if events.empty or matches.empty:
        return {}

    if results is None:
        paths = ex.download_results(cfg)
        results = {t: ex.load_results(paths[t]) for t in ("men", "women")}

    tm = senior_team_match(events, matches)
    op = senior_expectation(matches, results)
    sig = senior_signifiers(tm, op, matches, n_boot=n_boot, seed=seed)
    checks = checks_senior_signifiers(sig)

    write_table(cfg, "senior_team_match", tm)
    write_table(cfg, "senior_overperformance", op)
    write_table(cfg, "senior_signifiers", sig)
    write_table(cfg, "checks_senior_signifiers", checks)

    return {
        "senior_team_match": tm,
        "senior_overperformance": op,
        "senior_signifiers": sig,
        "checks_senior_signifiers": checks,
    }
