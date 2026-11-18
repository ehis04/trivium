"""Progressing vs exiting profiles, raw then adjusted (supplement addition 4), and the
profiles stage, which also writes the efficiency tables (features/efficiency.py).

- Team profile: the team's mean per match of each signifier, dominance part and spending measure
  (pressing, defensive actions, possession, running, players used), standardised within tournament
  (team level).
- Raw: the profile by stage reached, and the contrast "progressing" (won at least one
  knockout tie) minus "exiting" (the rest).
- Adjusted: the same contrast between teams above and below their Elo expectation (summed
  over-performance > 0 vs <= 0), so strength is accounted for.
- Intervals: bootstrap by team within tournament (2.5 / 97.5 percentiles).
- survives_adjustment: the adjusted interval excludes zero with the same sign as the raw
  difference. verdict, in words: "survives adjustment", "explained by strength" (raw only: it
  looks important because strong teams have it), "only after adjustment" (hidden by strength),
  "reverses" (both clear, opposite signs) or "no clear difference". corr_with_elo shows how
  much of a trait is strength.

Output tables: profile_by_stage, profile_by_overperformance, profile_comparison,
efficiency_team_match, efficiency_team_tournament, efficiency_position_group,
tactical_profile_team, tactical_profile_summary, tactical_profile_axes (models/tactical_profiles.py),
tactical_profile_sensitivity, checks_profiles; Excel profiles.xlsx; figures/profile_comparison_<tournament>.html.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from three_trees.config import Config
from three_trees.definitions import MEASURE_POINTS
from three_trees.features import efficiency
from three_trees.models import tactical_profiles
from three_trees.measures import measures
from three_trees.storage import read_table, table_exists, write_table

SEED = 20251103
N_BOOT = 2_000
LEVEL = 0.95
PROGRESSING_STAGE = 2
ABOVE, BELOW = "above expectation", "below expectation"


def profile_measures(team_match: pd.DataFrame, tournament: str) -> pd.DataFrame:
    """Signifiers, dominance parts and what was spent (definitions: resourcefulness, part "Spent") with data for
    the tournament (registry rows)."""
    reg = measures(tournament=tournament, table="team_match")
    spent = [m for m, (point, part) in MEASURE_POINTS.items() if part == "Spent"]
    reg = reg[reg["signifier"].notna() | (reg["group"] == "dominance") | reg["measure"].isin(spent)]
    rows = team_match[team_match["tournament"] == tournament]
    present = [c for c in reg["column"] if c in rows and rows[c].notna().any()]
    return reg[reg["column"].isin(present)].drop_duplicates("measure").reset_index(drop=True)


def team_profiles(team_match: pd.DataFrame, progress: pd.DataFrame, team_table: pd.DataFrame,
                  tournament: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per team with raw means, z-scores, stage and over-performance group; registry rows)."""
    reg = profile_measures(team_match, tournament)
    cols = reg["column"].tolist()
    means = team_match[team_match["tournament"] == tournament].groupby("team", as_index=False)[cols].mean()
    for col in cols:
        sd = means[col].std()
        means[f"{col}_z"] = (means[col] - means[col].mean()) / sd if sd else 0.0
    p = progress[progress["tournament"] == tournament][["team", "elo", "actual_stage", "actual_stage_name"]]
    o = team_table[team_table["tournament"] == tournament][["team", "overperformance"]]
    df = means.merge(p, on="team", how="inner").merge(o, on="team", how="inner")
    df["progress_group"] = np.where(df["actual_stage"] >= PROGRESSING_STAGE, "progressing", "exiting")
    df["expectation_group"] = np.where(df["overperformance"] > 0, ABOVE, BELOW)
    return df.reset_index(drop=True), reg


def bootstrap_difference(z: np.ndarray, in_group: np.ndarray, n_boot: int = N_BOOT, level: float = LEVEL,
                         seed: int = SEED) -> tuple[float, float, float]:
    """Mean z of in_group minus the rest, with a percentile interval from resampling teams."""
    estimate = float(z[in_group].mean() - z[~in_group].mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(z), size=(n_boot, len(z)))
    zz, gg = z[idx], in_group[idx]
    n_in, n_out = gg.sum(axis=1), (~gg).sum(axis=1)
    valid = (n_in > 0) & (n_out > 0)
    diffs = (np.where(gg, zz, 0).sum(axis=1)[valid] / n_in[valid]
             - np.where(~gg, zz, 0).sum(axis=1)[valid] / n_out[valid])
    low, high = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return estimate, float(low), float(high)


def verdict(raw_excludes_zero: bool, adjusted_excludes_zero: bool, same_sign: bool) -> str:
    if raw_excludes_zero and adjusted_excludes_zero:
        return "survives adjustment" if same_sign else "reverses"
    if raw_excludes_zero:
        return "explained by strength"
    if adjusted_excludes_zero:
        return "only after adjustment"
    return "no clear difference"


def profile_tables(team_match: pd.DataFrame, progress: pd.DataFrame, team_table: pd.DataFrame,
                   n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, pd.DataFrame]:
    by_stage, by_over, comparison = [], [], []
    for tournament in team_match["tournament"].drop_duplicates():
        df, reg = team_profiles(team_match, progress, team_table, tournament)
        if df.empty or reg.empty:
            continue
        for row in reg.itertuples(index=False):
            col, z = row.column, f"{row.column}_z"
            for (index, name), g in df.groupby(["actual_stage", "actual_stage_name"]):
                by_stage.append({"tournament": tournament, "stage_index": int(index), "stage_name": name,
                                 "n_teams": len(g), "measure": row.measure, "label": row.label,
                                 "mean": g[col].mean(), "z_mean": g[z].mean()})
            for group, g in df.groupby("expectation_group"):
                by_over.append({"tournament": tournament, "group": group, "n_teams": len(g),
                                "measure": row.measure, "label": row.label, "mean": g[col].mean(),
                                "z_mean": g[z].mean()})
            values = df[z].to_numpy(dtype=float)
            raw = bootstrap_difference(values, (df["progress_group"] == "progressing").to_numpy(), n_boot, seed=seed)
            adj = bootstrap_difference(values, (df["expectation_group"] == ABOVE).to_numpy(), n_boot, seed=seed)
            raw_excludes, adjusted_excludes = raw[1] > 0 or raw[2] < 0, adj[1] > 0 or adj[2] < 0
            same_sign = bool(np.sign(adj[0]) == np.sign(raw[0]))
            comparison.append({
                "tournament": tournament, "measure": row.measure, "label": row.label,
                "signifier": row.signifier, "higher_is_better": row.higher_is_better,
                "n_progressing": int((df["progress_group"] == "progressing").sum()),
                "n_above": int((df["expectation_group"] == ABOVE).sum()), "n_teams": len(df),
                "raw_difference_z": raw[0], "raw_low": raw[1], "raw_high": raw[2],
                "raw_excludes_zero": raw_excludes,
                "adjusted_difference_z": adj[0], "adjusted_low": adj[1], "adjusted_high": adj[2],
                "adjusted_excludes_zero": adjusted_excludes,
                "survives_adjustment": bool(adjusted_excludes and same_sign),
                "verdict": verdict(raw_excludes, adjusted_excludes, same_sign),
                "corr_with_elo": float(df[col].corr(df["elo"])),
            })
    return {
        "profile_by_stage": pd.DataFrame(by_stage),
        "profile_by_overperformance": pd.DataFrame(by_over),
        "profile_comparison": pd.DataFrame(comparison),
    }


def _check(name: str, target: float, value: float, passed: bool) -> dict[str, object]:
    return {"check": name, "target": float(target), "value": float(value), "passed": bool(passed)}


def checks_profiles(tables: dict[str, pd.DataFrame], team_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparison = tables["profile_comparison"]
    eff = tables["efficiency_team_tournament"]
    for tournament, teams in team_table.groupby("tournament", sort=False):
        c = comparison[comparison["tournament"] == tournament]
        e = eff[eff["tournament"] == tournament]
        n = len(teams)
        rows += [
            _check(f"{tournament}: profile measures compared", np.nan, len(c), len(c) > 0),
            _check(f"{tournament}: teams in profiles", n, c["n_teams"].max() if len(c) else 0,
                   len(c) > 0 and c["n_teams"].max() == n),
            _check(f"{tournament}: progressing teams (won a knockout tie)", np.nan,
                   c["n_progressing"].max() if len(c) else 0, True),
            _check(f"{tournament}: traits that survive adjustment for strength", np.nan,
                   int(c["survives_adjustment"].sum()) if len(c) else 0, True),
            _check(f"{tournament}: traits clear after accounting for strength (interval excludes zero)", np.nan,
                   int(c["adjusted_excludes_zero"].sum()) if len(c) else 0, True),
            _check(f"{tournament}: teams with an efficiency index", n, int(e["efficiency_index"].notna().sum()),
                   int(e["efficiency_index"].notna().sum()) == n),
        ]
        if "tactical_profile_team" in tables:
            tp = tables["tactical_profile_team"]
            tp = tp[tp["tournament"] == tournament]
            profiles = tables["tactical_profile_summary"]
            profiles = profiles[profiles["tournament"] == tournament]
            rows += [
                _check(f"{tournament}: teams with a tactical profile (dominant, efficient, resourceful)", n, len(tp),
                       len(tp) == n),
                _check(f"{tournament}: tactical profiles in use (of 8)", np.nan, len(profiles), True),
                _check(f"{tournament}: tactical profiles clear of zero in points above expectation", np.nan,
                       int(profiles["excludes_zero"].sum()), True),
            ]
        if "tactical_profile_sensitivity" in tables:
            sensitivity = tables["tactical_profile_sensitivity"]
            sensitivity = sensitivity[sensitivity["tournament"] == tournament]
            for score, variants in sensitivity.groupby("score", sort=True):
                keeps_sign = bool(variants["same_sign_as_main"].fillna(False).all())
                rows.append(_check(
                    f"{tournament}: sensitivity: {score} keeps its sign in every variant",
                    np.nan, float(keeps_sign), keeps_sign,
                ))
            if not sensitivity.empty:
                stability = sensitivity.iloc[0]
                rows.append(_check(
                    f"{tournament}: sensitivity: profile assignment stability "
                    f"(median {stability['median_threshold_share']:.1%}; "
                    f"leave-one-out {stability['leave_one_out_share']:.1%})",
                    np.nan, np.nan, True,
                ))
    return pd.DataFrame(rows)


def run(cfg: Config) -> None:
    from three_trees.charts import profiles as charts
    from three_trees.models.expectation import write_excel, write_figures

    team_match = read_table(cfg, "team_match")
    progress = read_table(cfg, "expected_progress")
    team_table = read_table(cfg, "overperformance_team")
    running = read_table(cfg, "running_player_match") if table_exists(cfg, "running_player_match") else None
    if running is not None:
        from three_trees.tracking.running import excluded_matches

        out = excluded_matches(running)[["tournament", "match_id"]].assign(_out=True)
        running = running.merge(out, on=["tournament", "match_id"], how="left")
        running = running[running["_out"].isna()].drop(columns="_out")

    tables = profile_tables(team_match, progress, team_table)
    tables["efficiency_team_match"] = efficiency.efficiency_team_match(team_match)
    tables["efficiency_team_tournament"] = efficiency.efficiency_team_tournament(team_match)
    tables["efficiency_position_group"] = efficiency.efficiency_position_group(running, team_match)
    summary = read_table(cfg, "team_summary") if table_exists(cfg, "team_summary") else None
    tables.update(tactical_profiles.tactical_profile_tables(
        read_table(cfg, "team_tournament"), team_table, summary, team_match, read_table(cfg, "overperformance")))
    tables["checks_profiles"] = checks_profiles(tables, team_table)
    for name, df in tables.items():
        write_table(cfg, name, df)
    write_excel(cfg.build_dir / "profiles.xlsx", tables)
    write_figures(
        cfg.build_dir / "figures",
        {f"profile_comparison_{t}": charts.profile_comparison_chart(tables["profile_comparison"], t)
         for t in tables["profile_comparison"]["tournament"].drop_duplicates()},
    )
    with pd.option_context("display.width", 180):
        cols = ["tournament", "label", "raw_difference_z", "raw_low", "raw_high", "adjusted_difference_z",
                "adjusted_low", "adjusted_high", "verdict", "corr_with_elo"]
        print(tables["profile_comparison"][cols].round(2).to_string(index=False))
        print(tables["checks_profiles"].to_string(index=False))
