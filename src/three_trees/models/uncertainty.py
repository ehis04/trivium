"""Team-level uncertainty for expected points, via simulation and team bootstrap.

Each team's match outcomes are simulated independently from that team's leave-team-out
probabilities; the two rows for one match therefore use their respective fits.

Output tables: luck_bands, luck_summary, calibration, calibration_errors, checks_uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

SEED = 20251103
LEVEL = 0.95
POINTS = np.array([0.0, 1.0, 3.0])
PROB_COLUMNS = ["p_loss", "p_draw", "p_win"]
CHECK_COLUMNS = ["check", "target", "value", "passed"]


def _validate_level(level: float) -> None:
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be strictly between 0 and 1.")


def _validate_simulation_count(n_sims: int, name: str) -> None:
    if not isinstance(n_sims, (int, np.integer)) or isinstance(n_sims, bool) or n_sims < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_probabilities(table: pd.DataFrame) -> np.ndarray:
    missing = [column for column in PROB_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"Missing probability columns: {', '.join(missing)}.")
    probs = table[PROB_COLUMNS].to_numpy(dtype=float)
    valid = np.isfinite(probs).all(axis=1) & (probs >= 0).all(axis=1)
    valid &= np.isclose(probs.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    if not valid.all():
        labels = table.loc[~valid, "team"].astype(str).unique().tolist() if "team" in table else []
        detail = f" for teams {', '.join(labels)}" if labels else ""
        raise ValueError(f"Outcome probabilities must be finite, non-negative, and sum to 1{detail}.")
    return probs


def simulate_points(table: pd.DataFrame, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Draw independent 0/1/3 points for every row, returning rows by simulations."""
    _validate_simulation_count(n_sims, "n_sims")
    probs = _validate_probabilities(table)
    draws = rng.random((len(table), n_sims))
    return np.where(
        draws < probs[:, [0]],
        0.0,
        np.where(draws < probs[:, :2].sum(axis=1, keepdims=True), 1.0, 3.0),
    )


def luck_band(
    table: pd.DataFrame,
    n_sims: int = 10_000,
    level: float = LEVEL,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return each team's simulated over-performance interval and tail probabilities."""
    _validate_simulation_count(n_sims, "n_sims")
    _validate_level(level)
    required = ["tournament", "team", "expected_points", "overperformance", *PROB_COLUMNS]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    if table[["tournament", "team"]].isna().any().any():
        raise ValueError("Tournament and team labels must not be missing.")
    if not np.isfinite(table[["expected_points", "overperformance"]].to_numpy(dtype=float)).all():
        raise ValueError("Expected points and over-performance must be finite.")

    rng = np.random.default_rng(seed)
    rows = []
    for (tournament, team), group in table.groupby(["tournament", "team"], sort=False, dropna=False):
        points = simulate_points(group, n_sims, rng)
        simulated = (points - group["expected_points"].to_numpy(dtype=float)[:, None]).sum(axis=0)
        observed = float(group["overperformance"].sum())
        low, high = np.quantile(simulated, [(1 - level) / 2, 1 - (1 - level) / 2])
        rows.append(
            {
                "tournament": tournament,
                "team": team,
                "matches": len(group),
                "overperformance": observed,
                "overperformance_per_match": observed / len(group),
                "band_low": float(low),
                "band_high": float(high),
                "band_low_per_match": float(low / len(group)),
                "band_high_per_match": float(high / len(group)),
                "p_upper": float((1 + np.count_nonzero(simulated >= observed)) / (n_sims + 1)),
                "p_lower": float((1 + np.count_nonzero(simulated <= observed)) / (n_sims + 1)),
                "beyond_luck": "above" if observed > high else "below" if observed < low else "",
            }
        )
    columns = [
        "tournament", "team", "matches", "overperformance", "overperformance_per_match",
        "band_low", "band_high", "band_low_per_match", "band_high_per_match", "p_upper", "p_lower",
        "beyond_luck",
    ]
    return pd.DataFrame(rows, columns=columns)


def luck_summary(bands: pd.DataFrame, level: float = LEVEL) -> pd.DataFrame:
    """Count teams beyond their luck bands and give the expected chance count."""
    _validate_level(level)
    required = ["tournament", "beyond_luck"]
    missing = [column for column in required if column not in bands]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    rows = []
    for tournament, group in bands.groupby("tournament", sort=False, dropna=False):
        rows.append(
            {
                "tournament": tournament,
                "teams": len(group),
                "above": int(group["beyond_luck"].eq("above").sum()),
                "below": int(group["beyond_luck"].eq("below").sum()),
                "expected_by_chance": float(len(group) * (1 - level)),
            }
        )
    return pd.DataFrame(rows, columns=["tournament", "teams", "above", "below", "expected_by_chance"])


def bootstrap_by_team(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 2_000,
    level: float = LEVEL,
    seed: int = SEED,
    team_col: str = "team",
    strata: str | None = "tournament",
) -> dict[str, float]:
    """Bootstrap a statistic by resampling whole teams within each optional stratum."""
    _validate_simulation_count(n_boot, "n_boot")
    _validate_level(level)
    required = [team_col] + ([strata] if strata is not None else [])
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    if df.empty:
        raise ValueError("Cannot bootstrap an empty table.")
    if df[team_col].isna().any() or (strata is not None and df[strata].isna().any()):
        raise ValueError("Team and stratum labels must not be missing.")

    estimate = float(statistic(df.copy()))
    if not np.isfinite(estimate):
        raise ValueError("The statistic must return a finite number.")
    rng = np.random.default_rng(seed)
    if strata is None:
        strata_groups = [(None, df)]
    else:
        strata_groups = list(df.groupby(strata, sort=False, dropna=False))
    team_groups = []
    for _, stratum_df in strata_groups:
        teams = stratum_df[team_col].drop_duplicates().to_numpy()
        members = {team: stratum_df.loc[stratum_df[team_col].eq(team)] for team in teams}
        team_groups.append((teams, members))

    estimates = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        pieces = []
        for teams, members in team_groups:
            sampled = rng.choice(teams, size=len(teams), replace=True)
            pieces.extend(members[team] for team in sampled)
        sample = pd.concat(pieces, ignore_index=True)
        estimates[index] = float(statistic(sample))
    if not np.isfinite(estimates).all():
        raise ValueError("The statistic must return a finite number for every bootstrap sample.")
    low, high = np.quantile(estimates, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "n_boot": float(n_boot),
        "excludes_zero": float(low > 0 or high < 0),
    }


def checks_uncertainty(
    bands: pd.DataFrame,
    table: pd.DataFrame,
    level: float = LEVEL,
    n_sims: int = 2_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Check simulation centring and intervals, and report teams beyond luck."""
    _validate_level(level)
    _validate_simulation_count(n_sims, "n_sims")
    required = ["tournament", "team", "expected_points", *PROB_COLUMNS]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    for column in ("tournament", "team", "band_low", "band_high", "beyond_luck"):
        if column not in bands:
            raise ValueError(f"Missing columns: {column}.")
    if table[["tournament", "team"]].isna().any().any():
        raise ValueError("Tournament and team labels must not be missing.")
    if not np.isfinite(table["expected_points"].to_numpy(dtype=float)).all():
        raise ValueError("Expected points must be finite.")
    missing_tournaments = set(table["tournament"].unique()) - set(bands["tournament"].unique())
    if missing_tournaments:
        labels = ", ".join(sorted(map(str, missing_tournaments)))
        raise ValueError(f"Missing luck bands for tournaments: {labels}.")

    work = table.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    simulated = simulate_points(work, n_sims, rng) - work["expected_points"].to_numpy(dtype=float)[:, None]
    centred_by_tournament = {}
    for tournament, tournament_rows in work.groupby("tournament", sort=False, dropna=False):
        means = []
        for _, group in tournament_rows.groupby("team", sort=False, dropna=False):
            means.append(float(simulated[group.index.to_numpy()].mean()))
        centred_by_tournament[tournament] = max((abs(value) for value in means), default=0.0)
    rows = []
    summary = luck_summary(bands, level)
    for tournament in work["tournament"].drop_duplicates():
        max_mean = centred_by_tournament[tournament]
        team_bands = bands.loc[bands["tournament"].eq(tournament)]
        invalid = int((~(team_bands["band_low"] <= team_bands["band_high"])).sum())
        row_summary = summary.loc[summary["tournament"].eq(tournament)]
        rows.extend(
            [
                {
                    "check": f"{tournament} simulation is centred",
                    "target": 0.0,
                    "value": max_mean,
                    "passed": max_mean <= 0.05,
                },
                {
                    "check": f"{tournament} luck bands are ordered",
                    "target": 0.0,
                    "value": float(invalid),
                    "passed": invalid == 0,
                },
            ]
        )
        if not row_summary.empty:
            counts = row_summary.iloc[0]
            rows.append(
                {
                    "check": f"{tournament} teams beyond luck",
                    "target": float(counts["expected_by_chance"]),
                    "value": float(counts["above"] + counts["below"]),
                    "passed": True,
                }
            )
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def calibration_table(table: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Bin predicted win probabilities by tournament and compare with observed outcomes."""
    if not isinstance(n_bins, (int, np.integer)) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    required = ["tournament", "p_win", "result", "expected_points", "actual_points"]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    if table[["tournament", "p_win", "expected_points", "actual_points"]].isna().any().any():
        raise ValueError("Calibration inputs must not be missing.")
    if not np.isfinite(table[["p_win", "expected_points", "actual_points"]].to_numpy(dtype=float)).all():
        raise ValueError("Calibration inputs must be finite.")
    if not table["result"].isin([0, 1, 2]).all():
        raise ValueError("result must contain only 0, 1, or 2.")

    rows = []
    for tournament, group in table.groupby("tournament", sort=False, dropna=False):
        bin_count = min(n_bins, len(group))
        bin_ids = pd.qcut(group["p_win"].rank(method="first"), q=bin_count, labels=False) + 1
        binned = group.assign(bin=bin_ids.to_numpy())
        for bin_id, bin_rows in binned.groupby("bin", sort=True):
            rows.append(
                {
                    "tournament": tournament,
                    "bin": int(bin_id),
                    "rows": len(bin_rows),
                    "p_win_mean": float(bin_rows["p_win"].mean()),
                    "win_share": float(bin_rows["result"].eq(2).mean()),
                    "expected_points_mean": float(bin_rows["expected_points"].mean()),
                    "actual_points_mean": float(bin_rows["actual_points"].mean()),
                }
            )
    columns = [
        "tournament", "bin", "rows", "p_win_mean", "win_share",
        "expected_points_mean", "actual_points_mean",
    ]
    return pd.DataFrame(rows, columns=columns)


def calibration_error(calibration: pd.DataFrame) -> pd.DataFrame:
    """Return row-weighted win-probability and points calibration errors per tournament."""
    required = [
        "tournament", "rows", "p_win_mean", "win_share",
        "expected_points_mean", "actual_points_mean",
    ]
    missing = [column for column in required if column not in calibration]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    if calibration.empty:
        return pd.DataFrame(columns=["tournament", "win_probability_error", "points_error"])
    if calibration["rows"].isna().any() or (calibration["rows"] <= 0).any():
        raise ValueError("Calibration bin row counts must be positive.")
    numeric = calibration[
        ["p_win_mean", "win_share", "expected_points_mean", "actual_points_mean"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Calibration values must be finite.")
    rows = []
    for tournament, group in calibration.groupby("tournament", sort=False, dropna=False):
        weights = group["rows"].to_numpy(dtype=float)
        rows.append(
            {
                "tournament": tournament,
                "win_probability_error": float(
                    np.average((group["p_win_mean"] - group["win_share"]).abs(), weights=weights)
                ),
                "points_error": float(
                    np.average(
                        (group["expected_points_mean"] - group["actual_points_mean"]).abs(),
                        weights=weights,
                    )
                ),
            }
        )
    return pd.DataFrame(rows, columns=["tournament", "win_probability_error", "points_error"])


def checks_calibration(calibration: pd.DataFrame) -> pd.DataFrame:
    """Create informational check rows from the output of calibration_error."""
    required = ["tournament", "points_error"]
    missing = [column for column in required if column not in calibration]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}.")
    return pd.DataFrame(
        [
            {
                "check": f"{row.tournament} points calibration error",
                "target": np.nan,
                "value": float(row.points_error),
                "passed": True,
            }
            for row in calibration.itertuples(index=False)
        ],
        columns=CHECK_COLUMNS,
    )
