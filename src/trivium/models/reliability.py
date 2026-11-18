"""Split-half reliability of team means for every available registry measure.
Odd and even matches are correlated across teams and corrected with the
Spearman-Brown formula; random balanced splits give a robustness estimate.
At least two observed halves and non-constant team means are required.
Negative correlations are retained; an undefined correlation cannot pass.

Output tables: reliability, checks_reliability.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trivium import measures as registry
from trivium.config import Config
from trivium.storage import read_table, write_table

SEED = 20251103
THRESHOLD = 0.5
RANDOM_SPLITS = 200
RELIABILITY_COLUMNS = [
    "tournament", "measure", "label", "tier", "n_teams", "r_split", "r_sb",
    "r_sb_random_mean", "passed",
]
CHECK_COLUMNS = ["check", "target", "value", "passed"]


def spearman_brown(r: float) -> float:
    """Correct a split-half correlation for full length; r=-1 and NaN are undefined."""
    if pd.isna(r) or r == -1:
        return np.nan
    if not np.isfinite(r) or not -1 <= r <= 1:
        raise ValueError("A correlation must be between minus one and one.")
    return float(2 * r / (1 + r))


def _measure_specs(measure_cols: Sequence[str], tournament: str) -> list[dict[str, str]]:
    """Resolve semantic measure names or source columns, allowing literal synthetic columns."""
    known = registry.measures(table="team_match")
    specs = []
    for name in dict.fromkeys(measure_cols):
        matched = known.loc[known["measure"].eq(name)]
        if matched.empty:
            matched = known.loc[known["column"].eq(name)]
        if matched.empty:
            specs.append({"measure": name, "column": name, "label": name, "tier": "both"})
            continue
        row = matched.iloc[0]
        if tournament != "women" or row["tier"] == "both":
            specs.append({key: str(row[key]) for key in ("measure", "column", "label", "tier")})
    return specs


def _half_correlations(
    values: np.ndarray,
    team_indices: np.ndarray,
    halves: np.ndarray,
    n_teams: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the number of paired team means and their Pearson correlation per measure."""
    observed = np.isfinite(values)
    totals = np.zeros((2 * n_teams, values.shape[1]))
    counts = np.zeros_like(totals)
    groups = 2 * team_indices + halves
    np.add.at(totals, groups, np.where(observed, values, 0.0))
    np.add.at(counts, groups, observed)
    means = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    left, right = means[::2], means[1::2]
    valid = np.isfinite(left) & np.isfinite(right)
    paired = valid.sum(axis=0)
    left = np.where(valid, left, 0.0)
    right = np.where(valid, right, 0.0)
    left_mean = np.divide(left.sum(axis=0), paired, out=np.zeros(values.shape[1]), where=paired > 0)
    right_mean = np.divide(right.sum(axis=0), paired, out=np.zeros(values.shape[1]), where=paired > 0)
    dx = np.where(valid, left - left_mean, 0.0)
    dy = np.where(valid, right - right_mean, 0.0)
    denominator = np.sqrt((dx**2).sum(axis=0) * (dy**2).sum(axis=0))
    correlation = np.divide(
        (dx * dy).sum(axis=0), denominator, out=np.full(values.shape[1], np.nan),
        where=(paired >= 2) & (denominator > 0),
    )
    return paired, np.clip(correlation, -1.0, 1.0)


def split_half(
    team_match: pd.DataFrame,
    measure_cols: Sequence[str],
    tournament: str,
    random_splits: int = RANDOM_SPLITS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return reliability ordered by match date then ID; without dates use match ID order."""
    if (
        isinstance(random_splits, bool)
        or not isinstance(random_splits, (int, np.integer))
        or random_splits < 0
    ):
        raise ValueError("The number of random splits must be a non-negative integer.")
    specs = _measure_specs(measure_cols, tournament)
    required = ["tournament", "team", "match_id", *[spec["column"] for spec in specs]]
    missing = sorted(set(required) - set(team_match.columns))
    if missing:
        raise ValueError(f"Missing reliability columns: {', '.join(missing)}.")
    if not specs:
        return pd.DataFrame(columns=RELIABILITY_COLUMNS)
    work = team_match.loc[team_match["tournament"].eq(tournament)].copy()
    if work[["team", "match_id"]].isna().any().any():
        raise ValueError("Reliability team names and match IDs must not be missing.")
    if work.duplicated(["team", "match_id"]).any():
        raise ValueError("Reliability requires one row per team and match.")
    order = ["team", "match_id"]
    if "match_date" in work:
        work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")
        if work["match_date"].isna().any():
            raise ValueError("Reliability match dates must be present and valid.")
        order = ["team", "match_date", "match_id"]
    work = work.sort_values(order, ignore_index=True)
    indices, teams = pd.factorize(work["team"], sort=False)
    positions = work.groupby("team", sort=False).cumcount().to_numpy()
    values = work[[spec["column"] for spec in specs]].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, na_value=np.nan,
    )
    n_teams, r_split = _half_correlations(values, indices, positions % 2, len(teams))
    random_values = np.full((random_splits, len(specs)), np.nan)
    members = [np.flatnonzero(indices == i) for i in range(len(teams))]
    rng = np.random.default_rng(seed)
    for split_index in range(random_splits):
        halves = np.empty(len(work), dtype=int)
        for team_members in members:
            shuffled = rng.permutation(team_members)
            halves[shuffled] = np.arange(len(shuffled)) % 2
        _, correlations = _half_correlations(values, indices, halves, len(teams))
        random_values[split_index] = [spearman_brown(float(r)) for r in correlations]
    random_valid = np.isfinite(random_values)
    random_count = random_valid.sum(axis=0)
    random_mean = np.divide(
        np.where(random_valid, random_values, 0.0).sum(axis=0), random_count,
        out=np.full(len(specs), np.nan), where=random_count > 0,
    )
    rows = []
    for index, spec in enumerate(specs):
        corrected = spearman_brown(float(r_split[index]))
        rows.append({
            "tournament": tournament,
            "measure": spec["measure"],
            "label": spec["label"],
            "tier": spec["tier"],
            "n_teams": int(n_teams[index]),
            "r_split": float(r_split[index]),
            "r_sb": corrected,
            "r_sb_random_mean": float(random_mean[index]),
            "passed": bool(corrected >= THRESHOLD),
        })
    return pd.DataFrame(rows, columns=RELIABILITY_COLUMNS)


def reliability_table(
    team_match: pd.DataFrame,
    matches: pd.DataFrame,
    random_splits: int = RANDOM_SPLITS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return date-ordered reliability for all available tournament-appropriate registry measures."""
    keys = ["tournament", "match_id"]
    missing = sorted(set([*keys, "match_date"]) - set(matches.columns))
    if missing:
        raise ValueError(f"Missing match columns for reliability: {', '.join(missing)}.")
    if matches.duplicated(keys).any():
        raise ValueError("Reliability match dates require one row per tournament and match.")
    work = team_match.drop(columns="match_date", errors="ignore").merge(
        matches[[*keys, "match_date"]], on=keys, how="left", validate="many_to_one",
    )
    work["match_date"] = pd.to_datetime(work["match_date"], errors="coerce")
    if work["match_date"].isna().any():
        raise ValueError("Reliability match dates must be present and valid for every team-match row.")
    parts = []
    for tournament in sorted(work["tournament"].dropna().unique()):
        tournament_rows = work.loc[work["tournament"].eq(tournament)]
        candidates = registry.available(
            tournament_rows, registry.measures(tournament=tournament, table="team_match"),
        )
        parts.append(split_half(
            work, candidates["measure"].tolist(), tournament, random_splits=random_splits, seed=seed,
        ))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=RELIABILITY_COLUMNS)


def reliable_measures(
    reliability: pd.DataFrame,
    tournament: str,
    signifiers_only: bool = True,
) -> list[str]:
    """Return registry measure names that passed the prespecified threshold in this tournament."""
    candidates = registry.measures(
        tournament=tournament, table="team_match", signifiers_only=signifiers_only,
    )
    keep = reliability["tournament"].eq(tournament) & reliability["passed"].fillna(False).astype(bool)
    passed = reliability.loc[keep, "measure"]
    return candidates.loc[candidates["measure"].isin(passed), "measure"].tolist()


def run(cfg: Config) -> None:
    """Write measure reliability and the fixed-threshold signifier checks for each tournament."""
    team_match = read_table(cfg, "team_match")
    table = reliability_table(team_match, read_table(cfg, "matches"))
    checks = []
    for tournament in sorted(team_match["tournament"].dropna().unique()):
        available = table.loc[table["tournament"].eq(tournament)].set_index("measure")
        signifiers = registry.measures(tournament=tournament, table="team_match", signifiers_only=True)
        for row in signifiers.itertuples(index=False):
            value = float(available.loc[row.measure, "r_sb"]) if row.measure in available.index else np.nan
            checks.append({
                "check": f"{tournament}: split-half reliability of {row.label.lower()}",
                "target": THRESHOLD,
                "value": value,
                "passed": bool(value >= THRESHOLD),
            })
    write_table(cfg, "reliability", table)
    write_table(cfg, "checks_reliability", pd.DataFrame(checks, columns=CHECK_COLUMNS))
    print(
        f"Reliability: {len(table)} measure estimates; "
        f"{int(table['passed'].sum())} pass the 0.5 threshold."
    )
