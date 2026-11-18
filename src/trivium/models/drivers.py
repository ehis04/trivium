"""Explain points above Elo expectation with reliable, standardised signifiers.
Ridge models are evaluated on held-out teams after removing both sides of all
held-out matches. Scaling and alpha selection use training rows only. Team
bootstrap intervals describe associations, not causal effects; later-round
rows still reflect which teams survived. Men and women are fitted separately.

Output tables: drivers, drivers_summary, drivers_cv, checks_drivers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

from trivium import measures as registry
from trivium.config import Config
from trivium.models.reliability import reliable_measures
from trivium.storage import read_table, write_table

SEED = 20251103
N_BOOT = 2_000
N_FOLDS = 5
ALPHAS = np.logspace(-2, 3, 30)
IMPORTANCE_REPEATS = 10
KEYS = ["tournament", "match_id", "team"]
DRIVER_COLUMNS = [
    "tournament", "measure", "label", "coef", "coef_low", "coef_high", "excludes_zero",
    "importance", "importance_sd", "univariate_r", "univariate_low", "univariate_high",
    "corr_with_elo", "feature_mean", "feature_sd",
]
CV_COLUMNS = ["tournament", "fold", "n_test_rows", "mse_model", "mse_baseline", "alpha"]
SUMMARY_COLUMNS = [
    "tournament", "n_rows", "n_teams", "signifiers_used", "alpha", "mse_model", "mse_baseline",
    "oos_r2", "beats_baseline", "men_women_coef_corr", "n_rows_input", "n_rows_dropped",
]
CHECK_COLUMNS = ["check", "target", "value", "passed"]


def _metadata(name: str) -> tuple[str, str, str]:
    """Resolve a measure name or source column to its semantic name, column and label."""
    measures = registry.load_measures()
    found = measures.loc[measures["measure"].eq(name) | measures["column"].eq(name)]
    if found.empty:
        return name, name, name.replace("_", " ")
    row = found.iloc[0]
    return str(row["measure"]), str(row["column"]), str(row["label"])


def _names(measure_cols: Sequence[str]) -> list[str]:
    """Return unique semantic feature names in the requested order."""
    return list(dict.fromkeys(_metadata(name)[0] for name in measure_cols))


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise a plain error for missing input columns."""
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Missing driver columns: {', '.join(missing)}.")


def _one_tournament(rows: pd.DataFrame) -> str | None:
    """Return the single tournament name, rejecting mixed-tournament fits."""
    _require_columns(rows, ["tournament", "match_id", "team", "overperformance"])
    if rows["tournament"].nunique(dropna=False) > 1:
        raise ValueError("Driver fitting requires one tournament at a time.")
    if rows[["tournament", "match_id", "team"]].isna().any().any():
        raise ValueError("Driver tournament, match and team labels must not be missing.")
    return str(rows["tournament"].iloc[0]) if not rows.empty else None


def driver_rows(
    team_match: pd.DataFrame,
    overperformance: pd.DataFrame,
    measure_cols: Sequence[str],
    tournament: str,
) -> pd.DataFrame:
    """Join team-match features to points above expectation and count incomplete rows in attrs."""
    _require_columns(team_match, KEYS)
    _require_columns(overperformance, [*KEYS, "overperformance", "team_elo"])
    own = team_match.loc[team_match["tournament"].eq(tournament)].copy()
    target = overperformance.loc[overperformance["tournament"].eq(tournament)]
    if own.duplicated(KEYS).any() or target.duplicated(KEYS).any():
        raise ValueError("Driver inputs must have one row per tournament, match and team.")
    features = _names(measure_cols)
    selected = own[KEYS].copy()
    for name in measure_cols:
        semantic, source, _ = _metadata(name)
        column = source if source in own else semantic
        _require_columns(own, [column])
        selected[semantic] = pd.to_numeric(own[column], errors="coerce")
    out = selected.merge(
        target[[*KEYS, "overperformance", "team_elo"]], on=KEYS, how="left", validate="one_to_one",
    )
    numeric = [*features, "overperformance"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(out[numeric].to_numpy(dtype=float)).all(axis=1)
    out = out.loc[keep].sort_values(KEYS, ignore_index=True)
    out.attrs["n_rows_input"] = len(selected)
    out.attrs["n_rows_dropped"] = int(len(selected) - len(out))
    return out


def _clean_rows(rows: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Return finite modelling rows while preserving input counts."""
    _one_tournament(rows)
    _require_columns(rows, features)
    numeric = [*features, "overperformance"]
    values = rows[numeric].apply(pd.to_numeric, errors="coerce")
    keep = np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    clean = rows.loc[keep].copy()
    clean[numeric] = values.loc[keep]
    clean = clean.sort_values(KEYS, ignore_index=True)
    clean.attrs["n_rows_input"] = int(rows.attrs.get("n_rows_input", len(rows)))
    clean.attrs["n_rows_dropped"] = clean.attrs["n_rows_input"] - len(clean)
    return clean


def _scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardised values with the means and population standard deviations used."""
    mean = values.mean(axis=0)
    sd = values.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (values - mean) / sd, mean, sd


def _correlations(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return columnwise Pearson correlations, with NaN for a constant variable."""
    centred = values - values.mean(axis=0)
    response = target - target.mean()
    denominator = np.sqrt((centred ** 2).sum(axis=0) * (response ** 2).sum())
    return np.divide(
        centred.T @ response, denominator,
        out=np.full(values.shape[1], np.nan), where=denominator > 0,
    )


def cross_validate(
    rows: pd.DataFrame,
    measure_cols: Sequence[str],
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return team-held-out MSEs after purging mirrored matches; diagnostics are in attrs."""
    if not isinstance(n_folds, (int, np.integer)) or isinstance(n_folds, bool) or n_folds < 2:
        raise ValueError("The number of driver folds must be an integer of at least two.")
    features = _names(measure_cols)
    clean = _clean_rows(rows, features)
    n_teams = clean["team"].nunique()
    records, diagnostics, importances = [], [], []
    if n_teams >= 2:
        folds = GroupKFold(n_splits=min(n_folds, n_teams))
        for fold, (train_index, test_index) in enumerate(folds.split(clean, groups=clean["team"]), start=1):
            train = clean.iloc[train_index]
            test = clean.iloc[test_index]
            train = train.loc[~train["match_id"].isin(test["match_id"])]
            overlap = set(train["match_id"]).intersection(test["match_id"])
            diagnostics.append({
                "fold": fold,
                "train_match_ids": train["match_id"].drop_duplicates().tolist(),
                "test_match_ids": test["match_id"].drop_duplicates().tolist(),
                "train_teams": train["team"].drop_duplicates().tolist(),
                "test_teams": test["team"].drop_duplicates().tolist(),
                "overlapping_matches": len(overlap),
            })
            y_test = test["overperformance"].to_numpy(dtype=float)
            prediction = np.zeros(len(test))
            alpha = np.nan
            if features and len(train) >= 2:
                x_train, mean, sd = _scale(train[features].to_numpy(dtype=float))
                x_test = (test[features].to_numpy(dtype=float) - mean) / sd
                fitted = RidgeCV(alphas=ALPHAS).fit(x_train, train["overperformance"].to_numpy(dtype=float))
                prediction = fitted.predict(x_test)
                alpha = float(fitted.alpha_)
                importance = permutation_importance(
                    fitted, x_test, y_test, scoring="neg_mean_squared_error",
                    n_repeats=IMPORTANCE_REPEATS, random_state=seed + fold,
                )
                importances.append(importance.importances)
            records.append({
                "tournament": str(clean["tournament"].iloc[0]),
                "fold": fold,
                "n_test_rows": len(test),
                "mse_model": float(np.mean((prediction - y_test) ** 2)),
                "mse_baseline": float(np.mean(y_test ** 2)),
                "alpha": alpha,
            })
    out = pd.DataFrame(records, columns=CV_COLUMNS)
    out.attrs["folds"] = diagnostics
    out.attrs["leakage_violations"] = sum(item["overlapping_matches"] for item in diagnostics)
    out.attrs["importance"] = np.concatenate(importances, axis=1).tolist() if importances else []
    return out


def _summary(rows: pd.DataFrame, features: Sequence[str], cv: pd.DataFrame) -> dict[str, float]:
    """Summarise held-out improvement over the zero-points Elo baseline."""
    if cv.empty:
        baseline = float(np.mean(rows["overperformance"].to_numpy(dtype=float) ** 2)) if len(rows) else np.nan
        model = baseline
        denominator = baseline
        numerator = baseline
    else:
        model, baseline = float(cv["mse_model"].mean()), float(cv["mse_baseline"].mean())
        numerator = float((cv["mse_model"] * cv["n_test_rows"]).sum())
        denominator = float((cv["mse_baseline"] * cv["n_test_rows"]).sum())
    return {
        "n_rows": len(rows),
        "n_teams": rows["team"].nunique(),
        "signifiers_used": len(features),
        "alpha": np.nan,
        "mse_model": model,
        "mse_baseline": baseline,
        "oos_r2": 1.0 - numerator / denominator if not cv.empty and denominator > 0 else np.nan,
        "beats_baseline": bool(np.isfinite(model) and model < baseline),
        "men_women_coef_corr": np.nan,
        "n_rows_input": int(rows.attrs.get("n_rows_input", len(rows))),
        "n_rows_dropped": int(rows.attrs.get("n_rows_dropped", 0)),
    }


def fit_drivers(
    rows: pd.DataFrame,
    measure_cols: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Return standardised ridge associations and team-bootstrap intervals, plus held-out summary."""
    if not isinstance(n_boot, (int, np.integer)) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError("The number of driver bootstrap samples must be a positive integer.")
    features = _names(measure_cols)
    clean = _clean_rows(rows, features)
    cv = cross_validate(clean, features, seed=seed)
    summary = _summary(clean, features, cv)
    if not features or len(clean) < 2 or clean["team"].nunique() < 2:
        return pd.DataFrame(columns=DRIVER_COLUMNS), summary
    x, mean, sd = _scale(clean[features].to_numpy(dtype=float))
    y = clean["overperformance"].to_numpy(dtype=float)
    fitted = RidgeCV(alphas=ALPHAS).fit(x, y)
    alpha = float(fitted.alpha_)
    summary["alpha"] = alpha
    teams = clean["team"].drop_duplicates().to_numpy()
    members = [np.flatnonzero(clean["team"].eq(team).to_numpy()) for team in teams]
    rng = np.random.default_rng(seed)
    coefficients = np.empty((n_boot, len(features)))
    correlations = np.empty_like(coefficients)
    for index in range(n_boot):
        sampled = rng.integers(0, len(teams), size=len(teams))
        sample = np.concatenate([members[team] for team in sampled])
        coefficients[index] = Ridge(alpha=alpha).fit(x[sample], y[sample]).coef_
        correlations[index] = _correlations(x[sample], y[sample])
    coef_low, coef_high = np.quantile(coefficients, [0.025, 0.975], axis=0)
    univariate = _correlations(x, y)
    uni_low, uni_high = np.full(len(features), np.nan), np.full(len(features), np.nan)
    for index in range(len(features)):
        finite = correlations[:, index][np.isfinite(correlations[:, index])]
        if len(finite):
            uni_low[index], uni_high[index] = np.quantile(finite, [0.025, 0.975])
    importance = np.asarray(cv.attrs["importance"], dtype=float)
    importance_mean = importance.mean(axis=1) if importance.size else np.full(len(features), np.nan)
    importance_sd = importance.std(axis=1) if importance.size else np.full(len(features), np.nan)
    elo = pd.to_numeric(clean.get("team_elo", pd.Series(np.nan, index=clean.index)), errors="coerce")
    elo_correlations = np.full(len(features), np.nan)
    finite_elo = np.isfinite(elo.to_numpy(dtype=float))
    if finite_elo.sum() >= 2:
        elo_correlations = _correlations(x[finite_elo], elo.to_numpy(dtype=float)[finite_elo])
    records = []
    for index, name in enumerate(features):
        records.append({
            "tournament": str(clean["tournament"].iloc[0]),
            "measure": name,
            "label": _metadata(name)[2],
            "coef": float(fitted.coef_[index]),
            "coef_low": float(coef_low[index]),
            "coef_high": float(coef_high[index]),
            "excludes_zero": bool(coef_low[index] > 0 or coef_high[index] < 0),
            "importance": float(importance_mean[index]),
            "importance_sd": float(importance_sd[index]),
            "univariate_r": float(univariate[index]),
            "univariate_low": float(uni_low[index]),
            "univariate_high": float(uni_high[index]),
            "corr_with_elo": float(elo_correlations[index]),
            "feature_mean": float(mean[index]),
            "feature_sd": float(sd[index]),
        })
    return pd.DataFrame(records, columns=DRIVER_COLUMNS), summary


def run(cfg: Config) -> None:
    """Write per-tournament driver estimates, held-out scores and checks for reliable signifiers."""
    team_match = read_table(cfg, "team_match")
    performance = read_table(cfg, "overperformance")
    reliability = read_table(cfg, "reliability")
    driver_parts, cv_parts, summaries, checks = [], [], [], []
    for tournament in sorted(team_match["tournament"].dropna().unique()):
        available = registry.available(
            team_match.loc[team_match["tournament"].eq(tournament)],
            registry.measures(tournament=tournament, table="team_match", signifiers_only=True),
        )
        passed = set(reliable_measures(reliability, tournament))
        features = [name for name in available["measure"] if name in passed]
        rows = driver_rows(team_match, performance, features, tournament)
        drivers, summary = fit_drivers(rows, features)
        cv = cross_validate(rows, features)
        driver_parts.append(drivers)
        cv_parts.append(cv)
        summaries.append({"tournament": tournament, **summary})
        checks.extend([
            _check(f"{tournament}: signifiers improve held-out predictions", 1,
                   float(summary["beats_baseline"]), bool(summary["beats_baseline"])),
            _check(f"{tournament}: driver intervals excluding zero (count)", np.nan,
                   float(drivers["excludes_zero"].sum()), True),
            _check(f"{tournament}: held-out matches present in training", 0,
                   float(cv.attrs["leakage_violations"]), cv.attrs["leakage_violations"] == 0),
            _check(f"{tournament}: incomplete team-match rows dropped", 0,
                   float(summary["n_rows_dropped"]), True),
        ])
        print(f"{tournament}: {len(features)} reliable signifiers, {len(rows)} team-matches; "
              f"held-out improvement: {bool(summary['beats_baseline'])}.")
    drivers = pd.concat(driver_parts, ignore_index=True) if driver_parts else pd.DataFrame(
        columns=DRIVER_COLUMNS,
    )
    cv = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame(columns=CV_COLUMNS)
    summary_table = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    if not drivers.empty and set(drivers["tournament"]) >= {"men", "women"}:
        shared = drivers.pivot(index="measure", columns="tournament", values="coef")[["men", "women"]]
        shared = shared.dropna()
        if len(shared) >= 3 and shared["men"].nunique() > 1 and shared["women"].nunique() > 1:
            summary_table["men_women_coef_corr"] = float(shared["men"].corr(shared["women"]))
    write_table(cfg, "drivers", drivers)
    write_table(cfg, "drivers_summary", summary_table)
    write_table(cfg, "drivers_cv", cv)
    write_table(cfg, "checks_drivers", pd.DataFrame(checks, columns=CHECK_COLUMNS))


def _check(name: str, target: float, value: float, passed: bool) -> dict[str, object]:
    """Return a row of the shared checks schema."""
    return {"check": name, "target": float(target), "value": float(value), "passed": bool(passed)}
