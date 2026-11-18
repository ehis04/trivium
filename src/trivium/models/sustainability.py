"""Changes in dominance, defensive work and running over a tournament.
REML mixed models use standardised match number, accumulated starting-team
minutes, rest and opponent Elo, with a random intercept for each team.
Later matches contain only teams still in the tournament: this survivor
effect remains, while the random intercept separates team level from
within-team change. Associations do not establish a causal load effect.
Constant terms are omitted; missing or unidentifiable models are skipped.

Output tables: sustainability, checks_sustainability.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from trivium import measures as registry
from trivium.config import Config
from trivium.storage import read_table, write_table

OUTCOMES = ("dominance", "defensive_work_per_dominance", "hsr_without_ball_km")
PREDICTORS = ("match_number", "starters_minutes_cum", "rest_days", "opponent_elo")
FIT_METHODS = ("lbfgs", "bfgs", "cg")
KEYS = ["tournament", "match_id", "team"]
SUSTAINABILITY_COLUMNS = [
    "tournament", "outcome", "term", "coef", "ci_low", "ci_high", "p_value", "n_rows", "n_teams",
    "converged",
]
CHECK_COLUMNS = ["check", "target", "value", "passed"]


def _require_keys(table: pd.DataFrame) -> None:
    """Require a single row for each named tournament, match and team."""
    missing = sorted(set(KEYS) - set(table.columns))
    if missing:
        raise ValueError(f"Missing sustainability keys: {', '.join(missing)}.")
    if table[KEYS].isna().any().any() or table.duplicated(KEYS).any():
        raise ValueError("Sustainability requires one row per named tournament, match and team.")


def _first_match_rest(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill missing rest only for first matches using the tournament's observed median."""
    rows = rows.copy()
    if "match_number" in rows:
        rows["first_match"] = rows["match_number"].eq(1).fillna(False).astype(int)
    if "rest_days" in rows and "first_match" in rows:
        rest = pd.to_numeric(rows["rest_days"], errors="coerce").astype(float)
        rest = rest.replace([np.inf, -np.inf], np.nan)
        median = rest.median() if rest.notna().any() else np.nan
        rows["rest_days"] = rest.mask(rest.isna() & rows["first_match"].eq(1), median)
    return rows


def sustainability_rows(
    team_match: pd.DataFrame,
    overperformance: pd.DataFrame,
    tournament: str,
) -> pd.DataFrame:
    """Join canonical registry outcomes and predictors to opponent Elo for one tournament."""
    _require_keys(team_match)
    _require_keys(overperformance)
    own = team_match.loc[team_match["tournament"].eq(tournament)]
    target = overperformance.loc[overperformance["tournament"].eq(tournament)]
    metadata = registry.measures(tournament=tournament, table="team_match").set_index("measure")
    selected = own[KEYS].copy()
    for name in (*OUTCOMES, *PREDICTORS[:-1]):
        if name in metadata.index:
            source = str(metadata.loc[name, "column"])
            if source in own:
                selected[name] = pd.to_numeric(own[source], errors="coerce").astype(float)
    extra = [name for name in ("opponent_elo", "match_date") if name in target]
    rows = selected.merge(target[[*KEYS, *extra]], on=KEYS, how="left", validate="one_to_one")
    if "opponent_elo" in rows:
        rows["opponent_elo"] = pd.to_numeric(rows["opponent_elo"], errors="coerce").astype(float)
    return _first_match_rest(rows).sort_values(KEYS, ignore_index=True)


def _empty_fit(reason: str, n_rows_input: int = 0) -> pd.DataFrame:
    """Return the result schema and a reason why a model could not be estimated."""
    table = pd.DataFrame(columns=SUSTAINABILITY_COLUMNS)
    table.attrs.update({"reason": reason, "n_rows_input": n_rows_input, "warnings": []})
    return table


def fit_sustainability(rows: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Return fixed-effect estimates per predictor SD, capturing fitting warnings and failures."""
    required_keys = ["tournament", "team"]
    if any(name not in rows for name in required_keys):
        raise ValueError("Sustainability fits require tournament and team columns.")
    if rows[required_keys].isna().any().any() or rows["tournament"].nunique() > 1:
        raise ValueError("Fit sustainability for one tournament with non-missing team names.")
    missing = [name for name in (outcome, *PREDICTORS) if name not in rows]
    if missing:
        return _empty_fit(f"Missing outcome or predictors: {', '.join(missing)}.", len(rows))
    if rows.empty:
        return _empty_fit("No team-match rows are available.")
    work = rows[["tournament", "team", outcome, *PREDICTORS]].copy()
    numeric = [outcome, *PREDICTORS]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce").astype(float)
    work = _first_match_rest(work)
    finite = np.isfinite(work[[*numeric, "first_match"]].to_numpy(dtype=float)).all(axis=1)
    work = work.loc[finite].copy()
    if work["team"].nunique() < 2 or work.groupby("team").size().max() < 2:
        return _empty_fit("At least two teams and repeated matches are needed.", len(rows))
    if work[outcome].nunique() < 2:
        return _empty_fit(f"Outcome {outcome} has no variation.", len(rows))
    active, omitted = [], []
    for name in PREDICTORS:
        scale = float(work[name].std(ddof=0))
        if scale > 0:
            work[name] = (work[name] - work[name].mean()) / scale
            active.append(name)
        else:
            omitted.append(name)
    if work["first_match"].nunique() > 1:
        active.append("first_match")
    else:
        omitted.append("first_match")
    if not active or len(work) <= len(active) + 1:
        return _empty_fit("Too few rows or no varying predictors for a mixed model.", len(rows))
    design = np.column_stack([np.ones(len(work)), work[active].to_numpy(dtype=float)])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return _empty_fit("The fixed-effect predictors are linearly dependent.", len(rows))
    formula = f"{outcome} ~ {' + '.join(active)}"
    fitted = None
    captured = []
    failures = []
    for method in FIT_METHODS:
        candidate = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                candidate = smf.mixedlm(formula, work, groups=work["team"]).fit(
                    reml=True, method=method, disp=False, maxiter=500,
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError) as error:
                failures.append(str(error))
        captured.extend(str(item.message) for item in caught)
        if candidate is not None and np.isfinite(candidate.fe_params.to_numpy()).all():
            fitted = candidate
            if bool(candidate.converged):
                break
    if fitted is None:
        table = _empty_fit("Mixed model could not be estimated: " + "; ".join(failures), len(rows))
        table.attrs["warnings"] = captured
        return table
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        intervals = fitted.conf_int()
        p_values = fitted.pvalues
    captured.extend(str(item.message) for item in caught)
    records = []
    for term, coef in fitted.fe_params.items():
        records.append({
            "tournament": str(work["tournament"].iloc[0]),
            "outcome": outcome,
            "term": str(term),
            "coef": float(coef),
            "ci_low": float(intervals.loc[term].iloc[0]),
            "ci_high": float(intervals.loc[term].iloc[1]),
            "p_value": float(p_values.loc[term]),
            "n_rows": len(work),
            "n_teams": work["team"].nunique(),
            "converged": bool(fitted.converged),
        })
    table = pd.DataFrame(records, columns=SUSTAINABILITY_COLUMNS)
    table.attrs.update({
        "n_rows_input": len(rows), "warnings": captured, "omitted_terms": omitted,
        "reason": "" if bool(fitted.converged) else "The mixed model did not converge.",
    })
    return table


def run(cfg: Config) -> None:
    """Write tournament-specific mixed models and checks, listing skipped inputs and fits."""
    team_match = read_table(cfg, "team_match")
    performance = read_table(cfg, "overperformance")
    metadata = registry.load_measures().set_index("measure")
    parts, checks = [], []
    for tournament in sorted(team_match["tournament"].dropna().unique()):
        rows = sustainability_rows(team_match, performance, tournament)
        missing_predictors = []
        for predictor in PREDICTORS:
            available = predictor in rows and np.isfinite(rows[predictor].to_numpy(dtype=float)).any()
            checks.append(_check(f"{tournament}: sustainability predictor available: {predictor}",
                                 1, float(available), bool(available)))
            if not available:
                missing_predictors.append(predictor)
        fitted_outcomes = 0
        for outcome in OUTCOMES:
            label = str(metadata.loc[outcome, "label"]) if outcome in metadata.index else outcome
            men_only = outcome in metadata.index and metadata.loc[outcome, "tier"] == "men"
            if tournament == "women" and men_only:
                checks.append(_check(f"{tournament}: {label.lower()} skipped (men-only measure)",
                                     np.nan, 0, True))
                continue
            available = outcome in rows and np.isfinite(rows[outcome].to_numpy(dtype=float)).any()
            if not available:
                checks.append(_check(f"{tournament}: {label.lower()} unavailable; skipped", 1, 0, False))
                continue
            if missing_predictors:
                missing_names = ", ".join(missing_predictors)
                checks.append(_check(
                    f"{tournament}: {label.lower()} skipped; missing predictors: {missing_names}",
                    1, 0, False,
                ))
                continue
            fitted = fit_sustainability(rows, outcome)
            if fitted.empty:
                checks.append(_check(f"{tournament}: {label.lower()} skipped: {fitted.attrs['reason']}",
                                     1, 0, False))
                continue
            fitted_outcomes += 1
            parts.append(fitted)
            converged = bool(fitted["converged"].all())
            finite_intervals = bool(np.isfinite(fitted[["ci_low", "ci_high"]].to_numpy()).all())
            checks.extend([
                _check(f"{tournament}: {label.lower()} mixed model converged",
                       1, float(converged), converged),
                _check(f"{tournament}: {label.lower()} fixed-effect intervals are finite",
                       1, float(finite_intervals), finite_intervals),
                _check(f"{tournament}: {label.lower()} incomplete rows dropped", 0,
                       len(rows) - int(fitted["n_rows"].iloc[0]), True),
                _check(f"{tournament}: {label.lower()} fitting warnings captured", np.nan,
                       len(fitted.attrs["warnings"]), True),
                _check(f"{tournament}: {label.lower()} constant terms omitted", np.nan,
                       len(fitted.attrs["omitted_terms"]), True),
            ])
        print(f"{tournament}: {fitted_outcomes} sustainability outcomes estimated.")
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SUSTAINABILITY_COLUMNS)
    write_table(cfg, "sustainability", table)
    write_table(cfg, "checks_sustainability", pd.DataFrame(checks, columns=CHECK_COLUMNS))


def _check(name: str, target: float, value: float, passed: bool) -> dict[str, object]:
    """Return one row in the shared checks schema."""
    return {"check": name, "target": float(target), "value": float(value), "passed": bool(passed)}
