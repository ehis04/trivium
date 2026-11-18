"""Expected result: ordered logit of loss/draw/win on the Elo gap, fitted per
tournament with each team left out of the fit for its own predictions.
Expected points = 3 * P(win) + P(draw). Over-performance = actual minus
expected points per match; shootouts count as draws, progress on penalties
flagged separately.

Main model (agreed decisions): host advantage 0, one model for group and
knockout matches, men and women fitted separately. The stage also runs the
luck bands and calibration (models/uncertainty.py), expected stage reached
(models/progression.py) and sensitivity variants
(models/expectation_sensitivity.py), and draws the headline charts
(charts/expectation.py).

Output tables: expected_results, overperformance, overperformance_team,
team_summary, overperformance_luck, luck_summary, expectation_calibration,
expected_progress, overperformance_sensitivity, sensitivity_scores,
sensitivity_comparison, checks_expectation; Excel workbook expectation.xlsx;
figures/*.html.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from trivium.config import Config
from trivium.storage import read_table, write_table

LOSS, DRAW, WIN = 0, 1, 2
POINTS = np.array([0.0, 1.0, 3.0])
PROB_COLUMNS = ["p_loss", "p_draw", "p_win"]
HOST_ADVANTAGE = 0.0
EXPECTED_RESULTS_COLUMNS = ["tournament", "match_id", "team", "opponent", "elo_gap", *PROB_COLUMNS, "expected_points"]
SUM_TOLERANCE_PER_ROW = 0.05
MONOTONIC_SPEARMAN = 0.95
LUCK_SIMS = 10_000
PROGRESS_SIMS = 10_000
TEAM_SUMMARY_COLUMNS = [
    "tournament", "team", "elo", "matches", "actual_points", "expected_points", "overperformance",
    "overperformance_per_match", "band_low", "band_high", "p_upper", "beyond_luck", "shootouts", "shootouts_won",
    "actual_stage_name", "expected_stage", "progress_over_expectation", "p_actual_or_better", "path_easier_by",
]

# fit_predict(train_rows, test_rows) -> array (len(test_rows), 3) of P(loss, draw, win)
FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def team_match_rows(matches: pd.DataFrame, elo: pd.DataFrame, host_advantage: float = 0.0) -> pd.DataFrame:
    """One row per team per match, from that team's view.

    elo: tournament, team, elo (the elo_pre_tournament table).
    host_advantage: Elo points added to the host's side of the gap when
    neutral_ground is false (the home team is the host).
    Scores exclude shootout kicks, so a shootout match is a draw here.
    """
    keep = ["tournament", "match_id", "match_date", "competition_stage", "shootout", "shootout_winner"]
    host_match = ~matches["neutral_ground"].astype(bool)
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        s = matches[keep].copy()
        s["team"] = matches[f"{side}_team"]
        s["opponent"] = matches[f"{other}_team"]
        s["goals_for"] = matches[f"{side}_score"]
        s["goals_against"] = matches[f"{other}_score"]
        s["is_host"] = host_match & (side == "home")
        s["opponent_is_host"] = host_match & (side == "away")
        sides.append(s)
    rows = pd.concat(sides, ignore_index=True)

    ratings = elo[["tournament", "team", "elo"]]
    rows = rows.merge(ratings.rename(columns={"elo": "team_elo"}), on=["tournament", "team"], how="left")
    rows = rows.merge(
        ratings.rename(columns={"team": "opponent", "elo": "opponent_elo"}), on=["tournament", "opponent"], how="left"
    )
    host_term = host_advantage * (rows["is_host"].astype(float) - rows["opponent_is_host"].astype(float))
    rows["elo_gap"] = rows["team_elo"] - rows["opponent_elo"] + host_term

    rows["result"] = np.select(
        [rows["goals_for"] > rows["goals_against"], rows["goals_for"] == rows["goals_against"]], [WIN, DRAW], LOSS
    )
    rows["actual_points"] = POINTS[rows["result"]]
    rows["shootout"] = rows["shootout"].fillna(False).astype(bool)
    rows["won_shootout"] = rows["shootout"] & (rows["shootout_winner"] == rows["team"])
    return rows.drop(columns="shootout_winner").sort_values(["tournament", "match_id", "team"], ignore_index=True)


def leave_team_out(rows: pd.DataFrame, fit_predict: FitPredict) -> pd.DataFrame:
    """P(loss, draw, win) for every row of one tournament; each team's rows are
    predicted by a fit on the matches that team did not play."""
    if rows["tournament"].nunique() > 1:
        raise ValueError("leave_team_out takes one tournament at a time.")
    probs = np.full((len(rows), 3), np.nan)
    for team in rows["team"].unique():
        own = (rows["team"] == team).to_numpy()
        train = rows[~rows["match_id"].isin(rows.loc[own, "match_id"])]
        probs[own] = fit_predict(train, rows[own])
    return pd.DataFrame(probs, columns=PROB_COLUMNS, index=rows.index)


def ordered_logit(features: Sequence[str] = ("elo_gap",)) -> FitPredict:
    """Ordered logit of result on the features (no constant; the two cut points
    play that role). Features are scaled by their training standard deviation."""
    cols = list(features)

    def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        scale = train[cols].std().replace(0, 1).to_numpy()
        x_train = train[cols].to_numpy(dtype=float) / scale
        x_test = test[cols].to_numpy(dtype=float) / scale
        fitted = OrderedModel(train["result"].to_numpy(), x_train, distr="logit").fit(method="bfgs", disp=False)
        return np.asarray(fitted.model.predict(fitted.params, exog=x_test))

    return fit_predict


def base_rates(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """No-skill reference: the training share of losses, draws and wins for every row."""
    freq = np.bincount(train["result"].to_numpy(), minlength=3) / len(train)
    return np.tile(freq, (len(test), 1))


def score_predictions(result: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Log loss, Brier (3 outcomes), ranked probability score and accuracy; lower is better except accuracy."""
    result = np.asarray(result, dtype=int)
    probs = np.asarray(probs, dtype=float)
    observed = np.eye(3)[result]
    p_true = probs[np.arange(len(result)), result]
    cum_gap = np.cumsum(probs, axis=1)[:, :2] - np.cumsum(observed, axis=1)[:, :2]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(p_true, 1e-15, 1.0)))),
        "brier": float(np.mean(np.sum((probs - observed) ** 2, axis=1))),
        "rps": float(np.mean(np.sum(cum_gap**2, axis=1) / 2)),
        "accuracy": float(np.mean(probs.argmax(axis=1) == result)),
    }


def compare_models(rows: pd.DataFrame, candidates: Mapping[str, FitPredict]) -> pd.DataFrame:
    """Leave-team-out scores per tournament for each candidate model."""
    out = []
    for tournament, t_rows in rows.groupby("tournament", sort=False):
        t_rows = t_rows.reset_index(drop=True)
        for name, fit_predict in candidates.items():
            probs = leave_team_out(t_rows, fit_predict).to_numpy()
            out.append({"tournament": tournament, "model": name, **score_predictions(t_rows["result"], probs)})
    return pd.DataFrame(out)


def overperformance(rows: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """rows + P(loss, draw, win), expected points and over-performance per team-match."""
    df = pd.concat([rows, probs], axis=1)
    df["expected_points"] = probs[PROB_COLUMNS].to_numpy() @ POINTS
    df["overperformance"] = df["actual_points"] - df["expected_points"]
    return df


def team_overperformance(table: pd.DataFrame) -> pd.DataFrame:
    """Per team: matches, actual and expected points, over-performance summed and per match."""
    g = table.groupby(["tournament", "team"], as_index=False)
    out = g.agg(
        matches=("match_id", "nunique"),
        actual_points=("actual_points", "sum"),
        expected_points=("expected_points", "sum"),
        overperformance=("overperformance", "sum"),
        overperformance_per_match=("overperformance", "mean"),
        shootouts=("shootout", "sum"),
        shootouts_won=("won_shootout", "sum"),
    )
    return out.sort_values(["tournament", "overperformance"], ascending=[True, False], ignore_index=True)


def expectation_tables(matches: pd.DataFrame, elo_pre: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Main model per tournament: expected_results, overperformance, overperformance_team."""
    rows = team_match_rows(matches, elo_pre, host_advantage=HOST_ADVANTAGE)
    no_elo = sorted(set(rows.loc[rows["team_elo"].isna(), "team"]))
    if no_elo:
        raise ValueError(f"No pre-tournament Elo for: {', '.join(no_elo)}.")
    tables = []
    for _, t_rows in rows.groupby("tournament", sort=False):
        t_rows = t_rows.reset_index(drop=True)
        tables.append(overperformance(t_rows, leave_team_out(t_rows, ordered_logit())))
    table = pd.concat(tables, ignore_index=True)
    return {
        "expected_results": table[EXPECTED_RESULTS_COLUMNS],
        "overperformance": table,
        "overperformance_team": team_overperformance(table),
    }


def _check(name: str, target: float, value: float, passed: bool) -> dict[str, object]:
    return {"check": name, "target": float(target), "value": float(value), "passed": bool(passed)}


def checks_expectation(table: pd.DataFrame) -> pd.DataFrame:
    """Checks on the over-performance table, per tournament (shared checks schema)."""
    rows = []
    for tournament, t in table.groupby("tournament", sort=False):
        t = t.reset_index(drop=True)
        probs = t[PROB_COLUMNS].to_numpy()
        prob_error = float(np.abs(probs.sum(axis=1) - 1).max())
        spearman = float(t["elo_gap"].corr(t["p_win"], method="spearman"))
        outside = int(((t["expected_points"] < 0) | (t["expected_points"] > 3)).sum())
        total = float(t["overperformance"].sum())
        model_rps = score_predictions(t["result"], probs)["rps"]
        base_rps = score_predictions(t["result"], leave_team_out(t, base_rates).to_numpy())["rps"]
        rows += [
            _check(f"{tournament}: probabilities sum to 1 (largest error)", 0, prob_error, prob_error <= 1e-6),
            _check(f"{tournament}: P(win) rises with the Elo gap (Spearman)", 1, spearman,
                   spearman >= MONOTONIC_SPEARMAN),
            _check(f"{tournament}: expected points outside [0, 3] (rows)", 0, outside, outside == 0),
            _check(f"{tournament}: summed over-performance over {len(t)} team-matches", 0, total,
                   abs(total) <= SUM_TOLERANCE_PER_ROW * len(t)),
            _check(f"{tournament}: ranked probability score, ordered logit vs base rates (leave-team-out)",
                   base_rps, model_rps, model_rps < base_rps),
        ]
        group = t["competition_stage"].str.contains("group", case=False, na=False)
        for label, mask in (("group", group), ("knockout", ~group)):
            share = float((t.loc[mask, "result"] == DRAW).mean()) if mask.any() else np.nan
            rows.append(_check(f"{tournament}: draw share, {label} matches", np.nan, share, True))
    return pd.DataFrame(rows)


def write_excel(path: Path, sheets: Mapping[str, pd.DataFrame]) -> Path:
    """One tab per table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return path


def team_summary(team_table: pd.DataFrame, bands: pd.DataFrame, progress: pd.DataFrame) -> pd.DataFrame:
    """One row per team: points over expectation with its luck band, and stage over expectation."""
    keys = ["tournament", "team"]
    df = team_table.merge(bands[[*keys, "band_low", "band_high", "p_upper", "beyond_luck"]], on=keys, how="left")
    df = df.merge(
        progress[[*keys, "elo", "actual_stage_name", "expected_stage", "progress_over_expectation",
                  "p_actual_or_better", "path_easier_by"]],
        on=keys,
        how="left",
    )
    return df[TEAM_SUMMARY_COLUMNS].sort_values(["tournament", "overperformance"], ascending=[True, False],
                                                ignore_index=True)


def write_figures(folder: Path, figures: Mapping[str, object]) -> list[Path]:
    """Each Plotly figure as an HTML file; plotly.min.js is written once beside them (works offline)."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = folder / f"{name}.html"
        fig.write_html(path, include_plotlyjs="directory")
        paths.append(path)
    return paths


def _summary(summary: pd.DataFrame, table: pd.DataFrame, luck: pd.DataFrame, n: int = 10) -> str:
    cols = ["team", "matches", "actual_points", "expected_points", "overperformance", "band_low", "band_high",
            "beyond_luck", "actual_stage_name", "expected_stage", "progress_over_expectation"]
    parts = ["=== teams beyond luck", luck.to_string(index=False)]
    for tournament, t in summary.groupby("tournament", sort=False):
        t = t[cols].round(2)
        parts += [f"=== {tournament}: top {n}", t.head(n).to_string(index=False),
                  f"=== {tournament}: bottom {n}", t.tail(n).to_string(index=False)]
    ireland = table[table["team"].str.contains("Ireland", case=False, na=False)]
    if not ireland.empty:
        match_cols = ["competition_stage", "team", "opponent", "goals_for", "goals_against", "shootout",
                      "won_shootout", "elo_gap", "expected_points", "actual_points", "overperformance"]
        parts += ["=== Ireland", ireland[match_cols].round(2).to_string(index=False)]
    return "\n".join(parts)


def run(cfg: Config, luck_sims: int | None = None, progress_sims: int | None = None) -> None:
    # Imported here because these modules import from this one.
    from trivium.charts import expectation as charts
    from trivium.models import expectation_sensitivity as sensitivity
    from trivium.models import progression, uncertainty

    matches = read_table(cfg, "matches")
    elo_pre = read_table(cfg, "elo_pre_tournament")
    tds_team = read_table(cfg, "tds_team")

    tables = expectation_tables(matches, elo_pre)
    table = tables["overperformance"]
    bands = uncertainty.luck_band(table, n_sims=LUCK_SIMS if luck_sims is None else luck_sims)
    calibration = uncertainty.calibration_table(table)
    progress = progression.expected_progress(
        matches, elo_pre, n_sims=PROGRESS_SIMS if progress_sims is None else progress_sims
    )
    variant_teams, variant_scores = sensitivity.run_variants(matches, elo_pre, tds_team)
    comparison = sensitivity.compare_to_main(variant_teams, sensitivity.knockout_teams(matches))
    checks = pd.concat(
        [
            checks_expectation(table),
            uncertainty.checks_uncertainty(bands, table),
            uncertainty.checks_calibration(uncertainty.calibration_error(calibration)),
            progression.checks_progression(progress, matches, elo_pre),
            sensitivity.checks_sensitivity(comparison),
        ],
        ignore_index=True,
    )
    tables.update(
        {
            "team_summary": team_summary(tables["overperformance_team"], bands, progress),
            "overperformance_luck": bands,
            "luck_summary": uncertainty.luck_summary(bands),
            "expectation_calibration": calibration,
            "expected_progress": progress,
            "overperformance_sensitivity": variant_teams,
            "sensitivity_scores": variant_scores,
            "sensitivity_comparison": comparison,
            "checks_expectation": checks,
        }
    )
    for name, df in tables.items():
        write_table(cfg, name, df)

    write_excel(
        cfg.build_dir / "expectation.xlsx",
        {
            "team_summary": tables["team_summary"],
            "luck_summary": tables["luck_summary"],
            "expected_progress": progress,
            "sensitivity_comparison": comparison,
            "sensitivity_rps": sensitivity.scores_summary(variant_scores),
            "expectation_calibration": calibration,
            "overperformance": table,
            "checks_expectation": checks,
        },
    )
    figures = {}
    for tournament in matches["tournament"].drop_duplicates():
        figures[f"overperformance_{tournament}"] = charts.overperformance_chart(
            tables["overperformance_team"], tournament, bands=bands
        )
        figures[f"expected_vs_actual_points_{tournament}"] = charts.expected_vs_actual_points(
            tables["overperformance_team"], tournament
        )
        figures[f"expected_vs_actual_stage_{tournament}"] = charts.expected_vs_actual_stage(progress, tournament)
        figures[f"calibration_{tournament}"] = charts.calibration_chart(calibration, tournament)
    write_figures(cfg.build_dir / "figures", figures)

    with pd.option_context("display.width", 180, "display.max_colwidth", 90):
        print(checks.to_string(index=False))
        print(_summary(tables["team_summary"], table, tables["luck_summary"]))
