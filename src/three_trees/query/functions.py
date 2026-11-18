"""Query box function catalogue, executed on DuckDB/Parquet tables at team and
position-group level only.

Every function returns a Result object containing an aggregated DataFrame, key numbers
for text generation, optional Plotly charts, and table availability status. Player-level
minors data is strictly forbidden and blocked by ensure_aggregated().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rapidfuzz.process

from three_trees.charts import match as match_charts
from three_trees.config import Config
from three_trees.definitions import POINTS, POINT_DEFINITIONS, RESEARCH_QUESTION
from three_trees.measures import load_measures, measure as get_measure, measures as filter_measures
from three_trees.models import answer as answer_model
from three_trees.storage import read_table, table_exists

MAX_ROWS = 50
FORBIDDEN_COLUMNS = frozenset(
    {
        "person_id",
        "player_id",
        "player_name",
        "person_name",
        "jersey_number",
        "date_of_birth",
        "dob",
        "birth_date",
    }
)
ACCENT = "#0072B2"
GREY = "#777777"


class QueryError(ValueError):
    """Plain-English error message suitable for displaying to a user."""


@dataclass(frozen=True)
class Result:
    function: str
    args: dict[str, object]
    title: str
    table: pd.DataFrame  # aggregated rows only, at most MAX_ROWS
    numbers: dict[str, object]  # the key numbers the sentences may use
    chart: go.Figure | None
    sources: tuple[str, ...]  # tables read
    available: bool  # False when a needed table is not built yet
    note: str  # e.g. "Needs table style_types (styles stage)."


def ensure_aggregated(result: Result) -> Result:
    """Validate that the result contains only aggregated data with no forbidden columns."""
    if len(result.table) > MAX_ROWS:
        raise ValueError(f"Result table has {len(result.table)} rows, exceeding MAX_ROWS={MAX_ROWS}.")

    forbidden_found = [c for c in result.table.columns if str(c).lower() in FORBIDDEN_COLUMNS]
    if forbidden_found:
        raise ValueError(f"Forbidden player-level columns found in result table: {forbidden_found}.")

    return result


def _get_tournament_teams(cfg: Config, tournament: str) -> list[str]:
    """Retrieve unique team names for the tournament."""
    if table_exists(cfg, "team_summary"):
        ts = read_table(cfg, "team_summary")
        t_teams = ts[ts["tournament"] == tournament]["team"].dropna().unique().tolist()
        if t_teams:
            return sorted(t_teams)

    if table_exists(cfg, "matches"):
        m = read_table(cfg, "matches")
        t_m = m[m["tournament"] == tournament]
        teams = set(t_m["home_team"].dropna()).union(t_m["away_team"].dropna())
        if teams:
            return sorted(teams)

    if table_exists(cfg, "elo_pre_tournament"):
        elo = read_table(cfg, "elo_pre_tournament")
        t_elo = elo[elo["tournament"] == tournament]["team"].dropna().unique().tolist()
        if t_elo:
            return sorted(t_elo)

    return []


def resolve_team(cfg: Config, name: str, tournament: str) -> str:
    """Resolve a team name using fuzzy matching (rapidfuzz, score cutoff 80)."""
    teams = _get_tournament_teams(cfg, tournament)
    if not teams:
        raise QueryError(f"No teams found in the {tournament}'s tournament build.")

    match = rapidfuzz.process.extractOne(name, teams, score_cutoff=80)
    if match is None:
        raise QueryError(f"Team {name!r} not recognized in the {tournament}'s tournament.")
    return str(match[0])


def _base_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="simple_white", font={"size": 14})
    return fig


# ---------------------------------------------------------------- catalogue functions


def list_measures(cfg: Config, tournament: str = "men") -> Result:
    """List available registry measures for the tournament with labels and units."""
    reg = filter_measures(tournament=tournament)
    cols = ["measure", "label", "unit", "signifier", "group", "definition"]
    df = reg[cols].copy().sort_values(["group", "measure"], ignore_index=True)
    df["signifier"] = df["signifier"].astype(str).replace("<NA>", "")

    numbers = {
        "tournament": tournament,
        "total_measures": len(df),
        "signifier_count": int(df["signifier"].ne("").sum()),
    }
    return ensure_aggregated(
        Result(
            function="list_measures",
            args={"tournament": tournament},
            title=f"Available measures ({tournament})",
            table=df.head(MAX_ROWS),
            numbers=numbers,
            chart=None,
            sources=("measures_registry",),
            available=True,
            note="",
        )
    )


def team_summary(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Senior Elo, points vs expectation, luck band, stage reached, and key dominance metrics."""
    if not table_exists(cfg, "team_summary"):
        return Result(
            function="team_summary",
            args={"team": team, "tournament": tournament},
            title=f"Team summary: {team}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("team_summary", "team_match"),
            available=False,
            note="Needs table team_summary (expectation stage).",
        )

    resolved = resolve_team(cfg, team, tournament)
    ts = read_table(cfg, "team_summary")
    row = ts[(ts["tournament"] == tournament) & (ts["team"] == resolved)]
    if row.empty:
        raise QueryError(f"Team {resolved!r} not found in team summary table.")
    r = row.iloc[0]

    table_data = [
        {"Metric": "Senior Elo Rating", "Value": f"{r['elo']:.0f}"},
        {"Metric": "Matches Played", "Value": f"{r['matches']}"},
        {"Metric": "Actual Points", "Value": f"{r['actual_points']:.1f}"},
        {"Metric": "Expected Points", "Value": f"{r['expected_points']:.2f}"},
        {"Metric": "Over-performance", "Value": f"{r['overperformance']:+.2f}"},
        {"Metric": "Over-performance / Match", "Value": f"{r['overperformance_per_match']:+.2f}"},
        {"Metric": "Luck Band (95% CI)", "Value": f"[{r['band_low']:+.2f}, {r['band_high']:+.2f}]"},
        {"Metric": "Performance Verdict", "Value": str(r["beyond_luck"]) or "within luck band"},
        {"Metric": "Stage Reached", "Value": str(r.get("actual_stage_name", ""))},
        {"Metric": "Expected Stage", "Value": f"{r.get('expected_stage', 0.0):.1f}"},
        {"Metric": "Progress vs Expectation", "Value": f"{r.get('progress_over_expectation', 0.0):+.2f}"},
    ]
    df = pd.DataFrame(table_data)

    numbers = {
        "team": resolved,
        "tournament": tournament,
        "elo": round(float(r["elo"]), 1),
        "matches": int(r["matches"]),
        "actual_points": round(float(r["actual_points"]), 1),
        "expected_points": round(float(r["expected_points"]), 2),
        "overperformance": round(float(r["overperformance"]), 2),
        "overperformance_per_match": round(float(r["overperformance_per_match"]), 2),
        "band_low": round(float(r["band_low"]), 2),
        "band_high": round(float(r["band_high"]), 2),
        "beyond_luck": str(r["beyond_luck"]),
        "actual_stage": str(r.get("actual_stage_name", "")),
        "expected_stage": round(float(r.get("expected_stage", 0.0)), 1),
        "progress_over_expectation": round(float(r.get("progress_over_expectation", 0.0)), 2),
    }

    # Chart: over-performance bar with luck band error lines
    fig = _base_figure()
    fig.add_trace(
        go.Bar(
            x=[resolved],
            y=[numbers["overperformance"]],
            marker_color=ACCENT,
            name="Over-performance",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[numbers["band_high"] - numbers["overperformance"]],
                arrayminus=[numbers["overperformance"] - numbers["band_low"]],
                color=GREY,
            ),
        )
    )
    fig.update_layout(yaxis_title="Points vs Expectation", showlegend=False)

    return ensure_aggregated(
        Result(
            function="team_summary",
            args={"team": resolved, "tournament": tournament},
            title=f"Summary: {resolved} ({tournament})",
            table=df,
            numbers=numbers,
            chart=fig,
            sources=("team_summary", "team_match"),
            available=True,
            note="",
        )
    )


def rank_teams(
    cfg: Config,
    measure: str,
    tournament: str = "men",
    n: int = 10,
    order: str = "best",
) -> Result:
    """Rank top or bottom n teams by any registry measure or expectation metric."""
    n_clamped = max(1, min(MAX_ROWS, int(n)))

    try:
        m_info = get_measure(measure)
    except ValueError:
        raise QueryError(f"Measure {measure!r} is not in the measures registry.")

    if tournament == "women" and m_info["tier"] == "men":
        raise QueryError(f"Measure {measure!r} ({m_info['label']}) is only available for the men's tournament.")

    source_table = str(m_info["table"])
    col = str(m_info["column"])

    # Determine sorting direction
    higher_better = m_info["higher_is_better"]
    if order in ("best", "highest"):
        ascending = not higher_better if (order == "best" and higher_better is not None) else False
    else:  # worst, lowest
        ascending = higher_better if (order == "worst" and higher_better is not None) else True

    if source_table == "overperformance":
        table_name = "overperformance_team" if table_exists(cfg, "overperformance_team") else "team_summary"
        if not table_exists(cfg, table_name):
            return Result(
                function="rank_teams",
                args={"measure": measure, "tournament": tournament, "n": n, "order": order},
                title=f"Rank teams by {m_info['label']}",
                table=pd.DataFrame(),
                numbers={},
                chart=None,
                sources=(table_name,),
                available=False,
                note=f"Needs table {table_name}.",
            )
        df_src = read_table(cfg, table_name)
        df_tourney = df_src[df_src["tournament"] == tournament].copy()
        if col not in df_tourney.columns:
            raise QueryError(f"Column '{col}' not found in {table_name}.")
        ranked = (
            df_tourney.sort_values(col, ascending=ascending)
            .head(n_clamped)[["team", col]]
            .reset_index(drop=True)
        )
    else:
        if not table_exists(cfg, "team_match"):
            return Result(
                function="rank_teams",
                args={"measure": measure, "tournament": tournament, "n": n, "order": order},
                title=f"Rank teams by {m_info['label']}",
                table=pd.DataFrame(),
                numbers={},
                chart=None,
                sources=("team_match",),
                available=False,
                note="Needs table team_match.",
            )
        tm = read_table(cfg, "team_match")
        tm_tourney = tm[tm["tournament"] == tournament]
        if col not in tm_tourney.columns:
            raise QueryError(f"Column '{col}' not found in team_match table.")
        agg = tm_tourney.groupby("team", as_index=False)[col].mean()
        ranked = agg.sort_values(col, ascending=ascending).head(n_clamped).reset_index(drop=True)

    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    unit_str = f" ({m_info['unit']})" if m_info["unit"] else ""
    ranked = ranked.rename(columns={col: f"{m_info['label']}{unit_str}"})

    val_col = f"{m_info['label']}{unit_str}"
    top_team = str(ranked["team"].iloc[0]) if not ranked.empty else ""
    top_val = round(float(ranked[val_col].iloc[0]), 3) if not ranked.empty else 0.0

    numbers = {
        "measure": measure,
        "label": str(m_info["label"]),
        "unit": str(m_info["unit"]),
        "tournament": tournament,
        "n": len(ranked),
        "order": order,
        "top_team": top_team,
        "top_value": top_val,
    }

    # Chart
    fig = _base_figure()
    fig.add_trace(
        go.Bar(
            x=ranked[val_col],
            y=ranked["team"],
            orientation="h",
            marker_color=ACCENT,
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title=f"{m_info['label']}{unit_str}",
        showlegend=False,
    )

    return ensure_aggregated(
        Result(
            function="rank_teams",
            args={"measure": measure, "tournament": tournament, "n": n, "order": order},
            title=f"Top {len(ranked)} teams by {m_info['label']} ({tournament})",
            table=ranked,
            numbers=numbers,
            chart=fig,
            sources=("team_match", "overperformance_team"),
            available=True,
            note="",
        )
    )


def compare_teams(
    cfg: Config,
    team_a: str,
    team_b: str,
    tournament: str = "men",
    measures: list[str] | None = None,
) -> Result:
    """Side-by-side comparison of two teams across key metrics with tournament ranks."""
    if not table_exists(cfg, "team_match") or not table_exists(cfg, "team_summary"):
        return Result(
            function="compare_teams",
            args={"team_a": team_a, "team_b": team_b, "tournament": tournament, "measures": measures or []},
            title=f"Compare {team_a} vs {team_b}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("team_match", "team_summary"),
            available=False,
            note="Needs tables team_match and team_summary.",
        )

    t_a = resolve_team(cfg, team_a, tournament)
    t_b = resolve_team(cfg, team_b, tournament)

    tm = read_table(cfg, "team_match")
    tm = tm[tm["tournament"] == tournament]
    ts = read_table(cfg, "team_summary")
    ts = ts[ts["tournament"] == tournament]

    # Select measures: signifiers + dominance parts by default
    if not measures:
        reg = filter_measures(tournament=tournament)
        selected = reg[(reg["signifier"].notna()) | (reg["group"] == "dominance")]["measure"].tolist()
        # Ensure conversion, security, territory_share are included
        for m in ("conversion", "security", "territory_share", "dominance"):
            if m not in selected and m in reg["measure"].values:
                selected.append(m)
    else:
        selected = list(measures)

    # Compute team means and tournament ranks for each measure
    comp_rows = []
    num_dict: dict[str, Any] = {"team_a": t_a, "team_b": t_b, "tournament": tournament}

    for m_name in selected:
        try:
            m_info = get_measure(m_name)
        except ValueError:
            continue
        if tournament == "women" and m_info["tier"] == "men":
            continue

        col = str(m_info["column"])
        if col in tm.columns:
            means = tm.groupby("team")[col].mean()
        elif col in ts.columns:
            means = ts.set_index("team")[col]
        else:
            continue

        if t_a not in means.index or t_b not in means.index:
            continue

        val_a = float(means.loc[t_a])
        val_b = float(means.loc[t_b])

        higher_better = m_info["higher_is_better"]
        asc = not higher_better if higher_better is not None else False
        ranks = means.rank(ascending=asc, method="min")
        rank_a = int(ranks.loc[t_a])
        rank_b = int(ranks.loc[t_b])

        comp_rows.append(
            {
                "measure": m_name,
                "label": m_info["label"],
                f"{t_a} Value": round(val_a, 2),
                f"{t_a} Rank": rank_a,
                f"{t_b} Value": round(val_b, 2),
                f"{t_b} Rank": rank_b,
                "unit": m_info["unit"],
            }
        )
        num_dict[f"{m_name}_{t_a}"] = round(val_a, 2)
        num_dict[f"{m_name}_{t_b}"] = round(val_b, 2)

    df_comp = pd.DataFrame(comp_rows)

    # Chart comparing the first 5 measures
    fig = _base_figure()
    if not df_comp.empty:
        plot_df = df_comp.head(6)
        fig.add_trace(go.Bar(name=t_a, x=plot_df["label"], y=plot_df[f"{t_a} Value"], marker_color=ACCENT))
        fig.add_trace(go.Bar(name=t_b, x=plot_df["label"], y=plot_df[f"{t_b} Value"], marker_color=GREY))
        fig.update_layout(barmode="group", xaxis_tickangle=-30)

    return ensure_aggregated(
        Result(
            function="compare_teams",
            args={"team_a": t_a, "team_b": t_b, "tournament": tournament, "measures": selected},
            title=f"Comparison: {t_a} vs {t_b} ({tournament})",
            table=df_comp,
            numbers=num_dict,
            chart=fig,
            sources=("team_match", "team_summary"),
            available=True,
            note="",
        )
    )


def explain_overperformance(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Over-performance verdict with luck band, per-match points, and signifier analysis."""
    if not table_exists(cfg, "team_summary") or not table_exists(cfg, "overperformance"):
        return Result(
            function="explain_overperformance",
            args={"team": team, "tournament": tournament},
            title=f"Explain over-performance: {team}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("team_summary", "overperformance", "team_match"),
            available=False,
            note="Needs tables team_summary and overperformance.",
        )

    resolved = resolve_team(cfg, team, tournament)
    ts = read_table(cfg, "team_summary")
    row = ts[(ts["tournament"] == tournament) & (ts["team"] == resolved)]
    if row.empty:
        raise QueryError(f"Team {resolved!r} not found in team summary table.")
    r = row.iloc[0]

    op = read_table(cfg, "overperformance")
    matches_df = op[(op["tournament"] == tournament) & (op["team"] == resolved)].copy()
    matches_df = matches_df.sort_values("match_id").reset_index(drop=True)

    match_rows = []
    for _, m in matches_df.iterrows():
        match_rows.append(
            {
                "Stage": str(m["competition_stage"]),
                "Opponent": str(m["opponent"]),
                "Result": "Win" if m["result"] == 2 else ("Draw" if m["result"] == 1 else "Loss"),
                "Actual Pts": float(m["actual_points"]),
                "Expected Pts": round(float(m["expected_points"]), 2),
                "Over-perf": round(float(m["overperformance"]), 2),
            }
        )
    table_df = pd.DataFrame(match_rows)

    verdict = (
        "significantly above expectation (exceeds luck band)"
        if r["beyond_luck"] == "above"
        else (
            "significantly below expectation (below luck band)"
            if r["beyond_luck"] == "below"
            else "within expected variation (within luck band)"
        )
    )

    numbers = {
        "team": resolved,
        "tournament": tournament,
        "elo": round(float(r["elo"]), 1),
        "actual_points": round(float(r["actual_points"]), 1),
        "expected_points": round(float(r["expected_points"]), 2),
        "overperformance": round(float(r["overperformance"]), 2),
        "band_low": round(float(r["band_low"]), 2),
        "band_high": round(float(r["band_high"]), 2),
        "verdict": verdict,
        "matches_count": len(table_df),
    }

    # Chart: Actual vs Expected points per match
    fig = _base_figure()
    if not table_df.empty:
        x_labels = [f"{row['Stage']}: vs {row['Opponent']}" for _, row in table_df.iterrows()]
        fig.add_trace(go.Bar(name="Actual Points", x=x_labels, y=table_df["Actual Pts"], marker_color=ACCENT))
        fig.add_trace(go.Bar(name="Expected Points", x=x_labels, y=table_df["Expected Pts"], marker_color=GREY))
        fig.update_layout(barmode="group", yaxis_title="Points", xaxis_tickangle=-30)

    return ensure_aggregated(
        Result(
            function="explain_overperformance",
            args={"team": resolved, "tournament": tournament},
            title=f"Over-performance breakdown: {resolved} ({tournament})",
            table=table_df,
            numbers=numbers,
            chart=fig,
            sources=("team_summary", "overperformance", "team_match"),
            available=True,
            note="",
        )
    )


def match_review(cfg: Config, team: str, opponent: str, tournament: str = "men") -> Result:
    """Detailed review of a head-to-head match: actual score, expectations, and dominance."""
    if not table_exists(cfg, "matches") or not table_exists(cfg, "expected_results"):
        return Result(
            function="match_review",
            args={"team": team, "opponent": opponent, "tournament": tournament},
            title=f"Match review: {team} vs {opponent}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("matches", "expected_results", "team_match"),
            available=False,
            note="Needs tables matches and expected_results.",
        )

    t_a = resolve_team(cfg, team, tournament)
    t_b = resolve_team(cfg, opponent, tournament)

    matches = read_table(cfg, "matches")
    matches = matches[matches["tournament"] == tournament]
    m = matches[
        ((matches["home_team"] == t_a) & (matches["away_team"] == t_b))
        | ((matches["home_team"] == t_b) & (matches["away_team"] == t_a))
    ]
    if m.empty:
        raise QueryError(f"No match found between {t_a} and {t_b} in the {tournament}'s tournament.")

    match_row = m.iloc[0]
    match_id = match_row["match_id"]

    exp = read_table(cfg, "expected_results")
    exp = exp[(exp["tournament"] == tournament) & (exp["match_id"] == match_id)]
    if not {t_a, t_b}.issubset(set(exp["team"])):
        return _unavailable("match_review", {"team": t_a, "opponent": t_b, "tournament": tournament},
                            ("expected_results",), "Expected results for this match are not built in this build.")
    exp_team = exp[exp["team"] == t_a].iloc[0]
    exp_opp = exp[exp["team"] == t_b].iloc[0]

    # Team match stats if available
    tm_stats: dict[str, pd.Series] = {}
    if table_exists(cfg, "team_match"):
        tm = read_table(cfg, "team_match")
        tm = tm[(tm["tournament"] == tournament) & (tm["match_id"] == match_id)]
        for t in (t_a, t_b):
            t_match = tm[(tm["match_id"] == match_id) & (tm["team"] == t)]
            if not t_match.empty:
                tm_stats[t] = t_match.iloc[0]

    score_str = f"{match_row['home_team']} {match_row['home_score']} - {match_row['away_score']} {match_row['away_team']}"
    if match_row["shootout"]:
        score_str += f" (penalties won by {match_row['shootout_winner']})"

    rows = [
        {"Metric": "Stage", t_a: str(match_row["competition_stage"]), t_b: str(match_row["competition_stage"])},
        {"Metric": "Final Score", t_a: score_str, t_b: score_str},
        {"Metric": "Win Probability", t_a: f"{exp_team['p_win']:.1%}", t_b: f"{exp_opp['p_win']:.1%}"},
        {"Metric": "Draw Probability", t_a: f"{exp_team['p_draw']:.1%}", t_b: f"{exp_opp['p_draw']:.1%}"},
        {"Metric": "Expected Points", t_a: f"{exp_team['expected_points']:.2f}", t_b: f"{exp_opp['expected_points']:.2f}"},
    ]

    for key, label in [
        ("territory_share", "Field Tilt"),
        ("contests_share", "Contests Share"),
        ("chances_share", "Chances Share"),
        ("np_xg", "Non-penalty xG"),
    ]:
        if t_a in tm_stats and key in tm_stats[t_a] and t_b in tm_stats and key in tm_stats[t_b]:
            val_a = tm_stats[t_a][key]
            val_b = tm_stats[t_b][key]
            rows.append({
                "Metric": label,
                t_a: f"{float(val_a):.2f}" if pd.notna(val_a) else "N/A",
                t_b: f"{float(val_b):.2f}" if pd.notna(val_b) else "N/A",
            })

    table_df = pd.DataFrame(rows)

    numbers = {
        "match_id": int(match_id),
        "stage": str(match_row["competition_stage"]),
        "team": t_a,
        "opponent": t_b,
        "home_team": str(match_row["home_team"]),
        "away_team": str(match_row["away_team"]),
        "home_score": int(match_row["home_score"]),
        "away_score": int(match_row["away_score"]),
        "shootout": bool(match_row["shootout"]),
        "p_win_team": round(float(exp_team["p_win"]), 3),
        "p_win_opp": round(float(exp_opp["p_win"]), 3),
        "expected_points_team": round(float(exp_team["expected_points"]), 2),
        "expected_points_opp": round(float(exp_opp["expected_points"]), 2),
    }

    # Chart: win/draw/loss probabilities
    fig = _base_figure()
    fig.add_trace(go.Bar(name=t_a, x=["Win", "Draw", "Loss"], y=[exp_team["p_win"], exp_team["p_draw"], exp_team["p_loss"]], marker_color=ACCENT))
    fig.add_trace(go.Bar(name=t_b, x=["Win", "Draw", "Loss"], y=[exp_opp["p_win"], exp_opp["p_draw"], exp_opp["p_loss"]], marker_color=GREY))
    fig.update_layout(barmode="group", yaxis_title="Probability", yaxis_tickformat=".0%")
    if t_a in tm_stats:
        fig = match_charts.dominance_bars(pd.DataFrame(tm_stats.values()), t_a)

    return ensure_aggregated(
        Result(
            function="match_review",
            args={"team": t_a, "opponent": t_b, "tournament": tournament},
            title=f"Match: {t_a} vs {t_b} ({tournament})",
            table=table_df,
            numbers=numbers,
            chart=fig,
            sources=("matches", "expected_results", "team_match"),
            available=True,
            note="",
        )
    )


def style_of(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Style classification and tactical style profile for a team."""
    if not table_exists(cfg, "style_types"):
        return Result(
            function="style_of",
            args={"team": team, "tournament": tournament},
            title=f"Style of {team}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("style_types", "style_profiles"),
            available=False,
            note="Needs table style_types (styles stage).",
        )

    resolved = resolve_team(cfg, team, tournament)
    st = read_table(cfg, "style_types")
    row = st[(st["tournament"] == tournament) & (st["team"] == resolved)]
    if row.empty:
        raise QueryError(f"Team {resolved!r} not found in style types table.")
    r = row.iloc[0]

    table_data = [
        {"Attribute": "Assigned Style", "Value": str(r["style_label"])},
        {"Attribute": "Style ID", "Value": str(r["style_id"])},
        {"Attribute": "Membership Probability", "Value": f"{float(r.get('probability', 1.0)):.1%}"},
    ]
    df = pd.DataFrame(table_data)

    numbers = {
        "team": resolved,
        "tournament": tournament,
        "style_label": str(r["style_label"]),
        "style_id": int(r["style_id"]),
        "probability": round(float(r.get("probability", 1.0)), 2),
    }

    return ensure_aggregated(
        Result(
            function="style_of",
            args={"team": resolved, "tournament": tournament},
            title=f"Style: {resolved} ({tournament})",
            table=df,
            numbers=numbers,
            chart=None,
            sources=("style_types", "style_profiles"),
            available=True,
            note="",
        )
    )


def teams_by_style(cfg: Config, style: str, tournament: str = "men") -> Result:
    """List teams belonging to a given tactical style cluster with their over-performance."""
    if not table_exists(cfg, "style_types"):
        return Result(
            function="teams_by_style",
            args={"style": style, "tournament": tournament},
            title=f"Teams with style {style}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("style_types", "team_summary"),
            available=False,
            note="Needs table style_types (styles stage).",
        )

    st = read_table(cfg, "style_types")
    st = st[st["tournament"] == tournament]

    # Match style label or ID
    unique_labels = st["style_label"].dropna().unique().tolist()
    match = rapidfuzz.process.extractOne(style, unique_labels, scorer=rapidfuzz.fuzz.token_set_ratio,
                                         score_cutoff=70)
    if match is not None:
        matched_label = str(match[0])
        subset = st[st["style_label"] == matched_label]
    elif style.isdigit():
        subset = st[st["style_id"] == int(style)]
        matched_label = str(subset["style_label"].iloc[0]) if not subset.empty else style
    else:
        raise QueryError(f"Style {style!r} not recognized in {tournament} style clusters.")

    if subset.empty:
        raise QueryError(f"No teams found for style {style!r}.")

    cols = ["team", "probability"] if "probability" in subset.columns else ["team"]
    res_df = subset[cols].copy()

    if table_exists(cfg, "team_summary"):
        ts = read_table(cfg, "team_summary")
        ts = ts[ts["tournament"] == tournament]
        res_df = res_df.merge(
            ts[["team", "overperformance", "overperformance_per_match", "actual_stage_name"]],
            on="team",
            how="left",
            validate="one_to_one",
        )

    res_df = res_df.head(MAX_ROWS)
    numbers = {
        "style_label": matched_label,
        "tournament": tournament,
        "team_count": len(res_df),
        "teams": res_df["team"].tolist(),
    }

    return ensure_aggregated(
        Result(
            function="teams_by_style",
            args={"style": matched_label, "tournament": tournament},
            title=f"Teams matching style '{matched_label}' ({tournament})",
            table=res_df,
            numbers=numbers,
            chart=None,
            sources=("style_types", "team_summary"),
            available=True,
            note="",
        )
    )


def running_profile(
    cfg: Config,
    team: str,
    tournament: str = "men",
    position_group: str | None = None,
) -> Result:
    """Running volume and intensity (distance, high-speed, sprint) aggregated by position group."""
    if tournament == "women":
        raise QueryError("Physical running tracking data is available for the men's tournament only.")

    if not table_exists(cfg, "running_player_match"):
        return Result(
            function="running_profile",
            args={"team": team, "tournament": tournament, "position_group": position_group},
            title=f"Running profile: {team}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("running_player_match",),
            available=False,
            note="Needs table running_player_match.",
        )

    resolved = resolve_team(cfg, team, tournament)
    rpm = read_table(cfg, "running_player_match")
    t_df = rpm[(rpm["tournament"] == tournament) & (rpm["team"] == resolved)].copy()

    if position_group:
        unique_groups = t_df["position_group"].dropna().unique().tolist()
        match = rapidfuzz.process.extractOne(position_group, unique_groups, score_cutoff=75)
        if match is not None:
            t_df = t_df[t_df["position_group"] == match[0]]
        else:
            raise QueryError(f"Position group {position_group!r} not recognized. Valid: {unique_groups}")

    # Aggregate strictly by position_group and possession state - NEVER return player rows!
    agg = (
        t_df.groupby(["position_group", "state"], as_index=False)
        .agg(
            distance_km=("distance_m", lambda s: round(s.sum() / 1000.0, 2)),
            hsr_km=("hsr_m", lambda s: round(s.sum() / 1000.0, 2)),
            sprint_km=("sprint_m", lambda s: round(s.sum() / 1000.0, 2)),
            total_minutes=("minutes", "sum"),
        )
        .reset_index(drop=True)
    )

    total_dist = round(float(agg["distance_km"].sum()), 1)
    total_hsr = round(float(agg["hsr_km"].sum()), 1)

    numbers = {
        "team": resolved,
        "tournament": tournament,
        "total_distance_km": total_dist,
        "total_hsr_km": total_hsr,
        "position_groups": agg["position_group"].unique().tolist(),
    }

    # Chart
    fig = _base_figure()
    fig.add_trace(
        go.Bar(
            name="Distance (km)",
            x=agg["position_group"] + " (" + agg["state"] + ")",
            y=agg["distance_km"],
            marker_color=ACCENT,
        )
    )
    fig.update_layout(xaxis_tickangle=-30, yaxis_title="Distance (km)")

    return ensure_aggregated(
        Result(
            function="running_profile",
            args={"team": resolved, "tournament": tournament, "position_group": position_group},
            title=f"Running profile: {resolved} ({tournament})",
            table=agg,
            numbers=numbers,
            chart=fig,
            sources=("running_player_match",),
            available=True,
            note="",
        )
    )


def zone_map(
    cfg: Config,
    team: str,
    tournament: str = "men",
    opponent: str | None = None,
) -> Result:
    """Territory share (field tilt) and contests across 18 pitch zones with a 6x3 heatmap."""
    resolved_team = resolve_team(cfg, team, tournament)

    if opponent:
        resolved_opp = resolve_team(cfg, opponent, tournament)
        if not table_exists(cfg, "team_match_zone") or not table_exists(cfg, "matches"):
            return Result(
                function="zone_map",
                args={"team": team, "tournament": tournament, "opponent": opponent},
                title=f"Zone map: {resolved_team} vs {resolved_opp}",
                table=pd.DataFrame(),
                numbers={},
                chart=None,
                sources=("team_match_zone",),
                available=False,
                note="Needs table team_match_zone.",
            )
        matches = read_table(cfg, "matches")
        m = matches[
            (matches["tournament"] == tournament)
            & (
                ((matches["home_team"] == resolved_team) & (matches["away_team"] == resolved_opp))
                | ((matches["home_team"] == resolved_opp) & (matches["away_team"] == resolved_team))
            )
        ]
        if m.empty:
            raise QueryError(f"No match between {resolved_team} and {resolved_opp} found.")
        mid = m.iloc[0]["match_id"]
        tmz = read_table(cfg, "team_match_zone")
        zones_df = tmz[(tmz["match_id"] == mid) & (tmz["team"] == resolved_team)].copy()
        title = f"Zone territory: {resolved_team} vs {resolved_opp} ({tournament})"
    else:
        if not table_exists(cfg, "team_tournament_zone"):
            return Result(
                function="zone_map",
                args={"team": team, "tournament": tournament, "opponent": None},
                title=f"Zone map: {resolved_team}",
                table=pd.DataFrame(),
                numbers={},
                chart=None,
                sources=("team_tournament_zone",),
                available=False,
                note="Needs table team_tournament_zone.",
            )
        ttz = read_table(cfg, "team_tournament_zone")
        zones_df = ttz[(ttz["tournament"] == tournament) & (ttz["team"] == resolved_team)].copy()
        title = f"Tournament zone territory: {resolved_team} ({tournament})"

    if zones_df.empty:
        raise QueryError(f"No zone data found for {resolved_team}.")

    keep_cols = [
        c
        for c in ["zone_x", "zone_y", "zone", "territory_share", "contests_share", "actions_count"]
        if c in zones_df.columns
    ]
    table_df = zones_df[keep_cols].sort_values(["zone_y", "zone_x"]).reset_index(drop=True)

    # Build 3 x 6 2D matrix for heatmap (Y: 3 rows, X: 6 columns)
    grid = np.zeros((3, 6))
    for _, z in zones_df.iterrows():
        zx = int(z["zone_x"])
        zy = int(z["zone_y"])
        if 0 <= zx < 6 and 0 <= zy < 3:
            grid[zy, zx] = float(z["territory_share"])

    fig = _base_figure()
    fig.add_trace(
        go.Heatmap(
            z=grid,
            x=["Def 1", "Def 2", "Mid 1", "Mid 2", "Att 1", "Att 2"],
            y=["Left", "Centre", "Right"],
            colorscale="Blues",
            colorbar=dict(title="Territory Share"),
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    avg_tilt = round(float(zones_df["territory_share"].mean()), 3)
    att_tilt = round(float(zones_df[zones_df["zone_x"] >= 4]["territory_share"].mean()), 3)

    numbers = {
        "team": resolved_team,
        "tournament": tournament,
        "opponent": opponent,
        "mean_territory_share": avg_tilt,
        "attacking_third_territory": att_tilt,
    }

    return ensure_aggregated(
        Result(
            function="zone_map",
            args={"team": resolved_team, "tournament": tournament, "opponent": opponent},
            title=title,
            table=table_df,
            numbers=numbers,
            chart=fig,
            sources=("team_tournament_zone", "team_match_zone"),
            available=True,
            note="",
        )
    )


def conversion_by_phase(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Non-penalty expected goals and conversion by tactical possession phase."""
    if not table_exists(cfg, "team_tournament_phase"):
        return Result(
            function="conversion_by_phase",
            args={"team": team, "tournament": tournament},
            title=f"Conversion by phase: {team}",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("team_tournament_phase",),
            available=False,
            note="Needs table team_tournament_phase (tactical phases stage).",
        )

    resolved = resolve_team(cfg, team, tournament)
    ttp = read_table(cfg, "team_tournament_phase")
    df = ttp[(ttp["tournament"] == tournament) & (ttp["team"] == resolved)].copy()
    if df.empty:
        raise QueryError(f"No phase data found for team {resolved!r}.")

    numbers = {
        "team": resolved,
        "tournament": tournament,
        "phases_count": len(df),
    }

    return ensure_aggregated(
        Result(
            function="conversion_by_phase",
            args={"team": resolved, "tournament": tournament},
            title=f"Conversion by phase: {resolved} ({tournament})",
            table=df.head(MAX_ROWS),
            numbers=numbers,
            chart=None,
            sources=("team_tournament_phase",),
            available=True,
            note="",
        )
    )


def efficiency_index(cfg: Config, tournament: str = "men", n: int = 10) -> Result:
    """Tournament leaderboard ranked by the composite efficiency index."""
    if not table_exists(cfg, "efficiency_team_tournament"):
        return Result(
            function="efficiency_index",
            args={"tournament": tournament, "n": n},
            title=f"Efficiency index ({tournament})",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("efficiency_team_tournament",),
            available=False,
            note="Needs table efficiency_team_tournament (efficiency stage).",
        )

    ett = read_table(cfg, "efficiency_team_tournament")
    df = ett[ett["tournament"] == tournament].copy()
    if df.empty:
        raise QueryError(f"No efficiency data found for {tournament}.")

    n_clamped = max(1, min(MAX_ROWS, int(n)))
    ranked = df.sort_values("efficiency_index", ascending=False).head(n_clamped).reset_index(drop=True)

    numbers = {
        "tournament": tournament,
        "leader": str(ranked["team"].iloc[0]) if not ranked.empty else "",
        "top_index": round(float(ranked["efficiency_index"].iloc[0]), 2) if not ranked.empty else 0.0,
    }

    fig = _base_figure()
    fig.add_trace(go.Bar(x=ranked["efficiency_index"], y=ranked["team"], orientation="h", marker_color=ACCENT))
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Efficiency Index")

    return ensure_aggregated(
        Result(
            function="efficiency_index",
            args={"tournament": tournament, "n": n},
            title=f"Top {len(ranked)} teams by Efficiency Index ({tournament})",
            table=ranked,
            numbers=numbers,
            chart=fig,
            sources=("efficiency_team_tournament",),
            available=True,
            note="",
        )
    )


def compare_progressing_vs_exiting(cfg: Config, tournament: str = "men") -> Result:
    """Signifier profile comparison: teams that progressed vs exited, before and after strength adjustment."""
    if not table_exists(cfg, "profile_comparison"):
        return Result(
            function="compare_progressing_vs_exiting",
            args={"tournament": tournament},
            title=f"Progressing vs exiting teams ({tournament})",
            table=pd.DataFrame(),
            numbers={},
            chart=None,
            sources=("profile_comparison",),
            available=False,
            note="Needs table profile_comparison (profiles stage).",
        )

    pc = read_table(cfg, "profile_comparison")
    df = pc[pc["tournament"] == tournament].copy()
    if df.empty:
        raise QueryError(f"No profile comparison data for {tournament}.")

    from three_trees.charts.profiles import profile_comparison_chart

    by_verdict = df.groupby("verdict")["label"].apply(list).to_dict() if "verdict" in df else {}
    numbers = {
        "tournament": tournament,
        "total_measures": len(df),
        "surviving_adjustment_count": int(df["survives_adjustment"].sum()) if "survives_adjustment" in df else 0,
        "survives_adjustment": by_verdict.get("survives adjustment", []),
        "explained_by_strength": by_verdict.get("explained by strength", []),
        "only_after_adjustment": by_verdict.get("only after adjustment", []),
    }
    view = ["label", "raw_difference_z", "raw_low", "raw_high", "adjusted_difference_z", "adjusted_low",
            "adjusted_high", "verdict", "corr_with_elo"]

    return ensure_aggregated(
        Result(
            function="compare_progressing_vs_exiting",
            args={"tournament": tournament},
            title=f"Progressing vs Exiting signifiers ({tournament})",
            table=df[[c for c in view if c in df]].round(3).head(MAX_ROWS),
            numbers=numbers,
            chart=profile_comparison_chart(df, tournament),
            sources=("profile_comparison",),
            available=True,
            note="",
        )
    )


def _unavailable(
    function: str,
    args: dict[str, object],
    sources: tuple[str, ...],
    note: str = "",
) -> Result:
    """Return a plain availability answer with an empty aggregated table."""
    return ensure_aggregated(Result(
        function=function, args=args, title=function.replace("_", " ").capitalize(),
        table=pd.DataFrame(), numbers={}, chart=None, sources=sources, available=False,
        note=note or f"{', '.join(sources)} not built in this build.",
    ))


def _match_pair(
    cfg: Config, team: str, opponent: str, tournament: str,
) -> tuple[str, str, pd.Series]:
    """Return resolved teams and their first head-to-head fixture in this tournament."""
    first = resolve_team(cfg, team, tournament)
    second = resolve_team(cfg, opponent, tournament)
    matches = read_table(cfg, "matches")
    selected = matches.loc[matches["tournament"].eq(tournament) & (
        (matches["home_team"].eq(first) & matches["away_team"].eq(second))
        | (matches["home_team"].eq(second) & matches["away_team"].eq(first))
    )]
    if selected.empty:
        raise QueryError(f"No match found between {first} and {second} in the {tournament}'s tournament.")
    return first, second, selected.iloc[0]


def xg_timeline(cfg: Config, team: str, opponent: str, tournament: str = "men") -> Result:
    """Return both teams' xG, goals and goal times with a cumulative team-level chart."""
    args = {"team": team, "opponent": opponent, "tournament": tournament}
    needed = ("shot_timeline", "matches")
    missing = tuple(name for name in needed if not table_exists(cfg, name))
    if missing:
        return _unavailable("xg_timeline", args, missing)
    first, second, match = _match_pair(cfg, team, opponent, tournament)
    shots = read_table(cfg, "shot_timeline")
    selected = shots.loc[
        shots["tournament"].eq(tournament) & shots["match_id"].eq(match["match_id"])
        & shots["team"].isin([first, second]) & shots["period"].isin([1, 2])
    ].copy()
    columns = ["team", "elapsed_minute", "half_time_minute", "minute", "second", "xg",
               "goal", "penalty", "own_goal"]
    selected = selected[columns].sort_values("elapsed_minute", kind="stable", ignore_index=True)
    records = []
    numbers: dict[str, object] = {"team": first, "opponent": second, "tournament": tournament,
                                  "match_id": int(match["match_id"])}
    for side, name in (("team", first), ("opponent", second)):
        own = selected.loc[selected["team"].eq(name)]
        goals = own.loc[own["goal"].fillna(False).astype(bool)]
        minutes = [
            f"{int(row.minute) + 1}'" + (" (own goal)" if row.own_goal else " (penalty)" if row.penalty else "")
            for row in goals.itertuples(index=False)
        ]
        xg = float(own["xg"].sum())
        records.append({"team": name, "xg": xg, "goals": len(goals), "goal_minutes": ", ".join(minutes)})
        numbers.update({f"{side}_xg": round(xg, 2), f"{side}_goals": len(goals),
                        f"{side}_goal_minutes": minutes})
    return ensure_aggregated(Result(
        function="xg_timeline", args={"team": first, "opponent": second, "tournament": tournament},
        title=f"Chances over time: {first} vs {second}", table=pd.DataFrame(records), numbers=numbers,
        chart=match_charts.xg_timeline(selected, first, second), sources=needed, available=True, note="",
    ))


def _team_measure_comparison(
    cfg: Config, function: str, team: str, tournament: str, source: str, registry_rows: pd.DataFrame,
) -> Result:
    """Return team means against equally weighted tournament team means for registry measures."""
    args = {"team": team, "tournament": tournament}
    if tournament != "men":
        return _unavailable(function, args, (source,), "This comparison is built for the men's tournament only.")
    if not table_exists(cfg, source):
        return _unavailable(function, args, (source,))
    resolved = resolve_team(cfg, team, tournament)
    table = read_table(cfg, source)
    table = table.loc[table["tournament"].eq(tournament)].copy()
    if "state" in table:
        table = table.loc[table["state"].eq("without_ball")].copy()
    records = []
    for row in registry_rows.itertuples(index=False):
        if row.column not in table:
            continue
        values = pd.to_numeric(table[row.column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        measured = table.assign(_value=values)
        if "match_id" in measured:
            match_means = measured.groupby(["team", "match_id"])["_value"].mean()
            means = match_means.groupby("team").mean()
        else:
            means = measured.groupby("team")["_value"].mean()
        own = means.get(resolved, np.nan)
        if pd.notna(own):
            records.append({"measure": row.measure, "label": row.label, "unit": row.unit,
                            "team_mean": float(own), "tournament_mean": float(means.mean())})
    if not records:
        return _unavailable(function, args, (source,), f"{source} values for {resolved} are not built in this build.")
    result_table = pd.DataFrame(records)
    fig = _base_figure()
    fig.add_bar(name=resolved, x=result_table["label"], y=result_table["team_mean"], marker_color=ACCENT)
    fig.add_bar(name="Tournament team mean", x=result_table["label"],
                y=result_table["tournament_mean"], marker_color=GREY)
    unit = "Share of FIFA phase clip time" if function == "fifa_phases" else "Metres without the ball"
    fig.update_layout(barmode="group", yaxis_title=unit)
    if function == "fifa_phases":
        fig.update_layout(yaxis_tickformat=".0%")
    return ensure_aggregated(Result(
        function=function, args={"team": resolved, "tournament": tournament},
        title=f"{resolved}: {'FIFA phases' if function == 'fifa_phases' else 'team shape without the ball'}",
        table=result_table.head(MAX_ROWS),
        numbers={"team": resolved, "tournament": tournament, "measures": result_table.to_dict("records")},
        chart=fig, sources=(source,), available=True,
        note="Phase shares measure clip time; overlapping phases can cover the same second."
        if function == "fifa_phases" else "Each team's match means receive equal weight in the comparison.",
    ))


def fifa_phases(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Return a men's team's FIFA phase shares against the tournament's team means."""
    registry = filter_measures(tournament=tournament)
    registry = registry.loc[registry["measure"].str.match(r"^fifa_.+_share$")]
    return _team_measure_comparison(cfg, "fifa_phases", team, tournament, "team_match_provider", registry)


def team_shape(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Return line height, length and width without the ball against the men's tournament."""
    registry = filter_measures(tournament=tournament)
    registry = registry.loc[registry["measure"].isin(["line_height_m", "team_length_m", "team_width_m"])]
    source = "team_match_provider"
    for candidate in ("team_match_provider", "team_match", "shape_team_second"):
        if table_exists(cfg, candidate):
            available = read_table(cfg, candidate)
            if any(column in available and available[column].notna().any() for column in registry["column"]):
                source = candidate
                break
    return _team_measure_comparison(cfg, "team_shape", team, tournament, source, registry)


def findings(cfg: Config, tournament: str = "men") -> Result:
    """Return the tournament's approved finding-gate rows with their plain wording."""
    source = "finding_gate"
    if not table_exists(cfg, source):
        return _unavailable("findings", {"tournament": tournament}, (source,))
    gate = read_table(cfg, source)
    selected = gate.loc[gate["tournament"].eq(tournament) & gate["passes_gate"].fillna(False).astype(bool)]
    columns = [name for name in ("tournament", "finding_type", "measure", "label", "estimate", "low", "high",
                                 "wording", "passes_gate") if name in selected]
    table = selected[columns].head(MAX_ROWS).reset_index(drop=True)
    wording = table["wording"].dropna().astype(str).tolist() if "wording" in table else []
    return ensure_aggregated(Result(
        function="findings", args={"tournament": tournament}, title=f"What we can say ({tournament})",
        table=table, numbers={"tournament": tournament, "count": len(selected), "wording": wording},
        chart=None, sources=(source,), available=True,
        note=f"Showing the first {MAX_ROWS} approved findings." if len(selected) > MAX_ROWS else "",
    ))


def senior_check(cfg: Config) -> Result:
    """Return model-versus-base-rate results for each senior tournament and optional signifier checks."""
    needed = ("senior_backtest", "checks_senior")
    missing = tuple(name for name in needed if not table_exists(cfg, name))
    if missing:
        return _unavailable("senior_check", {}, missing)
    backtest = read_table(cfg, "senior_backtest")
    checks = read_table(cfg, "checks_senior")
    sources = list(needed)
    signifiers = read_table(cfg, "senior_signifiers") if table_exists(cfg, "senior_signifiers") else None
    if signifiers is not None:
        sources.append("senior_signifiers")
    records = []
    for tournament, rows in backtest.groupby("tournament", sort=True):
        beaten = pd.to_numeric(rows["model_beats_base_rates"], errors="coerce").fillna(0).eq(1)
        years = [int(year) for year in rows.loc[beaten, "year"]]
        tournament_checks = checks.loc[checks["check"].astype(str).str.contains(
            rf"\bsenior {re.escape(str(tournament))}\b", case=False, regex=True,
        )]
        record = {"tournament": tournament, "editions": len(rows), "beats_base_rates": int(beaten.sum()),
                  "years_beating_base_rates": ", ".join(map(str, sorted(years))),
                  "checks_passed": int(tournament_checks["passed"].fillna(False).astype(bool).sum()),
                  "checks_total": len(tournament_checks), "signifiers_tested": 0}
        if signifiers is not None and "tournament" in signifiers:
            tested = signifiers.loc[signifiers["tournament"].eq(tournament)]
            record["signifiers_tested"] = int(tested["measure"].nunique()) if "measure" in tested else len(tested)
        records.append(record)
    table = pd.DataFrame(records)
    return ensure_aggregated(Result(
        function="senior_check", args={}, title="Senior World Cup model checks", table=table.head(MAX_ROWS),
        numbers={"tournaments": records}, chart=None, sources=tuple(sources), available=True,
        note="senior_signifiers is not built in this build." if signifiers is None else "",
    ))


def _count(value: object) -> int | None:
    """Return a count from a numeric or comma-separated check value."""
    number = pd.to_numeric(str(value).replace(",", ""), errors="coerce")
    return int(number) if pd.notna(number) and np.isfinite(number) and number >= 0 else None


def data_coverage(cfg: Config) -> Result:
    """Return tournament match/event counts and men's tracking and physical coverage from checks."""
    check_tables = ("checks_statsbomb", "checks_match_folders", "checks_tracking")
    sources, records, missing = [], [], []
    found: set[tuple[str, str]] = set()
    for source in check_tables:
        if not table_exists(cfg, source):
            missing.append(source)
            continue
        sources.append(source)
        checks = read_table(cfg, source)
        if not {"check", "value"}.issubset(checks.columns):
            continue
        for row in checks.itertuples(index=False):
            name = str(row.check).casefold()
            tournament, metric = "men", ""
            if source == "checks_statsbomb":
                tournament = "women" if name.startswith("women:") else "men" if name.startswith("men:") else ""
                if name.endswith(": matches"):
                    metric = "matches"
                elif ": rows (events" in name or name.endswith(": events"):
                    metric = "events"
            elif source == "checks_match_folders" and "folders with a physical csv" in name:
                metric = "physical_matches"
            elif source == "checks_tracking" and "folders with a tracking zip" in name:
                metric = "tracking_matches"
            value = _count(row.value)
            if tournament and metric and value is not None:
                labels = {"matches": "Matches", "events": "StatsBomb events including shootouts",
                          "physical_matches": "Matches with FIFA physical files",
                          "tracking_matches": "Matches with broadcast tracking"}
                records.append({"tournament": tournament, "metric": metric, "label": labels[metric],
                                "value": value, "target": str(getattr(row, "target", ""))})
                found.add((tournament, metric))
    if table_exists(cfg, "matches"):
        matches = read_table(cfg, "matches")
        sources.append("matches")
        for tournament, rows in matches.groupby("tournament", sort=True):
            if (tournament, "matches") not in found:
                records.append({"tournament": tournament, "metric": "matches", "label": "Matches",
                                "value": int(rows["match_id"].nunique()), "target": ""})
    if not records:
        return _unavailable("data_coverage", {}, tuple(missing) or check_tables)
    table = pd.DataFrame(records).drop_duplicates(["tournament", "metric"], keep="first")
    summaries = []
    for tournament, rows in table.groupby("tournament", sort=True):
        summaries.append({"tournament": tournament, **dict(zip(rows["metric"], rows["value"]))})
    return ensure_aggregated(Result(
        function="data_coverage", args={}, title="Data coverage by tournament", table=table.head(MAX_ROWS),
        numbers={"tournaments": summaries}, chart=None, sources=tuple(dict.fromkeys(sources)), available=True,
        note=f"{', '.join(missing)} not built in this build." if missing else "",
    ))


def tactical_profile(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Return the team's tactical profile: its dominant, efficient and resourceful scores, its points above
    expectation, and how the teams with the same profile did."""
    from three_trees.charts.tactical import profile_map

    sources = ("tactical_profile_team", "tactical_profile_summary")
    missing = tuple(name for name in sources if not table_exists(cfg, name))
    if missing:
        return _unavailable("tactical_profile", {"team": team, "tournament": tournament}, missing)
    team = resolve_team(cfg, team, tournament)
    teams, summary = read_table(cfg, sources[0]), read_table(cfg, sources[1])
    rows = teams.loc[teams["tournament"].eq(tournament) & teams["team"].eq(team)]
    if rows.empty:
        return _unavailable("tactical_profile", {"team": team, "tournament": tournament}, sources,
                            note=f"{team} has no tactical profile in this build.")
    r = rows.iloc[0]
    same = summary.loc[summary["tournament"].eq(tournament) & summary["profile"].eq(r["profile"])].iloc[0]
    table = pd.DataFrame({"score": ["Dominant", "Efficient", "Resourceful"],
                          "standard_deviations": [r["dominant"], r["efficient"], r["resourceful"]]})
    numbers = {"team": team, "tournament": tournament, "profile": r["profile"],
               **{k: float(r[k]) for k in ("dominant", "efficient", "resourceful", "overperformance_per_match")},
               "profile_teams": int(same["n_teams"]), "profile_mean": float(same["overperformance_per_match"]),
               "profile_low": float(same["low"]), "profile_high": float(same["high"])}
    return ensure_aggregated(Result(
        function="tactical_profile", args={"team": team, "tournament": tournament},
        title=f"Tactical profile: {team} ({tournament})", table=table.round(2), numbers=numbers,
        chart=profile_map(teams, tournament, team), sources=sources, available=True, note="",
    ))


def profiles_that_win(cfg: Config, tournament: str = "men") -> Result:
    """Return how each tactical profile did against its Elo expectation, and the profile findings that pass
    the slide gate."""
    from three_trees.charts.tactical import profile_outcomes

    source = "tactical_profile_summary"
    if not table_exists(cfg, source):
        return _unavailable("profiles_that_win", {"tournament": tournament}, (source,))
    summary = read_table(cfg, source)
    rows = summary.loc[summary["tournament"].eq(tournament)].sort_values("overperformance_per_match",
                                                                         ascending=False)
    columns = ["profile", "n_teams", "overperformance_per_match", "low", "high", "share_above_expectation"]
    wording = []
    if table_exists(cfg, "finding_gate"):
        gate = read_table(cfg, "finding_gate")
        passing = gate.loc[gate["tournament"].eq(tournament) & gate["finding_type"].eq("tactical profile")
                           & gate["passes_gate"].fillna(False).astype(bool)]
        wording = passing["wording"].astype(str).tolist()
    best, worst = rows.iloc[0], rows.iloc[-1]
    numbers = {"tournament": tournament, "best": best["profile"], "best_mean": float(best["overperformance_per_match"]),
               "best_teams": int(best["n_teams"]), "worst": worst["profile"],
               "worst_mean": float(worst["overperformance_per_match"]), "worst_teams": int(worst["n_teams"]),
               "wording": wording}
    return ensure_aggregated(Result(
        function="profiles_that_win", args={"tournament": tournament},
        title=f"Which tactical profiles beat their strength ({tournament})", table=rows[columns].round(2),
        numbers=numbers, chart=profile_outcomes(summary, tournament), sources=(source, "finding_gate"),
        available=True, note="",
    ))


def _answer_tournament(tournament: str) -> str:
    """Return a supported tournament name or a plain query error."""
    if tournament not in ("men", "women"):
        raise QueryError("Choose the men's or women's tournament.")
    return tournament


def _coach_terms(text: str) -> str:
    """Explain statistical and football abbreviations in the evidence labels."""
    for term, explanation in (
        (r"\bxG\b", "xG (expected goals: chance quality)"),
        (r"\bxA\b", "xA (expected assists: chance quality created for a shot)"),
        (r"\bfield tilt\b", "field tilt (share of final-third passes and carries)"),
        (r"\bHOPS\b", "HOPS (estimated aerial-duel difficulty)"),
    ):
        text = re.sub(term, explanation, text, flags=re.IGNORECASE)
    return text


def _point_numbers(row: pd.Series, tournament: str) -> dict[str, object]:
    """Return the shared point verdict and the requested tournament's evidence in words."""
    point = str(row["point"])
    clear, not_clear = str(row[f"{tournament}_clear"]), str(row[f"{tournament}_not_clear"])
    tournament_label = "men's" if tournament == "men" else "women's"
    if not clear and not not_clear:
        evidence = f"no measures for the {tournament_label} tournament are built in this build"
    else:
        unclear_labels = re.sub(r"\s+\((?:higher|lower)\)", "", not_clear)
        evidence = (
            f"in the {tournament_label} tournament, a clear difference appears for "
            f"{_coach_terms(clear) if clear else 'none of the measured traits'}; "
            f"no clear difference for {_coach_terms(unclear_labels) if not_clear else 'the remaining traits (none)'}"
        )
    return {
        "point": point, "definition": POINT_DEFINITIONS[point], "verdict": str(row["verdict"]),
        "tournament": tournament, "clear": clear, "not_clear": not_clear,
        "sentence": f"{point}: {row['verdict']}; {evidence}.",
    }


def research_answer(cfg: Config, tournament: str = "men") -> Result:
    """Return the research question, three shared point verdicts and approved profile findings."""
    from three_trees.charts.profiles import profile_comparison_chart

    tournament = _answer_tournament(tournament)
    sources = ("profile_comparison", "finding_gate")
    args = {"tournament": tournament}
    if not table_exists(cfg, sources[0]):
        return _unavailable("research_answer", args, (sources[0],))
    comparison = read_table(cfg, sources[0])
    evidence = answer_model.point_evidence(comparison)
    evidence = evidence.loc[evidence["tournament"].eq(tournament)].reset_index(drop=True)
    if evidence.empty:
        return _unavailable("research_answer", args, (sources[0],),
                            f"Point evidence for the {tournament} tournament is not built in this build.")
    verdicts = answer_model.point_verdicts(comparison)
    points = [_point_numbers(row, tournament) for _, row in verdicts.iterrows()]
    wording = []
    gate_missing = not table_exists(cfg, sources[1])
    if not gate_missing:
        gate = read_table(cfg, sources[1])
        passing = gate.loc[
            gate["tournament"].eq(tournament) & gate["finding_type"].eq("tactical profile")
            & gate["passes_gate"].fillna(False).astype(bool)
        ]
        wording = passing["wording"].dropna().astype(str).tolist()
    selected = comparison.loc[
        comparison["tournament"].eq(tournament) & comparison["measure"].isin(evidence["measure"])
    ].copy()
    chart = profile_comparison_chart(selected, tournament)
    colour = "blue" if tournament == "men" else "pink"
    if tournament == "women":
        for trace in chart.data:
            if trace.marker.color == ACCENT:
                trace.marker.color = "#CC79A7"
                trace.error_x.color = "#CC79A7"
    caption = (
        "Each row is a trait: hollow grey dots compare teams that progressed with teams that exited; "
        f"filled {colour} dots compare teams above with teams below their strength expectation after "
        "accounting for strength. The horizontal axis shows differences in units of spread; lines show "
        "95% intervals and faded grey dots mark intervals crossing zero. Right means higher and left "
        "lower among teams above expectation; whether higher is better depends on the trait, while "
        "crossing zero means no clear difference."
    )
    status_note = "finding_gate is not built in this build." if gate_missing else ""
    note = caption + (f" {status_note}" if status_note else "")
    return ensure_aggregated(Result(
        function="research_answer", args=args, title=RESEARCH_QUESTION,
        table=evidence.head(MAX_ROWS),
        numbers={"question": RESEARCH_QUESTION, "tournament": tournament, "points": points,
                 "wording": wording, "chart_caption": caption, "status_note": status_note},
        chart=chart, sources=sources, available=True, note=note,
    ))


def point_evidence(cfg: Config, point: str, tournament: str = "men") -> Result:
    """Return one research point's evidence and its shared verdict for a tournament."""
    tournament = _answer_tournament(tournament)
    canonical = {name.casefold(): name for name in POINTS}.get(str(point).strip().casefold())
    if canonical is None:
        raise QueryError(f"Choose a point: {', '.join(POINTS)}.")
    args = {"point": canonical, "tournament": tournament}
    source = "profile_comparison"
    if not table_exists(cfg, source):
        return _unavailable("point_evidence", args, (source,))
    comparison = read_table(cfg, source)
    evidence = answer_model.point_evidence(comparison)
    evidence = evidence.loc[
        evidence["tournament"].eq(tournament) & evidence["point"].eq(canonical)
    ].reset_index(drop=True)
    if evidence.empty:
        return _unavailable("point_evidence", args, (source,),
                            f"{canonical} evidence for the {tournament} tournament is not built in this build.")
    verdicts = answer_model.point_verdicts(comparison)
    row = verdicts.loc[verdicts["point"].eq(canonical)].iloc[0]
    return ensure_aggregated(Result(
        function="point_evidence", args=args, title=f"{canonical}: evidence ({tournament})",
        table=evidence.head(MAX_ROWS), numbers=_point_numbers(row, tournament), chart=None,
        sources=(source,), available=True, note="",
    ))


def _check_coach_values(value: object) -> None:
    """Reject player-level fields nested inside a team scouting report."""
    if isinstance(value, dict):
        if any(str(key).casefold() in FORBIDDEN_COLUMNS for key in value):
            raise QueryError("Scouting reports must contain team-level evidence only.")
        for nested in value.values():
            _check_coach_values(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _check_coach_values(nested)


def how_to_play_against(cfg: Config, team: str, tournament: str = "men") -> Result:
    """Return the coach module's team scouting notes, profile and game plan."""
    tournament = _answer_tournament(tournament)
    args = {"team": team, "tournament": tournament}
    source = "coach"
    try:
        from three_trees.coach import scouting_report
    except ImportError:
        return _unavailable("how_to_play_against", args, (source,))
    report = scouting_report(cfg, team, tournament)
    if not isinstance(report, dict):
        raise QueryError("A scouting report must contain a team summary and game plan.")
    plan = report.get("game_plan", [])
    if not isinstance(plan, (list, tuple)) or any(not isinstance(item, str) for item in plan):
        raise QueryError("A scouting game plan must be a list of short sentences.")
    if any(not isinstance(report.get(name, ""), str) for name in ("profile", "summary")):
        raise QueryError("A scouting profile and summary must be plain text.")
    notes = report.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(note, dict) for note in notes):
        raise QueryError("Scouting notes must be a list of team-level observations.")
    columns = ["area", "observation", "recommendation", "evidence"]
    records = [{column: note.get(column, "") for column in columns} for note in notes]
    numbers = {
        "team": str(report.get("team", team)), "tournament": str(report.get("tournament", tournament)),
        "profile": report.get("profile", ""), "summary": report.get("summary", ""),
        "game_plan": list(plan[:3]),
    }
    _check_coach_values([records, numbers])
    if numbers["tournament"] != tournament:
        raise QueryError("The scouting report must describe the requested tournament.")
    return ensure_aggregated(Result(
        function="how_to_play_against", args={"team": numbers["team"], "tournament": tournament},
        title=f"How to play against {numbers['team']}", table=pd.DataFrame(records, columns=columns).head(MAX_ROWS),
        numbers=numbers, chart=None, sources=(source,), available=True, note="",
    ))


# ---------------------------------------------------------------- catalogue registry


CATALOGUE: dict[str, Callable[..., Result]] = {
    "list_measures": list_measures,
    "team_summary": team_summary,
    "rank_teams": rank_teams,
    "compare_teams": compare_teams,
    "explain_overperformance": explain_overperformance,
    "match_review": match_review,
    "xg_timeline": xg_timeline,
    "fifa_phases": fifa_phases,
    "team_shape": team_shape,
    "findings": findings,
    "tactical_profile": tactical_profile,
    "profiles_that_win": profiles_that_win,
    "research_answer": research_answer,
    "point_evidence": point_evidence,
    "how_to_play_against": how_to_play_against,
    "senior_check": senior_check,
    "data_coverage": data_coverage,
    "style_of": style_of,
    "teams_by_style": teams_by_style,
    "running_profile": running_profile,
    "zone_map": zone_map,
    "conversion_by_phase": conversion_by_phase,
    "efficiency_index": efficiency_index,
    "compare_progressing_vs_exiting": compare_progressing_vs_exiting,
}


def _build_tools_schema() -> list[dict[str, Any]]:
    """Build strict Claude tool definitions for every catalogue function."""
    reg = load_measures()
    all_measures = sorted(reg["measure"].unique().tolist())

    tools = [
        {
            "name": "list_measures",
            "description": "List all available registry metrics, their units, signifier categories and definitions for a tournament.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "tournament": {
                        "type": "string",
                        "enum": ["men", "women"],
                        "description": "Tournament edition (men or women).",
                    },
                },
                "required": ["tournament"],
                "additionalProperties": False,
            },
        },
        {
            "name": "team_summary",
            "description": "Get a team's pre-tournament senior Elo, points above expectation, luck band, tournament stage reached, and key metrics.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {
                        "type": "string",
                        "description": "Team name (e.g. 'Team A1').",
                    },
                    "tournament": {
                        "type": "string",
                        "enum": ["men", "women"],
                        "description": "Tournament edition.",
                    },
                },
                "required": ["team", "tournament"],
                "additionalProperties": False,
            },
        },
        {
            "name": "rank_teams",
            "description": "Rank teams by any performance or expectation metric (e.g. overperformance, territory_share, conversion, security).",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "measure": {
                        "type": "string",
                        "enum": all_measures,
                        "description": "Registry measure to rank by.",
                    },
                    "tournament": {
                        "type": "string",
                        "enum": ["men", "women"],
                        "description": "Tournament edition.",
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of teams to return (1-48, default 10).",
                    },
                    "order": {
                        "type": "string",
                        "enum": ["best", "worst", "highest", "lowest"],
                        "description": "Ranking direction: best (follows measure direction), worst, highest, or lowest.",
                    },
                },
                "required": ["measure", "tournament", "n", "order"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_teams",
            "description": "Compare two teams side-by-side across key tactical signifiers and tournament ranks.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team_a": {"type": "string", "description": "First team name."},
                    "team_b": {"type": "string", "description": "Second team name."},
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                    "measures": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of measure names to compare. If empty, uses the 6 signifiers.",
                    },
                },
                "required": ["team_a", "team_b", "tournament", "measures"],
                "additionalProperties": False,
            },
        },
        {
            "name": "explain_overperformance",
            "description": "Explain why a team over-performed or under-performed: luck band verdict, per-match actual vs expected points, and key tactical drivers.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name."},
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                },
                "required": ["team", "tournament"],
                "additionalProperties": False,
            },
        },
        {
            "name": "match_review",
            "description": "Review a specific match between two teams: score, outcome probabilities, and tactical dominance metrics.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "First team."},
                    "opponent": {"type": "string", "description": "Second team."},
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                },
                "required": ["team", "opponent", "tournament"],
                "additionalProperties": False,
            },
        },
        {
            "name": "style_of",
            "description": "Get a team's tactical style cluster classification and profile.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name."},
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                },
                "required": ["team", "tournament"],
                "additionalProperties": False,
            },
        },
        {
            "name": "teams_by_style",
            "description": "Find all teams assigned to a specific playing style cluster.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "style": {"type": "string", "description": "Style cluster label or numeric id."},
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                },
                "required": ["style", "tournament"],
                "additionalProperties": False,
            },
        },
        {
            "name": "running_profile",
            "description": "Physical running output (distance, high-speed running, sprints) by position group and possession state (men only).",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name."},
                    "tournament": {"type": "string", "enum": ["men"], "description": "Tournament edition (men only)."},
                    "position_group": {
                        "type": ["string", "null"],
                        "description": "Optional position group filter (Goalkeeper, Defender, Midfielder, Forward).",
                    },
                },
                "required": ["team", "tournament", "position_group"],
                "additionalProperties": False,
            },
        },
        {
            "name": "zone_map",
            "description": "Spatial pitch control and contests across 18 zones for a team over the tournament or in one match.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name."},
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                    "opponent": {
                        "type": ["string", "null"],
                        "description": "Optional opponent name for single-match zone view.",
                    },
                },
                "required": ["team", "tournament", "opponent"],
                "additionalProperties": False,
            },
        },
        {
            "name": "conversion_by_phase",
            "description": "Conversion and xG efficiency across tactical possession phases for a team.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name."},
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                },
                "required": ["team", "tournament"],
                "additionalProperties": False,
            },
        },
        {
            "name": "efficiency_index",
            "description": "Leaderboard of teams ranked by the composite efficiency index (conversion vs cost).",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                    "n": {"type": "integer", "description": "Number of teams to return (default 10)."},
                },
                "required": ["tournament", "n"],
                "additionalProperties": False,
            },
        },
        {
            "name": "compare_progressing_vs_exiting",
            "description": "Compare tactical signifier profiles of teams that progressed versus exited, before and after strength adjustment.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "tournament": {"type": "string", "enum": ["men", "women"], "description": "Tournament edition."},
                },
                "required": ["tournament"],
                "additionalProperties": False,
            },
        },
    ]
    definitions = (
        ("xg_timeline", "Show both teams' xG, goals and goal times for one match.",
         ("team", "opponent", "tournament")),
        ("fifa_phases", "Compare a men's team's FIFA phase clip-time shares with the tournament.",
         ("team", "tournament")),
        ("team_shape", "Compare a men's team's line height, length and width without the ball.",
         ("team", "tournament")),
        ("findings", "List only findings that pass the slide gate, in plain English.", ("tournament",)),
        ("tactical_profile", "Show a team's tactical profile (dominant, efficient, resourceful scores), its points "
         "above Elo expectation, and how teams with the same profile did.", ("team", "tournament")),
        ("profiles_that_win", "Show which tactical profiles beat their strength (points above Elo expectation per "
         "match by profile) and the profile findings that pass the slide gate.", ("tournament",)),
        ("research_answer", "Answer the research question with the three point verdicts and approved tactical "
         "profile findings, using the built evidence.", ("tournament",)),
        ("point_evidence", "Explain the evidence for Dominance, Resourcefulness or Efficiency in one tournament.",
         ("point", "tournament")),
        ("how_to_play_against", "Give a team scouting summary and a game plan from the coach report.",
         ("team", "tournament")),
        ("senior_check", "Summarise senior World Cup editions where the model beats base rates.", ()),
        ("data_coverage", "Show tournament match/event counts and men's physical/tracking coverage.", ()),
    )
    for name, description, parameters in definitions:
        properties = {}
        for parameter in parameters:
            properties[parameter] = {"type": "string", "description": parameter.replace("_", " ").capitalize()}
            if parameter == "tournament":
                properties[parameter]["enum"] = ["men"] if name in ("fifa_phases", "team_shape") else ["men", "women"]
            elif parameter == "point":
                properties[parameter]["enum"] = list(POINTS)
        tools.append({
            "name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": properties, "required": list(parameters),
                             "additionalProperties": False},
        })
    for tool in tools:
        tool["strict"] = tool["name"] not in NON_STRICT_TOOLS
    return tools


# The API accepts at most MAX_STRICT_TOOLS strict tools; these simple, rarely asked ones run without strict
# schema checking (their inputs are a tournament or nothing).
MAX_STRICT_TOOLS = 20
NON_STRICT_TOOLS = frozenset({"list_measures", "data_coverage", "senior_check", "efficiency_index"})
TOOLS = _build_tools_schema()


def run_function(cfg: Config, name: str, args: dict[str, Any]) -> Result:
    """Look up a catalogue function by name, validate, execute, and verify aggregation."""
    if name not in CATALOGUE:
        raise QueryError(f"Function {name!r} is not in the query catalogue.")

    func = CATALOGUE[name]
    clean_args = {k: v for k, v in args.items() if v is not None}
    result = func(cfg, **clean_args)
    return ensure_aggregated(result)
