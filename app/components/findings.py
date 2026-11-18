"""Rule-based, non-causal findings from team-level registry measures."""

from __future__ import annotations

import pandas as pd

from trivium.measures import available, measures


def build_findings(team: str, team_match: pd.DataFrame, tournament: str) -> list[str]:
    """Return up to three plain-English deviations from tournament averages."""
    required = {"tournament", "team"}
    if not required.issubset(team_match.columns):
        return []
    tournament_rows = team_match.loc[team_match["tournament"].eq(tournament)].copy()
    team_rows = tournament_rows.loc[tournament_rows["team"].eq(team)]
    if team_rows.empty or tournament_rows.empty:
        return []

    registry = available(tournament_rows, measures(tournament=tournament, table="team_match"))
    candidates = []
    for measure in registry.itertuples(index=False):
        if measure.higher_is_better is None or measure.column not in tournament_rows:
            continue
        team_means = pd.to_numeric(
            tournament_rows.groupby("team")[measure.column].mean(),
            errors="coerce",
        )
        team_values = pd.to_numeric(team_rows[measure.column], errors="coerce")
        average = float(team_means.mean())
        spread = float(team_means.std(ddof=0))
        value = float(team_values.mean())
        if not pd.notna(average) or not pd.notna(value) or spread <= 0:
            continue
        difference = (value - average) / spread
        rank_values = team_means.rank(ascending=not bool(measure.higher_is_better), method="min")
        rank = int(rank_values.loc[team])
        candidates.append(
            (
                abs(difference),
                measure.label,
                measure.unit,
                value,
                average,
                difference,
                rank,
                int(team_means.notna().sum()),
                bool(measure.higher_is_better),
            )
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    findings = []
    for _, label, unit, value, average, difference, rank, count, higher_is_better in candidates[:3]:
        direction = (
            "higher than the tournament average"
            if difference > 0
            else "lower than the tournament average"
            if difference < 0
            else "close to the tournament average"
        )
        merit = (
            "better"
            if difference != 0 and (difference > 0) == higher_is_better
            else "worse"
            if difference != 0
            else "neutral"
        )
        findings.append(
            f"{label}: {team} was {direction}, a {merit} direction "
            f"({_format(value, unit)} vs {_format(average, unit)}; rank {rank} of {count})."
        )
    return findings


def _format(value: float, unit: str) -> str:
    if unit == "share":
        return f"{value:.0%}"
    return f"{value:.2f} {unit}".strip()
