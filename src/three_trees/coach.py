"""Team-level scouting reports for the coach view and query box."""

from __future__ import annotations

import numpy as np
import pandas as pd

from three_trees.config import Config
from three_trees.definitions import POINTS, PROFILE_SCORES
from three_trees.measures import measure
from three_trees.models.answer import point_evidence, point_verdicts
from three_trees.storage import read_table, table_exists

NOTABLE_Z = 0.5  # a trait is notable in roughly the top or bottom third of the tournament

TABLE_NAMES = (
    "team_tournament",
    "tactical_profile_team",
    "tactical_profile_summary",
    "profile_comparison",
    "finding_gate",
    "formation_listed",
    "formation_team_match",
)


def _one_row(table: pd.DataFrame | None, tournament: str, team: str) -> pd.Series | None:
    if table is None or not {"tournament", "team"}.issubset(table.columns):
        return None
    rows = table.loc[table["tournament"].eq(tournament) & table["team"].eq(team)]
    return rows.iloc[0] if not rows.empty else None


def _z_score(
    rows: pd.DataFrame,
    team: str,
    name: str,
    tournament: str,
) -> tuple[float, float, int] | None:
    definition = measure(name)
    if str(definition["tier"]) == "men" and tournament != "men":
        return None
    column = str(definition["column"])
    if column not in rows:
        return None
    values = pd.to_numeric(rows[column], errors="coerce")
    valid = values.notna()
    team_rows = rows.loc[rows["team"].eq(team) & valid]
    sample = values.loc[valid]
    if team_rows.empty or len(sample) < 2:
        return None
    spread = float(sample.std())
    if not np.isfinite(spread) or spread == 0:
        return None
    raw_z = float((pd.to_numeric(team_rows[column], errors="coerce").iloc[0] - sample.mean()) / spread)
    higher_is_better = definition["higher_is_better"]
    orientation = -1.0 if pd.notna(higher_is_better) and not bool(higher_is_better) else 1.0
    return raw_z * orientation, orientation, len(sample)


def _rank_evidence(
    z_score: float,
    n_teams: int,
    label: str,
) -> str:
    side = "top" if z_score >= NOTABLE_Z else "bottom"
    return f"Among the {side} third of the {n_teams} teams for {label}."


def _point_evidence_text(tables: dict[str, pd.DataFrame], name: str, tournament: str) -> str:
    evidence = tables.get("point_evidence")
    if evidence is None or not {"measure", "tournament", "verdict"}.issubset(evidence.columns):
        return ""
    rows = evidence.loc[evidence["measure"].eq(name) & evidence["tournament"].eq(tournament)]
    if rows.empty or pd.isna(rows.iloc[0]["verdict"]):
        return ""
    wording = {
        "survives adjustment": "The difference remained after accounting for senior strength.",
        "explained by strength": "The difference was explained by senior strength.",
        "reverses": "The difference changed direction after accounting for senior strength.",
        "only after adjustment": "The difference appeared only after accounting for senior strength.",
        "no clear difference": "There was no clear difference after accounting for senior strength.",
    }
    verdict = str(rows.iloc[0]["verdict"])
    return f" {wording.get(verdict, verdict)}"


def _gate_text(tables: dict[str, pd.DataFrame], name: str, tournament: str) -> str:
    gate = tables.get("finding_gate")
    if gate is None or not {"measure", "tournament", "passes_gate"}.issubset(gate.columns):
        return ""
    rows = gate.loc[gate["measure"].eq(name) & gate["tournament"].eq(tournament)]
    if rows.empty or pd.isna(rows.iloc[0]["passes_gate"]):
        return ""
    result = "passed" if bool(rows.iloc[0]["passes_gate"]) else "did not pass"
    return f" The pre-set evidence checks {result} for this measure."


def _note(
    *,
    area: str,
    observation: str,
    recommendation: str,
    evidence: str,
    rank: float,
) -> tuple[float, dict[str, str]]:
    return rank, {
        "area": area,
        "observation": observation,
        "recommendation": recommendation,
        "evidence": evidence,
    }


def _formation_notes(tables: dict[str, pd.DataFrame], tournament: str, team: str) -> list[tuple[float, dict[str, str]]]:
    listed = tables.get("formation_listed")
    detected = tables.get("formation_team_match")
    values: list[tuple[str, str]] = []

    def main_formation(table: pd.DataFrame | None, state: str | None = None) -> str | None:
        if table is None or not {"tournament", "team", "main_formation"}.issubset(table.columns):
            return None
        rows = table.loc[table["tournament"].eq(tournament) & table["team"].eq(team)]
        if state is not None and "state" in rows:
            rows = rows.loc[rows["state"].eq(state)]
        formations = rows["main_formation"].dropna().astype(str)
        if formations.empty:
            return None
        counts = formations.value_counts()
        return sorted(counts[counts.eq(counts.max())].index.astype(str))[0]

    listed_shape = main_formation(listed)
    defensive_shape = main_formation(detected, "without_ball") if tournament == "men" else None
    if listed_shape:
        values.append(("line up in a {} (team sheets)", listed_shape))
    if defensive_shape:
        values.append(("defend in a {} shape (tracking)", defensive_shape))
    if not values:
        return []

    shapes = [shape for _, shape in values]
    back_lines = [int(shape.split("-", 1)[0]) for shape in shapes if shape[:1].isdigit()]
    if not back_lines:
        return []
    primary = defensive_shape or listed_shape
    primary_back_line = int(primary.split("-", 1)[0]) if primary and primary[:1].isdigit() else back_lines[0]
    if primary_back_line in (3, 5):
        recommendation = "Use the space between their wing-back and wide centre-back, and switch play quickly."
    elif primary_back_line == 4:
        recommendation = "Overload their full-backs two against one."
    else:
        recommendation = "Check their shape early and adjust your attacking match-ups to the back line."
    if len(set(back_lines)) > 1:
        alternatives = [
            "use the wing-back channel if they defend with three or five"
            if line in (3, 5)
            else "overload the full-backs if they defend with four"
            for line in sorted(set(back_lines))
        ]
        recommendation = "Be ready to " + " or ".join(alternatives) + "."

    details = " and ".join(template.format(shape) for template, shape in values)
    note = _note(
        area="Shape",
        observation=f"They usually {details}.",
        recommendation=recommendation,
        evidence="Based on the most-used team formation in the available match records.",
        rank=0.0,
    )
    return [note]


def _profile_summary(
    table: pd.DataFrame | None,
    profile: str,
    tournament: str,
) -> str:
    if table is None or not {"tournament", "profile"}.issubset(table.columns):
        return f"No tournament outcome summary is available for the {profile} profile."
    rows = table.loc[table["tournament"].eq(tournament) & table["profile"].eq(profile)]
    if rows.empty:
        return f"No tournament outcome summary is available for the {profile} profile."
    row = rows.iloc[0]
    required = ("overperformance_per_match", "low", "high", "n_teams")
    if any(column not in row or pd.isna(row[column]) for column in required):
        return f"No complete tournament outcome summary is available for the {profile} profile."
    mean = float(row["overperformance_per_match"])
    clear = "excludes_zero" in row and pd.notna(row["excludes_zero"]) and bool(row["excludes_zero"])
    more = "more" if mean > 0 else "fewer"
    caution = "" if clear else "; the result is uncertain, so treat it as a tendency"
    return (
        f"Teams with this profile won on average {abs(mean):.2f} points per match {more} than their strength "
        f"predicted ({int(row['n_teams'])} teams{caution})."
    )


def scouting_report_from_tables(
    tables: dict[str, pd.DataFrame],
    team: str,
    tournament: str,
) -> dict:
    """Build a team-level scouting report from finished tables, without reading files."""
    tournament_table = tables.get("team_tournament")
    if tournament_table is None or not {"tournament", "team"}.issubset(tournament_table.columns):
        raise ValueError("The team_tournament table with tournament and team columns is required.")
    rows = tournament_table.loc[tournament_table["tournament"].eq(tournament)]
    selected = rows.loc[rows["team"].eq(team)]
    if selected.empty:
        raise ValueError(f"No {tournament} team-tournament row is available for {team!r}.")

    report_tables = dict(tables)
    comparison = report_tables.get("profile_comparison")
    if comparison is not None and not comparison.empty:
        report_tables["point_evidence"] = point_evidence(comparison)
        report_tables["point_verdicts"] = point_verdicts(comparison)

    profile_row = _one_row(report_tables.get("tactical_profile_team"), tournament, team)
    if profile_row is None:
        profile = "Profile not available"
    else:
        profile = str(profile_row.get("profile", "")).strip()
        if not profile or profile.lower() == "nan":
            labels = []
            for axis, (_, high_label, low_label) in PROFILE_SCORES.items():
                value = pd.to_numeric(pd.Series([profile_row.get(axis)]), errors="coerce").iloc[0]
                if pd.notna(value):
                    labels.append(high_label if value >= 0 else low_label)
            profile = " · ".join(labels) if labels else "Profile not available"

    notes: list[tuple[float, dict[str, str]]] = []

    def add_measure_note(
        name: str,
        *,
        area: str,
        high_observation: str,
        high_recommendation: str,
        low_observation: str | None = None,
        low_recommendation: str | None = None,
    ) -> None:
        result = _z_score(rows, team, name, tournament)
        if result is None:
            return
        oriented_z, orientation, n_teams = result
        if oriented_z >= NOTABLE_Z:
            notable_side = "high"
        elif oriented_z <= -NOTABLE_Z:
            notable_side = "low"
        else:
            return
        raw_side = notable_side if orientation > 0 else ("low" if notable_side == "high" else "high")
        if raw_side == "high":
            observation, recommendation = high_observation, high_recommendation
        else:
            if low_observation is None or low_recommendation is None:
                return
            observation, recommendation = low_observation, low_recommendation
        definition = measure(name)
        evidence = _rank_evidence(
            oriented_z,
            n_teams,
            str(definition["label"]),
        )
        evidence += _point_evidence_text(report_tables, name, tournament)
        evidence += _gate_text(report_tables, name, tournament)
        notes.append(_note(
            area=area,
            observation=observation,
            recommendation=recommendation,
            evidence=evidence,
            rank=abs(oriented_z),
        ))

    add_measure_note(
        "security",
        area="Chances",
        high_observation="They concede good chances when opponents enter their defensive third.",
        high_recommendation="Get into their defensive third often, even with simple entries.",
        low_observation="They protect the box well.",
        low_recommendation="Be patient and only shoot from good positions.",
    )
    add_measure_note(
        "chances_share",
        area="Control",
        high_observation="They win a large share of the match's chances.",
        high_recommendation="Cut off their supply and stay compact around your box.",
        low_observation="They create a smaller share of the match's chances.",
        low_recommendation="Be brave on the ball.",
    )
    add_measure_note(
        "territory_share",
        area="Control",
        high_observation="They take a large share of territory in the final third.",
        high_recommendation="Plan to beat their press and keep an outlet behind it.",
        low_observation="They take a smaller share of territory in the final third.",
        low_recommendation="Push your line up and squeeze the space.",
    )
    add_measure_note(
        "contests_share",
        area="Control",
        high_observation="They win a large share of contests.",
        high_recommendation="Avoid turning the match into a battle of duels and second balls.",
        low_observation="They win a smaller share of contests.",
        low_recommendation="Compete physically and play for the second ball.",
    )
    add_measure_note(
        "conversion",
        area="Chances",
        high_observation="They turn their territory into chances more efficiently.",
        high_recommendation="Stop the final pass and defend against cut-backs.",
        low_observation="They get fewer chances from the territory they win.",
        low_recommendation="Let them have territory, then defend the box.",
    )
    add_measure_note(
        "defensive_work_per_dominance",
        area="Spending and cost",
        high_observation="Their control takes a lot of defensive work.",
        high_recommendation="Keep the ball, move it quickly and make them chase; expect them to tire late.",
        low_observation="They control matches with less defensive work.",
        low_recommendation="Press them early.",
    )
    add_measure_note(
        "top2_xg_xa_share",
        area="Chances",
        high_observation="A large share of their expected goals and assists comes from only a few contributors.",
        high_recommendation="Crowd the areas their main creators work in.",
    )
    add_measure_note(
        "chance_zone_spread",
        area="Chances",
        high_observation="They spread chance creation across more areas of the pitch.",
        high_recommendation="Stay connected across the pitch so they cannot switch into open space.",
        low_observation="Their chance creation is concentrated in a few pitch areas.",
        low_recommendation="Protect the few areas they create chances from.",
    )
    add_measure_note(
        "pressures_per_opp_possession_min",
        area="Spending and cost",
        high_observation="They press often while the opponent has the ball.",
        high_recommendation="Prepare your build-up to beat the first press; the goalkeeper and centre-backs need a long option.",
    )
    add_measure_note(
        "possession_share",
        area="Spending and cost",
        high_observation="They keep the ball for a large share of the match.",
        high_recommendation="Defend compactly and counter.",
        low_observation="They have less of the ball.",
        low_recommendation="Be patient and use the width.",
    )
    add_measure_note(
        "directness",
        area="Spending and cost",
        high_observation="They play forward quickly and directly.",
        high_recommendation="Win the first header and the second ball.",
    )

    profile_resourceful = None
    if profile_row is not None and "resourceful" in profile_row:
        profile_resourceful = pd.to_numeric(pd.Series([profile_row["resourceful"]]), errors="coerce").iloc[0]
    if profile_resourceful is not None and pd.notna(profile_resourceful):
        score = float(profile_resourceful)
        if score <= -NOTABLE_Z:
            notes.append(_note(
                area="Spending and cost",
                observation="Their overall resourcefulness score is among the lower quarter of teams.",
                recommendation="Keep the ball, move it quickly and make them chase; expect them to tire late.",
                evidence="The resourcefulness score is oriented using the registry so higher scores mean better resourcefulness.",
                rank=abs(score),
            ))
        elif score >= NOTABLE_Z:
            notes.append(_note(
                area="Spending and cost",
                observation="Their overall resourcefulness score is among the top quarter of teams.",
                recommendation="They control games cheaply; press them early.",
                evidence="The resourcefulness score is oriented using the registry so higher scores mean better resourcefulness.",
                rank=abs(score),
            ))

    phases = (
        ("fifa_block_share", "Their FIFA phase profile includes a high share of block play.",
         "Be patient, use the width, and look for crosses and cut-backs."),
        ("fifa_press_share", "Their FIFA phase profile includes a high share of pressing.",
         "Go long early to get beyond their press."),
        ("fifa_counter_attack_share", "Their FIFA phase profile includes a high share of counter-attacks.",
         "Keep two defenders and a midfielder behind the ball."),
    )
    for name, observation, recommendation in phases:
        result = _z_score(rows, team, name, tournament)
        if result is None or result[0] < NOTABLE_Z:
            continue
        z_score, _, n_teams = result
        definition = measure(name)
        evidence = _rank_evidence(z_score, n_teams, str(definition["label"]))
        evidence += _point_evidence_text(report_tables, name, tournament)
        evidence += _gate_text(report_tables, name, tournament)
        notes.append(_note(
            area="Phases of play",
            observation=observation,
            recommendation=recommendation,
            evidence=evidence,
            rank=abs(z_score),
        ))

    notes.extend(_formation_notes(report_tables, tournament, team))
    ordered = sorted(enumerate(notes), key=lambda item: (-item[1][0], item[0]))
    game_plan = []
    for _, (_, note) in ordered:
        recommendation = note["recommendation"]
        if recommendation not in game_plan:
            game_plan.append(recommendation)
        if len(game_plan) == 3:
            break

    profile_words = profile.replace(" · ", ", ").lower()
    summary = (f"{team} are {profile_words}. "
               + _profile_summary(report_tables.get("tactical_profile_summary"), profile, tournament))
    return {
        "team": team,
        "tournament": tournament,
        "profile": profile,
        "summary": summary,
        "game_plan": game_plan,
        "notes": [note for _, note in notes],
    }


def scouting_report(cfg: Config, team: str, tournament: str) -> dict:
    """Read available build tables and prepare one team's scouting report."""
    tables = {name: read_table(cfg, name) for name in TABLE_NAMES if table_exists(cfg, name)}
    return scouting_report_from_tables(tables, team, tournament)
