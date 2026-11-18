"""Tournament, team and match selectors shared by dashboard pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.data import load

DEFAULT_TEAM = "Republic of Ireland"


def select_context() -> tuple[str, str | None, int | None]:
    """Return the shared tournament, team and match selection."""
    st.sidebar.header("Choose a team or match")
    st.sidebar.checkbox("Presenter mode", key="presenter_mode")
    tournament = st.sidebar.selectbox("Tournament", ["men", "women"], key="tournament")
    elo = load("elo_pre_tournament")
    if elo is None or elo.empty or not {"tournament", "team"}.issubset(elo.columns):
        st.sidebar.info("Team choices are not built yet.")
        return tournament, None, None

    teams = sorted(elo.loc[elo["tournament"].eq(tournament), "team"].dropna().astype(str).unique())
    if not teams:
        st.sidebar.info(f"No {tournament}'s teams are available in this build.")
        return tournament, None, None
    default = teams.index(DEFAULT_TEAM) if DEFAULT_TEAM in teams else 0
    selected_team = st.sidebar.selectbox("Team", teams, index=default, key="team")

    matches = load("matches")
    if matches is None:
        st.sidebar.info("Match choices are not built yet.")
        return tournament, selected_team, None
    team_matches = matches.loc[
        matches["tournament"].eq(tournament)
        & (matches["home_team"].eq(selected_team) | matches["away_team"].eq(selected_team))
    ].copy()
    if "match_date" in team_matches:
        team_matches = team_matches.sort_values(["match_date", "match_id"], kind="stable")
    if team_matches.empty:
        st.sidebar.info("This team has no matches in the current build.")
        return tournament, selected_team, None
    team_matches["_label"] = team_matches.apply(_match_label, axis=1)
    labels = team_matches["_label"].tolist()
    selected_label = st.sidebar.selectbox("Match", labels, key="match_label")
    match_id = int(team_matches.loc[team_matches["_label"].eq(selected_label), "match_id"].iloc[0])
    return tournament, selected_team, match_id


def _match_label(row: pd.Series) -> str:
    stage = str(row.get("competition_stage", "Match"))
    home_score = pd.to_numeric(pd.Series([row.get("home_score")]), errors="coerce").iloc[0]
    away_score = pd.to_numeric(pd.Series([row.get("away_score")]), errors="coerce").iloc[0]
    score = f"{home_score:.0f}-{away_score:.0f}" if pd.notna(home_score + away_score) else "v"
    return f"{stage}: {row['home_team']} {score} {row['away_team']}"
