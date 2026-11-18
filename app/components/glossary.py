"""Plain-English glossary of data and tactical terms for a football coach."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from three_trees.definitions import POINT_DEFINITIONS

TERMS: dict[str, str] = {
    "xG": (
        "Expected goals: the quality and scoring chance of each shot taken or faced, "
        "calculated from shot distance, angle, assist type, and defensive pressure."
    ),
    "field tilt": (
        "The share of territory in the attacking third (passes and carries in the opponent's final third "
        "divided by all final-third actions in the match); shows which team pushed play into the attacking end."
    ),
    "Elo": (
        "Senior World Football Elo rating the day before the tournament. It sets each team's baseline "
        "strength expectation, so tournament results reflect over-performance rather than pre-existing strength."
    ),
    "points above expectation": (
        "Tournament points won beyond what a team's senior Elo strength predicted (positive means the team "
        "beat its expectation; negative means it fell short)."
    ),
    "luck band": (
        "The range of tournament points a team could win through match-to-match luck alone (the 5th to 95th "
        "percentile across simulated tournaments). Results outside the band are genuine over- or under-performance."
    ),
    "rank correlation": (
        "Spearman's rank correlation (from -1.0 to +1.0): tests whether teams ranking higher on a tactical measure "
        "also rank higher in points above expectation."
    ),
    "95% interval": (
        "A statistical range from bootstrap resampling where we are 95% confident the true value lies. "
        "When the interval does not touch zero, the difference is statistically clear."
    ),
    "standard deviations": (
        "A standardised scale ('z-score') measuring how far above (+1, +2) or below (-1, -2) the tournament "
        "average (0) a team's score sits, making different measures comparable."
    ),
    "dominance": POINT_DEFINITIONS["Dominance"],
    "resourcefulness": POINT_DEFINITIONS["Resourcefulness"],
    "efficiency": POINT_DEFINITIONS["Efficiency"],
    "conversion": (
        "Chances created per unit of territory: turning final-third entries and field tilt into dangerous shots."
    ),
    "security": (
        "Goal protection: expected goals (xG) conceded per opponent entry into the team's defensive third. "
        "Lower xG conceded per entry means stronger security."
    ),
    "threat spread": (
        "How widely chance creation is distributed across players (top-2 players' share of xG + xA) and pitch zones, "
        "measuring whether the attack depends on one key player or is varied."
    ),
    "high-speed running": (
        "Distance covered running at speed (optical tracking, over 19.8 km/h), measured with and without the ball "
        "to quantify physical intensity and pressing work."
    ),
    "filled-in positions": (
        "Frames where optical tracking cameras lost a player's coordinates and the provider imputed them. "
        "Matches with high imputed shares have lower tracking certainty."
    ),
    "formation lines": (
        "The depth, compactness, and spacing of defensive, midfield, and forward lines detected from player "
        "tracking coordinates across match phases with and without the ball."
    ),
}


def glossary(terms: Sequence[str] | None = None) -> None:
    """Show plain-English definitions in an expander.

    If `terms` is specified, only matching terms are displayed.
    """
    with st.expander("What the words mean"):
        if terms is None:
            selected = TERMS
        else:
            requested = {t.strip().lower() for t in terms}
            selected = {
                k: v for k, v in TERMS.items()
                if k.lower() in requested
            }
        for term, definition in selected.items():
            st.markdown(f"- **{term}**: {definition}")
