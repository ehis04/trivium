"""The research question and the project's three points (dominance, resourcefulness, efficiency) in plain
words, and which point and part every registry measure belongs to. Pages, chart captions, the coach view and
the query box read their wording from here; results always come from the build tables, never from this file.
"""

from __future__ import annotations

RESEARCH_QUESTION = (
    "Can we define dominant, resourceful and efficient tactical profiles that carry teams further in a "
    "tournament than their strength predicts?"
)
STRENGTH = (
    "Strength is the team's senior World Football Elo the day before the tournament. Every result is measured "
    "against it: points above expectation are the points a team won beyond what its strength predicted."
)
POINTS = ("Dominance", "Resourcefulness", "Efficiency")
POINT_DEFINITIONS = {
    "Dominance": (
        "How much of the match the team controlled: its share of the contests won (duels, 50/50s, second balls), "
        "of the territory (passes and carries in the opponent's final third, called field tilt) and of the "
        "chances (non-penalty expected goals, xG)."
    ),
    "Resourcefulness": (
        "What the team spent to get that control, and how well it used it: running (men: distance and high-speed "
        "running), pressing and defensive actions, time on the ball, and the squad's minutes and rotation; and "
        "the cost of each point of dominance (defensive work, and for the men high-speed running without the "
        "ball, per point of dominance)."
    ),
    "Efficiency": (
        "What the team got from its control: chances made for its territory (conversion), chances conceded each "
        "time the opponent got into its defensive third (security), and how widely its chance creation was "
        "spread across players and pitch zones."
    ),
}
PROFILE_SCORES = {
    "dominant": ("Dominance", "Dominant", "Less dominant"),
    "resourceful": ("Resourcefulness", "Resourceful", "Costly"),
    "efficient": ("Efficiency", "Efficient", "Less efficient"),
}
# measure -> (point, part); parts keep spending apart from the cost of each point of dominance.
MEASURE_POINTS = {
    "dominance": ("Dominance", "Overall control"),
    "contests_share": ("Dominance", "Contests"),
    "duels_share": ("Dominance", "Contests"),
    "aerials_share": ("Dominance", "Contests"),
    "fifty_fifties_share": ("Dominance", "Contests"),
    "second_balls_share": ("Dominance", "Contests"),
    "territory_share": ("Dominance", "Territory"),
    "final_third_actions": ("Dominance", "Territory"),
    "line_height_m": ("Dominance", "Territory"),
    "chances_share": ("Dominance", "Chances"),
    "pressures": ("Resourcefulness", "Spent"),
    "pressures_per_opp_possession_min": ("Resourcefulness", "Spent"),
    "defensive_actions": ("Resourcefulness", "Spent"),
    "possession_share": ("Resourcefulness", "Spent"),
    "possession_minutes": ("Resourcefulness", "Spent"),
    "distance_km": ("Resourcefulness", "Spent"),
    "hsr_km": ("Resourcefulness", "Spent"),
    "sprint_km": ("Resourcefulness", "Spent"),
    "hsr_without_ball_km": ("Resourcefulness", "Spent"),
    "players_used": ("Resourcefulness", "Squad and minutes"),
    "minutes_outside_top11_share": ("Resourcefulness", "Squad and minutes"),
    "starters_changed": ("Resourcefulness", "Squad and minutes"),
    "top3_hsr_share": ("Resourcefulness", "Squad and minutes"),
    "defensive_work_per_dominance": ("Resourcefulness", "Cost per point of dominance"),
    "hsr_without_ball_per_dominance": ("Resourcefulness", "Cost per point of dominance"),
    "conversion": ("Efficiency", "Turning control into chances"),
    "security": ("Efficiency", "Protecting the goal"),
    "top2_xg_xa_share": ("Efficiency", "Spread of threat"),
    "chance_zone_spread": ("Efficiency", "Spread of threat"),
}


def point_of(measure: str) -> str | None:
    """The point (Dominance, Resourcefulness, Efficiency) a measure belongs to, or None."""
    return MEASURE_POINTS.get(measure, (None, None))[0]


def part_of(measure: str) -> str | None:
    """The part of its point a measure belongs to (e.g. "Spent", "Cost per point of dominance"), or None."""
    return MEASURE_POINTS.get(measure, (None, None))[1]
