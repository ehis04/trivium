"""Offline question routing and templated answering using rapidfuzz and fixed sentence patterns.

Used when Claude API credentials are absent, an API call fails or times out,
a model refusal occurs, or the user disables Claude.
"""

from __future__ import annotations

import re
from typing import Any

import rapidfuzz.fuzz
import rapidfuzz.process

from trivium.config import Config
from trivium.definitions import POINTS, RESEARCH_QUESTION
from trivium.measures import load_measures
from trivium.query.functions import (
    Result,
    _get_tournament_teams,
    run_function,
)

# Common aliases mapping user vocabulary to registry measure names
MEASURE_ALIASES: dict[str, str] = {
    "non-penalty xg conceded": "np_xg_against",
    "np xg conceded": "np_xg_against",
    "xg conceded": "np_xg_against",
    "goals conceded": "np_xg_against",
    "chances conceded": "np_xg_against",
    "possession share": "possession_share",
    "possession": "possession_share",
    "field tilt": "territory_share",
    "territory share": "territory_share",
    "territory": "territory_share",
    "non-penalty xg": "np_xg",
    "non penalty xg": "np_xg",
    "np xg": "np_xg",
    "expected goals": "np_xg",
    "xg": "np_xg",
    "pressures per minute": "pressures_per_opp_possession_min",
    "pressing": "pressures_per_opp_possession_min",
    "pressure": "pressures_per_opp_possession_min",
    "high-speed running": "hsr_km",
    "high speed running": "hsr_km",
    "high-speed": "hsr_km",
    "high speed": "hsr_km",
    "hsr": "hsr_km",
    "distance": "distance_km",
    "total distance": "distance_km",
    "running": "distance_km",
    "conversion": "conversion",
    "security": "security",
    "directness": "directness",
    "overperformance": "overperformance",
    "over-performance": "overperformance",
    "dominance": "dominance",
    "chances share": "chances_share",
    "chances": "chances_share",
    "contests share": "contests_share",
    "contests": "contests_share",
}


def _detect_tournament(question: str) -> str:
    """Detect whether question refers to women's or men's tournament (default men)."""
    q_lower = question.lower()
    if any(w in q_lower for w in ["women", "woman", "girls", "female", "womens", "women's"]):
        return "women"
    return "men"


def _short_names() -> dict[str, str]:
    """Short or senior names a coach may type, mapped to the tournament's team names."""
    from trivium.teams import _aliases

    names = {elo: name for name, elo in _aliases().items()}
    names.update({"Ireland": "Republic of Ireland", "USA": "United States", "UAE": "United Arab Emirates",
                  "Ivory Coast": "Côte d'Ivoire", "North Korea": "Korea DPR", "South Korea": "Korea Republic",
                  "China": "China PR"})
    return names


def _extract_teams(cfg: Config, question: str, tournament: str) -> list[str]:
    """Find tournament teams mentioned in the question without false positives."""
    teams = _get_tournament_teams(cfg, tournament)
    if not teams:
        return []

    found: list[tuple[int, str]] = []

    # 1. Regex match for "Team <Letter><Number>" (e.g. Team A1, Team B2)
    for m in re.finditer(r"\bteam\s+[a-z]\d+\b", question, re.IGNORECASE):
        matched_str = m.group(0)
        # Find closest team in tournament
        match = rapidfuzz.process.extractOne(matched_str, teams, score_cutoff=80)
        if match is not None:
            found.append((m.start(), str(match[0])))

    # 2. Word boundary match for standard team names (e.g. Argentina, France, Japan)
    for t in teams:
        if not re.match(r"^Team\s+[A-Z]\d+$", t, re.IGNORECASE):
            pattern = r"\b" + re.escape(t) + r"\b"
            for m in re.finditer(pattern, question, re.IGNORECASE):
                found.append((m.start(), t))

    # 3. Common short names ("Ireland", "South Korea", "USA"), when the tournament has that team.
    for alias, team in _short_names().items():
        if team in teams:
            for m in re.finditer(r"\b" + re.escape(alias) + r"\b", question, re.IGNORECASE):
                found.append((m.start(), team))

    if found:
        found.sort(key=lambda x: x[0])
        seen = set()
        out = []
        for _, t in found:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    return []


def _extract_measure(question: str) -> str | None:
    """Identify which registry measure or alias is requested in the question."""
    q_lower = question.lower()

    # Direct alias lookup (longest keys first)
    for alias, m_name in sorted(MEASURE_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
            return m_name

    reg = load_measures()
    for _, row in reg.iterrows():
        m_name = str(row["measure"])
        label = str(row["label"]).lower()
        if m_name in q_lower or label in q_lower:
            return m_name

    return None


def route(
    cfg: Config,
    question: str,
    previous_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Route a natural language question to a catalogue function and its arguments."""
    q_lower = question.lower().strip()
    tourney = _detect_tournament(question)

    ctx = previous_context or {}
    prev_team = ctx.get("team")
    prev_opponent = ctx.get("opponent")
    prev_tourney = ctx.get("tournament", "men")
    prev_fn = ctx.get("function")

    # Inherit tournament from previous turn (or the sidebar) unless the question names one
    names_men = re.search(r"\b(men|men's|mens|boys|male)\b", q_lower) is not None
    if tourney == "men" and not names_men and prev_tourney:
        tourney = prev_tourney

    teams = _extract_teams(cfg, question, tourney)
    if not teams and prev_team:
        # Check if question refers back to previous team
        if any(w in q_lower for w in ["they", "their", "them", "this team", "and style", "style", "running",
                                     "zone", "why", "overperform", "shape", "phase", "line height"]):
            teams = [prev_team]

    measure = _extract_measure(question)

    # Follow-up: "what about the women?" or "and women?"
    if any(q_lower.startswith(p) for p in ["what about the women", "what about women", "and the women", "and women"]):
        if prev_fn:
            if prev_fn in ("senior_check", "data_coverage"):
                return prev_fn, {}
            if prev_fn == "research_answer":
                return prev_fn, {"tournament": "women"}
            if prev_fn == "point_evidence":
                point = ctx.get("point")
                if point:
                    return prev_fn, {"point": point, "tournament": "women"}
                return "research_answer", {"tournament": "women"}
            args: dict[str, Any] = {"tournament": "women"}
            if prev_team:
                args["team"] = prev_team
            if prev_fn in ("xg_timeline", "match_review") and prev_opponent:
                args["opponent"] = prev_opponent
            if prev_fn == "rank_teams":
                args["measure"] = measure or "overperformance"
                args["n"] = 10
                args["order"] = "best"
            return prev_fn, args

    # 1. Measures list
    if any(k in q_lower for k in ["what can i ask", "list measures", "available measures", "available metrics", "show measures"]):
        return "list_measures", {"tournament": tourney}

    if any(k in q_lower for k in ["data coverage", "how much data", "matches and events", "physical coverage",
                                  "tracking coverage", "which data is missing", "what data is available"]):
        return "data_coverage", {}

    if any(k in q_lower for k in ["senior check", "senior world cup", "senior backtest", "senior validation",
                                  "base rates", "professional game", "senior model"]):
        return "senior_check", {}

    if any(phrase in q_lower for phrase in ("how to play against", "how do we beat", "scouting", "game plan")):
        team = teams[0] if teams else prev_team
        if team:
            return "how_to_play_against", {"team": team, "tournament": tourney}
        return None

    if any(phrase in q_lower for phrase in ("research question", "the answer", "three points")) \
            or RESEARCH_QUESTION.casefold().rstrip("?") in q_lower:
        return "research_answer", {"tournament": tourney}

    points = [point for point in POINTS if re.search(r"\b" + re.escape(point.casefold()) + r"\b", q_lower)]
    existing_intent = any(phrase in q_lower for phrase in (
        "profile", "dominant", "efficiency index", "leaderboard", "rank", "compare", "top ", "best", "worst",
    ))
    if points and not existing_intent:
        if len(points) > 1:
            return "research_answer", {"tournament": tourney}
        return "point_evidence", {"point": points[0], "tournament": tourney}

    if any(k in q_lower for k in ["findings", "what can we say", "safe to say", "slide gate", "approved claims"]):
        return "findings", {"tournament": tourney}

    if any(k in q_lower for k in ["tactical profile", "profile", "dominant", "resourceful"]) \
            and not any(k in q_lower for k in ["running profile", "progressing", "exiting"]):
        if teams:
            return "tactical_profile", {"team": teams[0], "tournament": tourney}
        return "profiles_that_win", {"tournament": tourney}

    if any(k in q_lower for k in ["xg timeline", "xg over time", "chances over time", "goal times",
                                  "when did", "when were"]):
        opponent = teams[1] if len(teams) > 1 else prev_opponent
        if teams and opponent:
            return "xg_timeline", {"team": teams[0], "opponent": opponent, "tournament": tourney}

    if teams and (("fifa" in q_lower and "phase" in q_lower) or "phases of play" in q_lower):
        return "fifa_phases", {"team": teams[0], "tournament": tourney}

    if teams and any(k in q_lower for k in ["team shape", "line height", "team length", "team width",
                                           "shape without the ball", "without-ball shape"]):
        return "team_shape", {"team": teams[0], "tournament": tourney}

    # 2. Compare Progressing vs Exiting
    went_out = any(w in q_lower for w in ["exit", "went out", "go out", "knocked out", "eliminated"])
    if any(k in q_lower for k in ["progressing vs exiting", "progress vs exit", "advancing vs exiting", "went out vs",
                                  "survive adjustment"]) or (("progress" in q_lower or "advanc" in q_lower) and went_out):
        return "compare_progressing_vs_exiting", {"tournament": tourney}

    # 3. Efficiency index
    if any(k in q_lower for k in ["efficiency index", "most efficient", "conversion vs cost", "efficient teams"]):
        m_num = re.search(r"\b(top|first)?\s*(\d+)\b", q_lower)
        n = int(m_num.group(2)) if m_num else 10
        return "efficiency_index", {"tournament": tourney, "n": n}

    # 4. Conversion by phase
    if any(k in q_lower for k in ["phase", "possession phase"]) and teams:
        return "conversion_by_phase", {"team": teams[0], "tournament": tourney}

    # 5. Zone territory map
    if any(k in q_lower for k in ["zone", "where did", "where does", "where can", "heatmap", "heat map", "territory map", "pitch map", "control the pitch"]):
        if teams:
            opp = teams[1] if len(teams) > 1 else None
            return "zone_map", {"team": teams[0], "tournament": tourney, "opponent": opp}

    # 6. Running profile
    if any(k in q_lower for k in ["running", "distance", "sprint", "high-speed", "hsr"]):
        if teams and not any(k in q_lower for k in ["rank", "top", "best", "most", "worst", "lowest"]):
            # Check for position group
            pos_group = None
            for pg in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
                if pg.lower() in q_lower:
                    pos_group = pg
                    break
            return "running_profile", {"team": teams[0], "tournament": tourney, "position_group": pos_group}

    # 7. Style functions
    if "style" in q_lower or "tactical cluster" in q_lower:
        if teams:
            return "style_of", {"team": teams[0], "tournament": tourney}
        words = [w for w in re.findall(r"\w+", q_lower) if w not in {"style", "teams", "with", "of", "which", "what", "show", "list", "in", "the", "men", "women"}]
        style_query = " ".join(words) if words else "Direct Transition"
        return "teams_by_style", {"style": style_query, "tournament": tourney}

    # 8. Match review vs Compare teams (two teams)
    if len(teams) >= 2:
        if any(k in q_lower for k in ["match", "game", "played", "head to head", "h2h", "score", "result", "against each other"]):
            return "match_review", {"team": teams[0], "opponent": teams[1], "tournament": tourney}
        return "compare_teams", {
            "team_a": teams[0],
            "team_b": teams[1],
            "tournament": tourney,
            "measures": [measure] if measure else [],
        }

    # 9. Explain over-performance
    if any(k in q_lower for k in ["why did", "explain", "overperform", "over-perform", "beat expectation", "exceed strength", "luck band"]):
        if teams:
            return "explain_overperformance", {"team": teams[0], "tournament": tourney}

    # 10. Ranking teams
    if any(k in q_lower for k in ["rank", "top", "best", "most", "worst", "lowest", "highest", "least", "leader", "who beat", "which team", "which men", "which women"]):
        if "lowest" in q_lower:
            order = "lowest"
        elif "highest" in q_lower:
            order = "highest"
        elif any(k in q_lower for k in ["worst", "least"]):
            order = "worst"
        else:
            order = "best"

        m_num = re.search(r"\b(top|first)?\s*(\d+)\b", q_lower)
        n = int(m_num.group(2)) if m_num else 10
        m_to_rank = measure or "overperformance"
        return "rank_teams", {"measure": m_to_rank, "tournament": tourney, "n": n, "order": order}

    # 11. Team summary (single team with no specific intent)
    if len(teams) == 1:
        return "team_summary", {"team": teams[0], "tournament": tourney}

    return None


def template(result: Result) -> str:
    """Generate 2-3 plain-English sentences from Result numbers only."""
    if not result.available:
        src = result.sources[0] if result.sources else "the requested table"
        return (
            f"{src} is not built in this build. {result.note}"
        )

    nums = result.numbers
    fn = result.function

    if fn == "research_answer":
        sentences = [str(nums["question"]), *[str(point["sentence"]) for point in nums["points"]]]
        sentences.extend(str(wording) for wording in nums["wording"])
        if nums.get("status_note"):
            sentences.append(str(nums["status_note"]))
        return " ".join(sentences) + "\n\n" + str(nums["chart_caption"])

    if fn == "point_evidence":
        return str(nums["sentence"])

    if fn == "how_to_play_against":
        profile = f" ({nums['profile']})" if nums["profile"] else ""
        summary = str(nums["summary"]).rstrip(". ")
        plan = " ".join(str(item).rstrip(". ") + "." for item in nums["game_plan"] if str(item).strip())
        return f"Against {nums['team']}{profile}: {summary}." + (f" Game plan: {plan}" if plan else "")

    if fn == "xg_timeline":
        def goal_times(side: str) -> str:
            return ", ".join(nums[f"{side}_goal_minutes"]) or "no goals"

        return (
            f"{nums['team']} created {nums['team_xg']:.2f} xG and scored {nums['team_goals']} goals; "
            f"{nums['opponent']} created {nums['opponent_xg']:.2f} xG and scored {nums['opponent_goals']} goals. "
            f"Goal times: {nums['team']}: {goal_times('team')}; "
            f"{nums['opponent']}: {goal_times('opponent')}. "
            "xG adds up the chances of scoring from each shot; shootout kicks are excluded."
        )

    if fn in ("fifa_phases", "team_shape"):
        values = nums["measures"]
        if fn == "fifa_phases":
            text = "; ".join(
                f"{row['label']}: {row['team_mean']:.1%} versus {row['tournament_mean']:.1%}"
                for row in values
            )
            return (f"{nums['team']}'s FIFA phase shares against the tournament team mean: {text}. "
                    "These are shares of phase clip time; overlapping phases can cover the same second.")
        text = "; ".join(
            f"{row['label']}: {row['team_mean']:.1f} m versus {row['tournament_mean']:.1f} m"
            for row in values
        )
        return f"Without the ball, {nums['team']} against the tournament team mean: {text}."

    if fn == "tactical_profile":
        return (
            f"{nums['team']} was {nums['profile'].replace(' · ', ', ').lower()}: dominant {nums['dominant']:+.1f}, "
            f"efficient {nums['efficient']:+.1f}, resourceful {nums['resourceful']:+.1f} standard deviations from "
            f"the {nums['tournament']}'s average. It earned {nums['overperformance_per_match']:+.2f} points per match "
            f"against its Elo expectation. The {nums['profile_teams']} teams with this profile averaged "
            f"{nums['profile_mean']:+.2f} (95% interval {nums['profile_low']:+.2f} to {nums['profile_high']:+.2f})."
        )

    if fn == "profiles_that_win":
        text = (
            f"In the {nums['tournament']}'s tournament, {nums['best'].replace(' · ', ', ').lower()} teams did best "
            f"against their strength ({nums['best_mean']:+.2f} points per match above expectation, "
            f"{nums['best_teams']} teams); {nums['worst'].replace(' · ', ', ').lower()} teams did worst "
            f"({nums['worst_mean']:+.2f}, {nums['worst_teams']} teams)."
        )
        return " ".join([text, *nums["wording"]])

    if fn == "findings":
        if not nums["count"]:
            return f"No findings pass the slide gate for the {nums['tournament']}'s tournament in this build."
        return " ".join(nums["wording"])

    if fn == "senior_check":
        sentences = []
        for row in nums["tournaments"]:
            years = row["years_beating_base_rates"] or "none"
            sentences.append(
                f"In the senior {row['tournament']}'s World Cups, the model beats base rates in "
                f"{row['beats_base_rates']} of {row['editions']} editions ({years}); "
                f"{row['checks_passed']} of {row['checks_total']} checks pass, with "
                f"{row['signifiers_tested']} signifiers tested."
            )
        return " ".join(sentences) or "No senior editions are listed in this build."

    if fn == "data_coverage":
        sentences = []
        for row in nums["tournaments"]:
            counts = []
            for field, label in (("matches", "matches"), ("events", "events including shootouts"),
                                 ("physical_matches", "matches with FIFA physical files"),
                                 ("tracking_matches", "matches with broadcast tracking")):
                if field in row:
                    counts.append(f"{row[field]:,} {label}")
            sentences.append(f"{row['tournament'].capitalize()}: {', '.join(counts)}.")
        return " ".join(sentences) + (f" {result.note}" if result.note else "")

    if fn == "team_summary":
        return (
            f"{nums['team']} ({nums['tournament']}) entered the tournament with a senior Elo rating of {nums['elo']:.0f} "
            f"and finished at the {nums['actual_stage']} stage with {nums['actual_points']:.1f} actual points. "
            f"Against their expected points of {nums['expected_points']:.2f}, their over-performance was {nums['overperformance']:+.2f} points "
            f"(luck band [{nums['band_low']:+.2f}, {nums['band_high']:+.2f}]), placing their performance {nums['beyond_luck'] or 'within expected variation'}. "
            f"Progress vs strength expectation was {nums['progress_over_expectation']:+.2f} stages."
        )

    if fn == "rank_teams":
        unit_str = f" {nums['unit']}" if nums['unit'] else ""
        return (
            f"In the {nums['tournament']}'s tournament, {nums['top_team']} leads the rankings for {nums['label']} "
            f"with a top value of {nums['top_value']}{unit_str}. "
            f"The leaderboard displays the top {nums['n']} teams ordered by {nums['order']} performance. "
            f"Review the full ranking table and chart below."
        )

    if fn == "compare_teams":
        return (
            f"Comparison of {nums['team_a']} and {nums['team_b']} in the {nums['tournament']}'s tournament across key tactical signifiers. "
            f"Each team's metrics are evaluated alongside tournament-wide rankings. "
            f"See the side-by-side table and grouped bar chart below for details."
        )

    if fn == "explain_overperformance":
        return (
            f"{nums['team']} achieved a total over-performance of {nums['overperformance']:+.2f} points "
            f"({nums['actual_points']:.1f} actual vs {nums['expected_points']:.2f} expected across {nums['matches_count']} matches). "
            f"Their tournament campaign is classified as {nums['verdict']}, relative to a 95% luck band of [{nums['band_low']:+.2f}, {nums['band_high']:+.2f}]. "
            f"Per-match expected points and outcomes are broken down below."
        )

    if fn == "match_review":
        so_note = " following a penalty shootout" if nums.get("shootout") else ""
        return (
            f"In the {nums['stage']} match, {nums['home_team']} and {nums['away_team']} finished {nums['home_score']}-{nums['away_score']}{so_note}. "
            f"The expectation model predicted {nums['expected_points_team']:.2f} points for {nums['team']} (win probability {nums['p_win_team']:.1%}) "
            f"versus {nums['expected_points_opp']:.2f} points for {nums['opponent']} (win probability {nums['p_win_opp']:.1%}). "
            f"Tactical field tilt and chances are summarized below."
        )

    if fn == "running_profile":
        pg_str = f" ({', '.join(nums['position_groups'])})" if nums.get("position_groups") else ""
        return (
            f"{nums['team']} recorded a cumulative running volume of {nums['total_distance_km']:.1f} km, "
            f"with {nums['total_hsr_km']:.1f} km at high-speed intensity. "
            f"Physical outputs are aggregated by position group{pg_str} and possession state without player identifiers. "
            f"Refer to the breakdown below."
        )

    if fn == "zone_map":
        return (
            f"Spatial pitch analysis for {nums['team']} reveals an average territory share of {nums['mean_territory_share']:.1%} across 18 zones, "
            f"including {nums['attacking_third_territory']:.1%} in the attacking third. "
            f"The 6x3 heatmap below illustrates field tilt from defensive to attacking thirds."
        )

    if fn == "style_of":
        return (
            f"{nums['team']} is categorized into tactical style cluster '{nums['style_label']}' (ID {nums['style_id']}) "
            f"with a classification confidence of {nums['probability']:.1%}. "
            f"Tactical profile metrics are listed in the table below."
        )

    if fn == "teams_by_style":
        return (
            f"Found {nums['team_count']} teams in the {nums['tournament']}'s tournament exhibiting style '{nums['style_label']}'. "
            f"Teams include: {', '.join(nums['teams'][:5])}{' and others' if len(nums['teams']) > 5 else ''}. "
            f"Detailed over-performance records for this cluster are presented below."
        )

    if fn == "efficiency_index":
        return (
            f"In the {nums['tournament']}'s tournament, {nums['leader']} leads the efficiency index with a rating of {nums['top_index']:.2f}. "
            f"The index synthesizes conversion effectiveness against defensive work per point of dominance. "
            f"The top-performing teams are ranked below."
        )

    if fn == "compare_progressing_vs_exiting":
        def listed(key: str) -> str:
            return ", ".join(nums.get(key) or []) or "none"

        return (
            f"In the {nums['tournament']}'s tournament, {nums['total_measures']} traits were compared between teams "
            f"that won a knockout tie and teams that went out. Clear after accounting for Elo strength: "
            f"{listed('survives_adjustment')}; only clear after adjustment: {listed('only_after_adjustment')}. "
            f"Looked important only because strong teams have them: {listed('explained_by_strength')}."
        )

    if fn == "list_measures":
        return (
            f"The measures registry contains {nums['total_measures']} metrics available for the {nums['tournament']}'s tournament, "
            f"including {nums['signifier_count']} primary tournament signifiers. "
            f"All metrics, definitions, and measurement units are catalogued below."
        )

    return (
        f"Query executed successfully, returning {len(result.table)} rows from sources: {', '.join(result.sources)}. "
        f"Summary figures and visual data are provided in the table and chart below."
    )


def offline_answer(
    cfg: Config,
    question: str,
    error: str = "",
    previous_context: dict[str, Any] | None = None,
) -> Any:
    """Produce an offline answer using rapidfuzz keyword routing and sentence templates."""
    # Imported inside function to avoid circular import
    from trivium.query.llm import Answer

    route_res = route(cfg, question, previous_context=previous_context)
    if route_res is None:
        text = (
            "I could not identify the specific analytical question. "
            "You can ask about team summaries, rankings, comparisons, or match reviews. "
            "Here are five example questions you can try:\n\n"
            "1. Which men's teams beat their strength most?\n"
            "2. Compare Team A1 and Team B2\n"
            "3. Why did Team C3 over-perform?\n"
            "4. What is the running profile of Team A1?\n"
            "5. Show the zone territory map for Team A1"
        )
        return Answer(
            question=question,
            text=text,
            result=None,
            mode="offline",
            function=None,
            args={},
            error=error or "Unroutable question",
        )

    func_name, func_args = route_res
    try:
        result = run_function(cfg, func_name, func_args)
        text = template(result)
    except Exception as e:
        result = None
        text = f"Could not process query: {e}"

    return Answer(
        question=question,
        text=text,
        result=result,
        mode="offline",
        function=func_name,
        args=func_args,
        error=error,
    )
