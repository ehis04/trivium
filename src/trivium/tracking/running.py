"""Running per player-match: distance, high-speed running (> 20 km/h) and
sprinting (> 25 km/h) by possession state (with ball, without ball, out of
play), third and FIFA phase; team high-speed running in the 10 s before each
StatsBomb tackle, interception, pressure and ball recovery.
"""

from __future__ import annotations

import math
import unicodedata
from array import array
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from trivium.ingest.match_folders import pair_team_names, team_key
from trivium.teams import clean_team_name
from trivium.tracking.shape import SAMPLE_EVERY, team_shape
from trivium.tracking.stream import PERIODS, iter_frames, parse_clock, read_metadata

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

FPS = 25              # tracking frames per second
PITCH_LENGTH = 105.0  # metres
HSR_KMH = 20.0        # high-speed running: faster than 20 km/h
SPRINT_KMH = 25.0     # sprinting: faster than 25 km/h

# FIFA physical-file speed bands in km/h, each (low, high]: the 20-25 and 25+
# bands add up to high-speed running exactly. Used for the check against FIFA (Task 3.4).
FIFA_BANDS = {
    "dist_1_7_kmh_m": (1.0, 7.0),
    "dist_7_15_kmh_m": (7.0, 15.0),
    "dist_15_20_kmh_m": (15.0, 20.0),
    "dist_20_25_kmh_m": (20.0, 25.0),
    "dist_25_plus_kmh_m": (25.0, math.inf),
}

# The labels used in the output. Inside the pass they are stored as numbers
# (their position in these lists) to save memory.
STATES = ("with_ball", "without_ball", "out_of_play")
THIRDS = ("defensive", "middle", "attacking", "unknown")  # unknown = no x position that frame

# Columns that describe the player (no names), and the columns that are added up.
PLAYER_KEYS = ["tracking_match_id", "team", "person_id", "jersey_number", "category_name", "position"]
SUM_COLUMNS = ["frames", "imputed_frames", "speed_missing_frames", "distance_m", "hsr_m", "sprint_m", *FIFA_BANDS]

# Position groups: the only level player data is shown at. Team-sheet codes (G, D, M, F) or
# FIFA codes (GK, DF, MF, FW); a goalkeeper category always counts as Goalkeeper.
POSITION_GROUPS = {"G": "Goalkeeper", "D": "Defender", "M": "Midfielder", "F": "Forward"}


def position_group(position: object, category_name: object = None) -> str:
    """Goalkeeper / Defender / Midfielder / Forward from a position code or word; Unknown otherwise."""
    if str(category_name).strip().lower() == "goalkeeper":
        return "Goalkeeper"
    code = str(position or "").strip().upper()
    return POSITION_GROUPS.get(code[:1], "Unknown") if code and code != "NONE" else "Unknown"


def with_position_group(running: pd.DataFrame) -> pd.DataFrame:
    """running with a position_group column (derived when missing or empty)."""
    out = running.copy()
    if "position_group" not in out or out["position_group"].isna().any():
        out["position_group"] = [position_group(p, c) for p, c in zip(out["position"], out["category_name"])]
    return out


# ---------------------------------------------------------------------------
# Rules for one player in one frame
# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _name(name: str) -> str:
    """A team name for comparing inside one tracking file: "Switzerland U17", "Cote d'Ivoire" and
    "Côte d'Ivoire" are written differently in ball_possession and team_name in some matches."""
    plain = unicodedata.normalize("NFKD", clean_team_name(str(name))).encode("ascii", "ignore").decode("ascii")
    return " ".join(plain.casefold().split())


def possession_state(ball_possession: str, team: str, pitch_side: dict[str, str]) -> str:
    """with_ball when the team has the ball, without_ball when the other team has it,
    otherwise out_of_play ("out of play" or "none"). Names compared with _name."""
    if _name(ball_possession) == _name(team):
        return "with_ball"
    # pitch_side names both teams, so this means the other team has the ball.
    if _name(ball_possession) in {_name(side) for side in pitch_side.values()}:
        return "without_ball"
    return "out_of_play"


def attacking_x(x: float, team: str, pitch_side: dict[str, str]) -> float:
    """x in the team's attacking direction: the team named in pitch_side.left attacks left to right."""
    # The team attacking right to left is flipped, so for every team a bigger x is nearer the opponent's goal.
    # Names compared with _name, as for possession (spellings can differ inside one tracking file).
    return x if _name(pitch_side["left"]) == _name(team) else PITCH_LENGTH - x


def third(x: float | None, team: str, pitch_side: dict[str, str]) -> str:
    """Third of the pitch from the team's view: defensive x' < 35, middle 35-70, attacking > 70."""
    if x is None:
        return "unknown"
    x = attacking_x(x, team, pitch_side)
    if x < PITCH_LENGTH / 3:
        return "defensive"
    if x <= 2 * PITCH_LENGTH / 3:
        return "middle"
    return "attacking"


# ---------------------------------------------------------------------------
# One match
# ---------------------------------------------------------------------------

def running_player_match(zip_path: str | Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """One pass over a match's tracking frames: (running, ball) from match_pass."""
    out = match_pass(zip_path)
    return out["running"], out["ball"]


SECOND_KEYS = ["team", "position_group", "period", "second"]
SHAPE_ROW_KEYS = ["team", "period", "clock_s", "state"]


def match_pass(zip_path: str | Path) -> dict[str, object]:
    """One pass over a match's tracking frames.

    Returns a dict:
      running: one row per player x possession state x third, with frames, minutes,
        imputed and speed-missing frames, distance, high-speed running, sprinting and
        the FIFA speed bands (all distances in metres). No player names.
      ball: per period number, an array of rows (clock seconds in the period, ball x, ball y),
        in tracking metres; NaN where the ball position is missing.
      shape: one row per team per second (every 25th frame): team, period, clock_s, state and the
        shape columns of tracking.shape (outfield players only).
      seconds: one row per team x position group x period x whole second of the clock, with
        distance_m, hsr_m and sprint_m summed over that team's players (team level only).
    """
    # Player positions (G, D, M, F) come from the team sheets in metadata.json.
    meta = read_metadata(zip_path)
    positions = {p["ID"]: p["Position"] for side in ("HomeTeam", "AwayTeam") for p in meta[side]["Players"]}

    # Step 1: one pass over the frames, saving five small values per player per frame.
    # array() stores numbers compactly: about 3.3 million player-frames per match.
    person = array("q")      # person_id
    state_code = array("b")  # possession state, as a position in STATES
    third_code = array("b")  # third, as a position in THIRDS
    speed = array("d")       # speed in m/s (NaN when missing)
    imputed = array("b")     # 1 when the position was filled in by the provider, not seen
    frame_of = array("i")    # which frame each player-frame belongs to
    frame_period = array("b")  # per frame: period number
    frame_clock = array("d")   # per frame: clock seconds within the period
    players: dict[int, dict[str, object]] = {}  # team, shirt number etc., once per player
    ball_rows: dict[int, list[tuple[float, float, float]]] = {}  # per period: (clock, x, y)
    shape_rows: list[tuple] = []  # once a second per team: (team, period, clock, state, line, length, width)

    for n, frame in enumerate(iter_frames(zip_path)):
        side = frame["pitch_side"]  # which team attacks left to right in this frame
        possession = frame["ball_possession"]
        period, clock = PERIODS[frame["period"]], parse_clock(frame["time"])
        frame_period.append(period)
        frame_clock.append(clock)
        states = {}  # possession state per team, worked out once per frame
        outfield: dict[str, tuple[list[float], list[float]]] = {}  # team -> (x in attacking direction, y)
        for p in frame["players"]:
            team = p["team_name"]
            if team not in states:
                states[team] = STATES.index(possession_state(possession, team, side))
            pid = p["person_id"]
            # First time a player is seen: keep their details (never their name).
            if pid not in players:
                players[pid] = {
                    "team": clean_team_name(team),
                    "jersey_number": p["jersey_number"],
                    "category_name": p["category_name"],
                    "position": positions.get(pid),
                }
            person.append(pid)
            frame_of.append(n)
            state_code.append(states[team])
            third_code.append(THIRDS.index(third(p.get("x"), team, side)))
            s = p.get("speed")
            speed.append(math.nan if s is None else s)
            imputed.append(bool(p.get("is_imputed")))
            if n % SAMPLE_EVERY == 0 and p.get("category_name") == "player" and p.get("x") is not None \
                    and p.get("y") is not None:
                xs, ys = outfield.setdefault(team, ([], []))
                xs.append(attacking_x(p["x"], team, side))
                ys.append(p["y"])

        # Team shape once a second (outfield players, attacking direction).
        for team, (xs, ys) in outfield.items():
            shape_rows.append((clean_team_name(team), period, clock, STATES[states[team]], *team_shape(xs, ys)))

        # The ball's position each frame, kept for lining up with StatsBomb events (Task 3.6).
        ball = frame.get("ball") or {}
        bx, by = ball.get("x"), ball.get("y")
        ball_rows.setdefault(period, []).append(
            (clock, math.nan if bx is None else bx, math.nan if by is None else by)
        )

    # Step 2: work out distance and speed bands for every player-frame at once.
    kmh = np.frombuffer(speed, dtype=np.float64) * 3.6  # m/s -> km/h
    dist = np.nan_to_num(kmh / 3.6 / FPS)  # metres in the frame = speed / 25; missing speed adds nothing
    df = pd.DataFrame(
        {
            "person_id": np.frombuffer(person, dtype=np.int64),
            "state": np.frombuffer(state_code, dtype=np.int8),
            "third": np.frombuffer(third_code, dtype=np.int8),
            "frames": 1,
            "imputed_frames": np.frombuffer(imputed, dtype=np.int8).astype(np.int64),
            "speed_missing_frames": np.isnan(kmh).astype(np.int64),
            "distance_m": dist,
            # Distance counts towards a band only in frames where the speed is in that band.
            "hsr_m": np.where(kmh > HSR_KMH, dist, 0.0),
            "sprint_m": np.where(kmh > SPRINT_KMH, dist, 0.0),
            **{col: np.where((kmh > lo) & (kmh <= hi), dist, 0.0) for col, (lo, hi) in FIFA_BANDS.items()},
        }
    )

    # Step 3: add up per player, state and third, then turn the codes back into labels.
    running = df.groupby(["person_id", "state", "third"], as_index=False)[SUM_COLUMNS].sum()
    running["state"] = running["state"].map(dict(enumerate(STATES)))
    running["third"] = running["third"].map(dict(enumerate(THIRDS)))
    running["minutes"] = running["frames"] / FPS / 60

    # Step 4: add the player details and the match id, in a fixed column order.
    info = pd.DataFrame.from_dict(players, orient="index").rename_axis("person_id").reset_index()
    info.insert(0, "tracking_match_id", meta.get("MatchID"))
    running = with_position_group(info.merge(running, on="person_id"))
    running = running[[*PLAYER_KEYS, "position_group", "state", "third", "minutes", *SUM_COLUMNS]]

    # Ball track: one array per period, columns (clock seconds, x, y).
    ball_track = {period: np.array(rows, dtype=np.float64) for period, rows in ball_rows.items()}

    # Team totals per whole second of the clock, by position group (for FIFA phases and the
    # 10-second windows before defensive actions).
    frames = np.frombuffer(frame_of, dtype=np.int32)
    groups = {pid: position_group(i["position"], i["category_name"]) for pid, i in players.items()}
    per_second = pd.DataFrame({
        "team": df["person_id"].map({pid: i["team"] for pid, i in players.items()}),
        "position_group": df["person_id"].map(groups),
        "period": np.frombuffer(frame_period, dtype=np.int8)[frames],
        "second": np.floor(np.frombuffer(frame_clock, dtype=np.float64)[frames]).astype(np.int32),
        "distance_m": df["distance_m"], "hsr_m": df["hsr_m"], "sprint_m": df["sprint_m"],
    })
    seconds = per_second.groupby(SECOND_KEYS, as_index=False)[["distance_m", "hsr_m", "sprint_m"]].sum()

    shape = pd.DataFrame(shape_rows, columns=[*SHAPE_ROW_KEYS, "line_height_m", "team_length_m", "team_width_m"])
    return {"running": running, "ball": ball_track, "shape": shape, "seconds": seconds}


def player_totals(running: pd.DataFrame) -> pd.DataFrame:
    """Sum the long running table over state and third: one row per player."""
    # Used for the check against FIFA's per-player physical file (Task 3.4).
    return running.groupby(PLAYER_KEYS, as_index=False, dropna=False)[["minutes", *SUM_COLUMNS]].sum()


# ---------------------------------------------------------------------------
# Check against FIFA's physical file (Task 3.4)
# ---------------------------------------------------------------------------

# Measure -> (tracking column, FIFA columns added together, correlation at least,
# median ratio range, M1 values from the round 1 plan).
FIFA_MEASURES = {
    "distance": ("distance_m", ["total_distance_m"], 0.99, (0.98, 1.02), "1.000, 1.006"),
    "high-speed running": ("hsr_m", ["dist_20_25_kmh_m", "dist_25_plus_kmh_m"], 0.95, (0.90, 1.20), "0.997, about 1.05"),
    "sprinting": ("sprint_m", ["dist_25_plus_kmh_m"], 0.95, (0.90, 1.30), "0.986, about 1.10"),
}
TEAM_DISTANCE_TOLERANCE = 0.02  # test 1: tracking distance within 2% of FIFA
IMPUTED_SHARE_MAX = 0.15
# Matches with more of their positions filled in by the provider leave the tracking measures (models and
# team_match): M81 Ireland v Canada 70%, M95 39%.
EXCLUDE_FILLED_IN_SHARE = 0.30


def filled_in_share(running: pd.DataFrame) -> pd.DataFrame:
    """Per match: share of player-frames whose position the provider filled in (imputed_frames / frames)."""
    sums = running.groupby(["tournament", "match_id"], as_index=False)[["imputed_frames", "frames"]].sum()
    sums["filled_in_share"] = sums["imputed_frames"] / sums["frames"].where(sums["frames"] > 0)
    return sums[["tournament", "match_id", "filled_in_share"]]


def excluded_matches(running: pd.DataFrame | None) -> pd.DataFrame:
    """Matches whose filled-in share is above EXCLUDE_FILLED_IN_SHARE: tournament, match_id, filled_in_share."""
    if running is None or running.empty or "imputed_frames" not in running:
        return pd.DataFrame(columns=["tournament", "match_id", "filled_in_share"])
    shares = filled_in_share(running)
    return shares[shares["filled_in_share"] > EXCLUDE_FILLED_IN_SHARE].reset_index(drop=True)


def match_to_fifa(running: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    """Tracking totals per player joined to FIFA's physical rows by team + shirt number.

    Every FIFA row is kept; tracking columns end in _tracking, FIFA columns in _fifa,
    and matched is False where no tracked player has that team and shirt number.
    """
    # Team names compared ignoring case and accents ("COTE D'IVOIRE" = "Côte d'Ivoire"); a FIFA name
    # with no alias is paired with the match's other tracking team ("UNITED ARAB EMIRATES" = "UAE").
    tracking = player_totals(running).assign(key=lambda d: team_key(d["team"]))
    pairs = pair_team_names(sorted(physical["fifa_team_name"].unique()), sorted(tracking["team"].unique()))
    fifa = physical.assign(key=lambda d: team_key(d["fifa_team_name"].map(pairs).fillna(d["fifa_team_name"])))
    joined = fifa.merge(tracking, on=["key", "jersey_number"], how="left",
                        suffixes=("_fifa", "_tracking"), indicator=True)
    joined["matched"] = joined.pop("_merge") == "both"
    return joined.drop(columns="key")


def check_against_fifa(running: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    """Checks table (check, target, value, passed) comparing tracking with FIFA's physical file."""
    joined = match_to_fifa(running, physical)
    both = joined[joined["matched"]]
    rows = []

    def add(check: str, target: str, value: float, passed: bool) -> None:
        rows.append({"check": check, "target": target, "value": float(value), "passed": bool(passed)})

    # Every FIFA player row should find its tracked player.
    add("physical rows matched", f"{len(joined)} of {len(joined)}", len(both), len(both) == len(joined))

    # Per player: correlation and median ratio (tracking / FIFA) for each measure.
    for name, (track_col, fifa_cols, min_corr, (lo, hi), m1) in FIFA_MEASURES.items():
        track = both[track_col]
        fifa = both[[c if c in both else f"{c}_fifa" for c in fifa_cols]].sum(axis=1)
        corr = track.corr(fifa)
        ratio = (track / fifa.where(fifa > 0)).median()  # players with 0 FIFA metres left out of the ratio
        add(f"{name}: correlation", f">= {min_corr} (M1 {m1.split(', ')[0]})", corr, corr >= min_corr)
        add(f"{name}: median ratio tracking / FIFA", f"{lo} to {hi} (M1 {m1.split(', ')[1]})", ratio, lo <= ratio <= hi)

    # Team totals in km: test 1, tracking distance within 2% of FIFA for each team.
    teams = both.groupby("team")[["distance_m", "total_distance_m"]].sum() / 1000
    for team, r in teams.iterrows():
        within = abs(r["distance_m"] / r["total_distance_m"] - 1) <= TEAM_DISTANCE_TOLERANCE
        add(f"team distance km, tracking ({team})", f"within 2% of FIFA {r['total_distance_m']:.1f}",
            r["distance_m"], within)

    # Share of player-frames whose position was filled in by the provider.
    imputed = running["imputed_frames"].sum() / running["frames"].sum()
    add("imputed player-frames share", f"under {IMPUTED_SHARE_MAX} (M1 0.092)", imputed, imputed < IMPUTED_SHARE_MAX)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Team-level table for team_match (round 2, Task 3.2)
# ---------------------------------------------------------------------------

TEAM_MATCH_KEYS = ["tournament", "match_id", "team"]


def team_match_provider(running: pd.DataFrame, physical: pd.DataFrame) -> pd.DataFrame:
    """team_match_provider: one row per team per men's match, the running measures that join team_match.

    running: running_player_match (StatsBomb team names). physical: physical_player_match.
    FIFA physical columns are team sums of the FIFA file; *_tracking and the possession-state columns
    are team sums of the tracking; top3_hsr_share is the three highest players' share of the team's
    tracking high-speed running. A team with only one of the two sources has the other's columns empty.
    """
    fifa = physical.groupby(TEAM_MATCH_KEYS, as_index=False).agg(
        distance_km=("total_distance_m", "sum"),
        hsr_20_25=("dist_20_25_kmh_m", "sum"),
        sprint_km=("dist_25_plus_kmh_m", "sum"),
    )
    fifa["hsr_km"] = (fifa.pop("hsr_20_25") + fifa["sprint_km"]) / 1000
    fifa["distance_km"] /= 1000
    fifa["sprint_km"] /= 1000

    tracking = running.groupby(TEAM_MATCH_KEYS, as_index=False).agg(
        distance_km_tracking=("distance_m", "sum"), hsr_km_tracking=("hsr_m", "sum"))
    by_state = running.pivot_table(index=TEAM_MATCH_KEYS, columns="state", values="hsr_m", aggfunc="sum",
                                   fill_value=0.0)
    by_state = by_state.reindex(columns=list(STATES), fill_value=0.0).add_prefix("hsr_").add_suffix("_km")
    defensive = running[(running["state"] == "without_ball") & (running["third"] == "defensive")]
    defensive = defensive.groupby(TEAM_MATCH_KEYS)["hsr_m"].sum().rename("hsr_without_ball_defensive_third_km")
    players = running.groupby([*TEAM_MATCH_KEYS, "person_id"])["hsr_m"].sum()
    top3 = players.groupby(level=TEAM_MATCH_KEYS).apply(
        lambda s: s.nlargest(3).sum() / s.sum() if s.sum() > 0 else np.nan).rename("top3_hsr_share")

    tracking = tracking.join(by_state, on=TEAM_MATCH_KEYS).join(defensive, on=TEAM_MATCH_KEYS)
    tracking["hsr_without_ball_defensive_third_km"] = tracking["hsr_without_ball_defensive_third_km"].fillna(0.0)
    tracking = tracking.join(top3, on=TEAM_MATCH_KEYS)
    metres = [c for c in tracking.columns if c.endswith("_km") or c.endswith("_tracking")]
    tracking[metres] = tracking[metres] / 1000

    out = fifa.merge(tracking, on=TEAM_MATCH_KEYS, how="outer", validate="one_to_one")
    columns = [*TEAM_MATCH_KEYS, "distance_km", "hsr_km", "sprint_km", "distance_km_tracking", "hsr_km_tracking",
               "hsr_with_ball_km", "hsr_without_ball_km", "hsr_out_of_play_km",
               "hsr_without_ball_defensive_third_km", "top3_hsr_share"]
    return out[columns].sort_values(TEAM_MATCH_KEYS, ignore_index=True)
