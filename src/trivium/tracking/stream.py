"""Tracking stage (men): one match at a time, streamed from the broadcast
tracking zip and never unzipped to disk. Each line has b"NaN" replaced with
b"null" and is parsed with orjson. Frames feed sync, running and shape.
Matches already processed are skipped.

Output tables: running_player_match, team_match_provider, sync_offsets, shape_team_second,
phase_clips, running_fifa_phase, event_running_windows, player_lookup,
provider_check_team_match, checks_tracking.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import orjson
import pandas as pd

from trivium.config import Config
from trivium.storage import read_table, table_exists, table_path, write_table

# The two files read from inside each tracking zip.
TRACKING_MEMBER = "tracking_data.jsonl"  # one frame per line, 25 frames a second
METADATA_MEMBER = "metadata.json"        # match info, frame rate, pitch size, team sheets

# Tracking period names -> period numbers (StatsBomb numbering).
PERIODS = {"half_1": 1, "half_2": 2, "extra_time_1": 3, "extra_time_2": 4}


def _member(zf: zipfile.ZipFile, name: str) -> str:
    """Full member path for a file name, whether at the zip root or inside a folder."""
    # Look through the zip's list of contents for a file with this name.
    for member in zf.namelist():
        if PurePosixPath(member).name == name:
            return member
    raise KeyError(f"{name} not found in {zf.filename}")


def _loads(raw: bytes) -> Any:
    """Parse one piece of JSON text from the tracking zip."""
    # The tracking files write missing values as bare NaN, which is not valid JSON.
    # Swap it for null so it parses, and comes out as None in Python.
    return orjson.loads(raw.replace(b"NaN", b"null"))


def iter_frames(zip_path: str | Path, member: str = TRACKING_MEMBER) -> Iterator[dict[str, Any]]:
    """Frames from a JSONL member of the tracking zip, one dict per line, read line by line."""
    # The file is read straight out of the zip, one line at a time, so the
    # 1.2 GB file is never unzipped to disk or held in memory all at once.
    with zipfile.ZipFile(zip_path) as zf, zf.open(_member(zf, member)) as lines:
        for line in lines:
            if line.strip():  # skip blank lines
                yield _loads(line)


def read_metadata(zip_path: str | Path, member: str = METADATA_MEMBER) -> dict[str, Any]:
    """metadata.json from the tracking zip."""
    # Small file, so it is read in one go.
    with zipfile.ZipFile(zip_path) as zf:
        return _loads(zf.read(_member(zf, member)))


def parse_clock(clock: str) -> float:
    """Clock within the period, "MM:SS.ffffff", to seconds."""
    # e.g. "12:30.500000" -> 12 * 60 + 30.5 = 750.5 seconds. The clock restarts at 00:00 each period.
    minutes, seconds = clock.split(":")
    return int(minutes) * 60 + float(seconds)


def run(cfg: Config) -> None:
    """Tracking stage: every match folder, one match at a time.

    Per match (keyed by StatsBomb match_id):
      tracking/<match_id>/running  running per player x state x third
      tracking/<match_id>/ball     ball position per frame, for the sync step
      tracking/<match_id>/sync     time offset per period (once the StatsBomb events table exists)
    Each is skipped when it already exists. Afterwards all matches are joined into
    running_player_match and sync_offsets, and checks_tracking is written and printed.
    """
    # Imported here because running.py and match_folders.py import this module.
    from trivium.ingest.match_folders import match_folders, men_matches, pair_team_names, read_physical
    from trivium.tracking.running import check_against_fifa, match_pass, team_match_provider, with_position_group
    from trivium.tracking.outofplay import fifa_ball_alive, out_of_play_shares
    from trivium.tracking.provider import provider_check, provider_summary, statsbomb_team_match
    from trivium.tracking.shape import shape_means
    from trivium.tracking.sync import SEARCH_LIMIT_S, SYNC_COLUMNS, left_teams, sync_match

    folders = match_folders(cfg)
    processed, skipped = [], []
    checks = []

    def add(check: str, target: str, value: float, passed: bool | None) -> None:
        # passed None = an information row (no pass or fail), as the checks schema asks.
        checks.append({"check": check, "target": target, "value": float(value),
                       "passed": None if passed is None else bool(passed)})

    # Step 1: each match with a tracking zip and a match_id, one at a time.
    ready = folders[folders["matched"] & folders["tracking_zip"].notna()]
    for f in ready.itertuples():
        names = {part: f"tracking/{f.match_id}/{part}" for part in PER_MATCH}
        if all(table_exists(cfg, name) for name in names.values()):
            skipped.append(f.folder)
            continue
        print(f"[tracking] {f.folder} (match {f.match_id})", flush=True)
        out = match_pass(f.tracking_zip)
        running = out["running"]
        running.insert(0, "match_id", int(f.match_id))
        running.insert(0, "tournament", "men")
        write_table(cfg, names["running"], running)
        write_table(cfg, names["ball"], ball_table(out["ball"]))
        write_table(cfg, names["shape"], out["shape"])
        write_table(cfg, names["seconds"], out["seconds"])
        processed.append(f.folder)

    # Step 2: sync offsets per match, from the saved ball table and the StatsBomb events.
    # Placeholder until H1: without Person 1's events table the sync is skipped (and checked as not done).
    events = sync_events(cfg)
    if events is not None:
        statsbomb = {int(r.match_id): [r.home_team, r.away_team] for r in men_matches(cfg).itertuples()}
        for f in ready.itertuples():
            sync_name = f"tracking/{f.match_id}/sync"
            if table_exists(cfg, sync_name):
                continue
            ball = read_table(cfg, f"tracking/{f.match_id}/ball")
            meta = read_metadata(f.tracking_zip)
            # The team attacking left to right, by its StatsBomb name, to compare with the event team.
            pairs = pair_team_names([meta["HomeTeam"]["Name"], meta["AwayTeam"]["Name"]], statsbomb[int(f.match_id)])
            left = {period: pairs.get(team, team) for period, team in left_teams(meta).items()}
            write_table(cfg, sync_name, sync_match(events[events["match_id"] == f.match_id], ball, left))

    # Step 2b: FIFA's ball_alive per frame, from its own tracking file in the same zip (out-of-play check).
    out_of_play = []
    for f in ready.itertuples():
        alive_name = f"tracking/{f.match_id}/ball_alive"
        if not table_exists(cfg, alive_name):
            write_table(cfg, alive_name, fifa_ball_alive(f.tracking_zip))
        shares = out_of_play_shares(read_table(cfg, f"tracking/{f.match_id}/shape"), read_table(cfg, alive_name))
        out_of_play.append({"folder": f.folder, **shares})

    # Step 3: join every match's tables into one, with position groups and StatsBomb team names
    # so the table joins to team_match.
    unmapped = 0
    if len(ready):
        all_running = pd.concat([read_table(cfg, f"tracking/{m}/running") for m in ready["match_id"]],
                                ignore_index=True)
        all_running = with_statsbomb_teams(with_position_group(all_running), men_matches(cfg))
        unmapped = int(all_running["team"].isna().sum())
        all_running["team"] = all_running["team"].fillna(all_running["tracking_team"])
        write_table(cfg, "running_player_match", all_running)

    synced = [m for m in ready["match_id"] if table_exists(cfg, f"tracking/{m}/sync")]
    sync_offsets = pd.concat([read_table(cfg, f"tracking/{m}/sync") for m in synced], ignore_index=True) \
        if synced else pd.DataFrame(columns=SYNC_COLUMNS)
    if synced:
        write_table(cfg, "sync_offsets", sync_offsets)

    # Step 3b: team shape, FIFA phases and running before defensive actions, from the per-match tables.
    extra, extra_checks = shape_phase_window_tables(cfg, folders, ready, sync_offsets)
    for name, table in extra.items():
        write_table(cfg, name, table)

    # Step 3c: the team-level table Person 1 merges into team_match (FIFA physical + tracking sums,
    # shape without the ball, FIFA phase shares).
    provider_table = None
    if len(ready) and table_exists(cfg, "physical_player_match"):
        provider_table = team_match_provider(all_running, read_table(cfg, "physical_player_match"))
        keys = ["tournament", "match_id", "team"]
        provider_table = provider_table.merge(shape_means(extra["shape_team_second"], keys), on=keys, how="left", validate="one_to_one")
        provider_table = provider_table.merge(extra["fifa_phase_shares"], on=keys, how="left", validate="one_to_one")
        write_table(cfg, "team_match_provider", provider_table)
    extra.pop("fifa_phase_shares", None)

    # Step 3d: player lookup (StatsBomb <-> FIFA <-> tracking ids) and the provider check (test 11).
    lookup = player_lookup(cfg, folders, all_running if len(ready) else None)
    if lookup is not None:
        write_table(cfg, "player_lookup", lookup)
    provider = None
    if table_exists(cfg, "events") and table_exists(cfg, "report_team_match"):
        ev = pd.read_parquet(table_path(cfg, "events"), columns=PROVIDER_EVENT_COLUMNS)
        provider = provider_check(statsbomb_team_match(ev[ev["tournament"] == "men"]),
                                  read_table(cfg, "report_team_match"), provider_table)
        write_table(cfg, "provider_check_team_match", provider)

    # Step 4: checks. Folder counts first.
    add("match folders found", "104", len(folders), len(folders) == 104)
    add("folders matched to a StatsBomb match_id", f"{len(folders)} of {len(folders)}",
        folders["matched"].sum(), folders["matched"].all())
    add("folders with a tracking zip", f"{len(folders)} of {len(folders)}",
        folders["tracking_zip"].notna().sum(), folders["tracking_zip"].notna().all())
    add("matches processed this run", "-", len(processed), True)
    add("matches skipped (already done)", "-", len(skipped), True)
    add("running rows whose tracking team matches a StatsBomb team in that match", "0 unmatched", unmapped,
        unmapped == 0)
    for f in folders[~folders["matched"]].itertuples():
        add(f"{f.folder}: not matched to StatsBomb", "matched", 0, False)

    # Then the check against FIFA's physical file, for each match that has one.
    for f in ready[ready["physical_csv"].notna()].itertuples():
        running = read_table(cfg, f"tracking/{f.match_id}/running")
        for row in check_against_fifa(running, read_physical(f.physical_csv)).itertuples():
            add(f"{f.folder}: {row.check}", row.target, row.value, row.passed)

    # Then team_match_provider: one row per team per men's match, both sources where they exist.
    if provider_table is None:
        add("team_match_provider written", "written", 0, False)
    else:
        n_teams = 2 * len(folders)
        add("team_match_provider rows (team-matches)", str(n_teams), len(provider_table), len(provider_table) == n_teams)
        add("team_match_provider rows with FIFA physical totals", "-", provider_table["distance_km"].notna().sum(), True)
        add("team_match_provider rows with tracking totals", "-", provider_table["distance_km_tracking"].notna().sum(), True)
        both = provider_table.dropna(subset=["distance_km", "distance_km_tracking"])
        within = (both["distance_km_tracking"] / both["distance_km"] - 1).abs() <= 0.02
        add("team_match_provider: tracking distance within 2% of FIFA", f"{len(both)} of {len(both)}", within.sum(),
            bool(within.all()))
        parts = provider_table["hsr_with_ball_km"] + provider_table["hsr_without_ball_km"] + provider_table["hsr_out_of_play_km"]
        adds_up = (parts - provider_table["hsr_km_tracking"]).abs().dropna() < 1e-6
        add("team_match_provider: HSR by possession state adds up to the team total", "all", adds_up.sum(),
            bool(adds_up.all()))

    checks.extend(extra_checks)

    # Then out of play: our tracking's ball_possession against FIFA's ball_alive, per match (information)
    # and overall (the two shares should be close; agreement is the share of seconds where they say the same).
    if out_of_play:
        oop = pd.DataFrame(out_of_play)
        for r in oop.itertuples():
            add(f"{r.folder}: out-of-play share, tracking ball_possession", "-", r.tracking_out_of_play_share, None)
            add(f"{r.folder}: out-of-play share, FIFA ball_alive", "-", r.fifa_out_of_play_share, None)
            add(f"{r.folder}: seconds where tracking and FIFA agree on in or out of play", "-", r.agreement_share, None)
        gap = (oop["tracking_out_of_play_share"] - oop["fifa_out_of_play_share"]).abs()
        add("out of play: matches compared with FIFA ball_alive", f"{len(ready)}", len(oop), len(oop) == len(ready))
        add("out of play: median share, tracking ball_possession", "-", oop["tracking_out_of_play_share"].median(), None)
        add("out of play: median share, FIFA ball_alive", "-", oop["fifa_out_of_play_share"].median(), None)
        add("out of play: matches where the two shares differ by more than 5 points", "0", int((gap > 0.05).sum()),
            bool((gap <= 0.05).all()))
        add("out of play: median agreement per second", "-", oop["agreement_share"].median(), None)

    # Then the player lookup: every row matched or listed by method (counts only).
    if lookup is None:
        add("player_lookup written", "written (needs the StatsBomb events table)", 0, False)
    else:
        methods = lookup["method"].value_counts()
        sb_rows = lookup[lookup["statsbomb_player_id"].notna()]
        add("player_lookup: StatsBomb players matched to FIFA or tracking", f"{len(sb_rows)}",
            int((sb_rows["method"] != "unmatched").sum()), bool((sb_rows["method"] != "unmatched").all()))
        for method in ("shirt number", "name check", "unmatched"):
            add(f"player_lookup rows: {method}", "-", int(methods.get(method, 0)), None)
        fifa_rows = lookup[lookup["fifa_player_id"].notna()]
        add("player_lookup: FIFA physical rows with a StatsBomb player", f"{len(fifa_rows)}",
            int(fifa_rows["statsbomb_player_id"].notna().sum()), bool(fifa_rows["statsbomb_player_id"].notna().all()))

    # Then the provider check: per measure, team-matches, correlation and median ratio (information).
    if provider is None:
        add("provider_check_team_match written", "written (needs events and report_team_match)", 0, False)
    else:
        for r in provider_summary(provider).itertuples():
            add(f"provider check {r.measure}: correlation StatsBomb v FIFA ({int(r.team_matches)} team-matches)",
                "-", r.correlation, None)
            add(f"provider check {r.measure}: median ratio StatsBomb / FIFA", "-", r.median_ratio, None)

    # Then the sync: done for every processed match, and test 1 (median error <= 5 yd) per period.
    if events is None:
        add("sync: StatsBomb events table built (H1)", "built", 0, False)
    add("matches with sync offsets", f"{len(ready)} of {len(ready)}", len(synced), len(synced) == len(ready))
    folder_of = dict(zip(ready["match_id"], ready["folder"]))
    for s in sync_offsets.itertuples():
        add(f"{folder_of[s.match_id]}: sync offset s, period {s.period}",
            f"inside -{SEARCH_LIMIT_S:.0f} to {SEARCH_LIMIT_S:.0f}", s.offset_s,
            abs(s.offset_s) < SEARCH_LIMIT_S)  # an offset at the edge of the search means no real fit was found
        add(f"{folder_of[s.match_id]}: sync median error yd, period {s.period}", f"<= {MAX_SYNC_ERROR_YD}",
            s.median_error_yd, s.median_error_yd <= MAX_SYNC_ERROR_YD)

    table = with_tournament(pd.DataFrame(checks, columns=["check", "target", "value", "passed"]))
    write_table(cfg, "checks_tracking", table)

    # Printed: the overall checks, then any per-match check that failed (the full table is saved).
    per_match = table["check"].str.match(r"men: M\d+ ")
    shown = table[~per_match | table["passed"].eq(False)]
    with pd.option_context("display.width", 200, "display.max_colwidth", 70, "display.max_rows", None):
        print(shown.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"per-match checks passed: {table.loc[per_match, 'passed'].eq(True).sum()} of {per_match.sum()}")


MAX_SYNC_ERROR_YD = 5.0  # test 1: tracking-event sync median error


def with_tournament(checks: pd.DataFrame, tournament: str = "men") -> pd.DataFrame:
    """Checks with the tournament they apply to at the start of the check text ("men: ..."), as the
    checks schema asks; texts that already start with a tournament are kept."""
    out = checks.copy()
    prefixed = out["check"].str.startswith(("men:", "women:"))
    out.loc[~prefixed, "check"] = f"{tournament}: " + out.loc[~prefixed, "check"]
    return out
PER_MATCH = ("running", "ball", "shape", "seconds")  # tables saved per match by the tracking pass
LOOKUP_EVENT_COLUMNS = ["tournament", "match_id", "type", "tactics", "team", "position", "substitution_replacement_id",
                        "substitution_replacement"]
PROVIDER_EVENT_COLUMNS = ["tournament", "match_id", "team", "type", "period", "period_seconds", "possession",
                          "possession_team", "shot_outcome", "pass_outcome"]


def player_lookup(cfg: Config, folders: pd.DataFrame, running: pd.DataFrame | None) -> pd.DataFrame | None:
    """player_lookup for every matched folder, or None before the StatsBomb events table exists.

    StatsBomb players from the line-ups in events; FIFA players from the physical CSV (names read for the
    name check only); tracking players from running_player_match. No names are written.
    """
    from trivium.ingest.match_folders import player_lookup_match, read_physical_names, statsbomb_lineups

    import pyarrow.parquet as pq

    if not table_exists(cfg, "events") or not set(LOOKUP_EVENT_COLUMNS) <= set(pq.read_schema(table_path(cfg, "events")).names):
        return None
    ev = pd.read_parquet(table_path(cfg, "events"), columns=LOOKUP_EVENT_COLUMNS)
    lineups = statsbomb_lineups(ev[ev["tournament"] == "men"])
    physical = read_table(cfg, "physical_player_match") if table_exists(cfg, "physical_player_match") else None
    tracked = None
    if running is not None:
        tracked = running[["match_id", "team", "person_id", "jersey_number", "position_group"]].drop_duplicates(
            ["match_id", "person_id"]).rename(columns={"person_id": "tracking_person_id"})

    parts = []
    for f in folders[folders["matched"]].itertuples():
        mid = int(f.match_id)
        fifa = pd.DataFrame(columns=["team", "fifa_player_id", "jersey_number", "name_check"])
        if physical is not None and f.physical_csv is not None:
            fifa = physical.loc[physical["match_id"] == mid, ["team", "fifa_player_id", "jersey_number"]].merge(
                read_physical_names(f.physical_csv), on="fifa_player_id", how="left", validate="one_to_one")
        track = pd.DataFrame(columns=["team", "tracking_person_id", "jersey_number", "position_group", "name_check"])
        if tracked is not None:
            track = tracked.loc[tracked["match_id"] == mid].drop(columns="match_id")
            # Team-sheet names from the tracking metadata, for the name check only (never written).
            meta = read_metadata(f.tracking_zip) if f.tracking_zip is not None else {}
            names = {p["ID"]: p["Name"] for side in ("HomeTeam", "AwayTeam") for p in meta.get(side, {}).get("Players", [])}
            track = track.assign(name_check=track["tracking_person_id"].map(names))
        sb = lineups.loc[lineups["match_id"] == mid].drop(columns="match_id")
        out = player_lookup_match(sb, fifa, track)
        out.insert(0, "match_id", mid)
        out.insert(0, "tournament", "men")
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else None


def shape_phase_window_tables(
    cfg: Config, folders: pd.DataFrame, ready: pd.DataFrame, sync_offsets: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """shape_team_second, phase_clips, running_fifa_phase, event_running_windows and the per-team FIFA
    phase shares (for team_match_provider), all with StatsBomb team names; plus their check rows."""
    from trivium.ingest.match_folders import men_matches, pair_team_names
    from trivium.tracking.phases import phase_shares, read_phase_clips, running_by_phase
    from trivium.tracking.windows import EVENT_COLUMNS, defensive_actions, event_windows

    statsbomb = {int(r.match_id): [r.home_team, r.away_team] for r in men_matches(cfg).itertuples()}
    actions = None
    if table_exists(cfg, "events"):
        ev = pd.read_parquet(table_path(cfg, "events"), columns=["tournament", *EVENT_COLUMNS])
        actions = defensive_actions(ev[ev["tournament"] == "men"])
    offsets = {(int(r.match_id), int(r.period)): float(r.offset_s) for r in sync_offsets.itertuples()}

    shapes, seconds_by_match, windows = [], {}, []
    for f in ready.itertuples():
        mid = int(f.match_id)
        seconds = read_table(cfg, f"tracking/{mid}/seconds")
        shape = read_table(cfg, f"tracking/{mid}/shape")
        pairs = pair_team_names(sorted(seconds["team"].unique()), statsbomb[mid])
        seconds = seconds.assign(team=seconds["team"].map(pairs))
        seconds_by_match[mid] = seconds
        shapes.append(shape.assign(team=shape["team"].map(pairs), match_id=mid))
        if actions is not None:
            half = {p: o for (m, p), o in offsets.items() if m == mid}
            windows.append(event_windows(actions[actions["match_id"] == mid], seconds, half).assign(match_id=mid))

    clips_all, phase_running, shares = [], [], []
    files = folders[folders["matched"] & folders["enhanced_json"].notna()]
    for f in files.itertuples():
        mid = int(f.match_id)
        clips = read_phase_clips(f.enhanced_json)
        pairs = pair_team_names(sorted(clips["fifa_team"].unique()), statsbomb[mid])
        clips = clips.assign(team=clips["fifa_team"].map(pairs), match_id=mid)
        clips_all.append(clips)
        shares.append(phase_shares(clips).assign(match_id=mid))
        if mid in seconds_by_match:
            phase_running.append(running_by_phase(clips, seconds_by_match[mid]).assign(match_id=mid))

    def table(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[c for c in columns if c != "tournament"])
        return out.assign(tournament="men")[columns]

    tables = {
        "shape_team_second": table(shapes, ["tournament", "match_id", "team", "period", "clock_s", "state",
                                            "line_height_m", "team_length_m", "team_width_m"]),
        "phase_clips": table(clips_all, ["tournament", "match_id", "team", "phase", "period", "start_s", "end_s"]),
        "running_fifa_phase": table(phase_running, ["tournament", "match_id", "team", "position_group", "phase",
                                                    "seconds", "distance_m", "hsr_m", "sprint_m"]),
        "fifa_phase_shares": table(shares, ["tournament", "match_id", "team",
                                            *[c for c in (shares[0].columns if shares else []) if c.startswith("fifa_")]]),
    }
    if actions is not None:
        tables["event_running_windows"] = table(windows, ["tournament", "match_id", "event_id", "team", "type",
                                                           "location_x", "location_y", "team_hsr_10s_m",
                                                           "team_distance_10s_m"])

    rows: list[dict] = []

    def add(check: str, target: str, value: float, passed: bool | None) -> None:
        rows.append({"check": check, "target": target, "value": float(value),
                     "passed": None if passed is None else bool(passed)})

    shape = tables["shape_team_second"]
    without = shape[shape["state"] == "without_ball"]
    add("shape_team_second: seconds with a team shape", "-", len(shape), len(shape) > 0)
    add("shape_team_second: teams mapped to StatsBomb names", "all", shape["team"].notna().mean(),
        bool(shape["team"].notna().all()))
    add("shape: median line height without the ball (m from own goal)", "-", without["line_height_m"].median(), None)
    add("shape: median team length without the ball (m)", "-", without["team_length_m"].median(), None)
    add("shape: median team width without the ball (m)", "-", without["team_width_m"].median(), None)
    clips = tables["phase_clips"]
    add("FIFA Enhanced files read", f"{len(folders)}", len(files), len(files) == len(folders))
    add("phase_clips: teams mapped to StatsBomb names", "all", clips["team"].notna().mean(),
        bool(clips["team"].notna().all()))
    add("phase_clips: matches with both teams' phases", f"{len(files)}",
        (clips.groupby("match_id")["team"].nunique() == 2).sum(),
        bool((clips.groupby("match_id")["team"].nunique() == 2).all()))
    phases = tables["running_fifa_phase"]
    add("running_fifa_phase: matches", f"{len(ready)}", phases["match_id"].nunique(),
        phases["match_id"].nunique() == len(ready))
    if actions is not None:
        w = tables["event_running_windows"]
        n_actions = int(actions["match_id"].isin(ready["match_id"].astype(int)).sum())
        add("event_running_windows: defensive actions with a 10 s window", f"{n_actions}", len(w), len(w) == n_actions)
        add("event_running_windows: median team HSR in the 10 s before (m)", "-", w["team_hsr_10s_m"].median(), None)
    return tables, rows


def sync_events(cfg: Config) -> pd.DataFrame | None:
    """Men's StatsBomb passes and shots with the columns sync needs, or None before the events table exists."""
    from trivium.tracking.sync import EVENT_COLUMNS, SYNC_TYPES

    if not table_exists(cfg, "events"):
        return None
    events = pd.read_parquet(table_path(cfg, "events"), columns=["tournament", *EVENT_COLUMNS])
    keep = (events["tournament"] == "men") & events["type"].isin(SYNC_TYPES) & events["period"].isin(PERIODS.values())
    return events.loc[keep, EVENT_COLUMNS]


def with_statsbomb_teams(running: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """running with team set to the StatsBomb name of that match's team (empty when the tracking
    name matches neither team) and the tracking name kept as tracking_team.

    Names are paired per match with pair_team_names, so a name with no alias still maps when the
    match's other team does."""
    from trivium.ingest.match_folders import pair_team_names

    statsbomb = {int(r.match_id): [r.home_team, r.away_team] for r in matches.itertuples()}
    out = running.copy()
    out["tracking_team"] = out["team"]
    names = {}
    for match_id, teams in out.groupby(out["match_id"].astype(int))["team"]:
        pairs = pair_team_names(sorted(teams.unique()), statsbomb.get(match_id, []))
        names.update({(match_id, t): s for t, s in pairs.items()})
    out["team"] = [names.get(k) for k in zip(out["match_id"].astype(int), out["tracking_team"])]
    return out


def ball_table(ball: dict[int, np.ndarray]) -> pd.DataFrame:
    """Ball track per period (from running_player_match) as one table:
    period, clock_s (seconds in the period), ball_x, ball_y (tracking metres)."""
    parts = [
        pd.DataFrame({"period": period, "clock_s": a[:, 0], "ball_x": a[:, 1], "ball_y": a[:, 2]})
        for period, a in sorted(ball.items())
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["period", "clock_s", "ball_x", "ball_y"])
