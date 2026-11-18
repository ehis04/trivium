"""Contract checks: does every built table have the columns, grain and tournaments the rest of
the pipeline relies on (the round 2 table contracts), and which registry measures have data in
each tournament.

A contract lists the required columns only; extra columns are allowed and reported as
information. A table that is not built yet is information, not a failure. team_match follows
the measure registry, so a new registry row extends its contract automatically.

Output tables: checks_contracts, measure_coverage, build_status.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from trivium.config import Config
from trivium.measures import load_measures, measures
from trivium.storage import read_table, table_exists, write_table

CHECK_COLUMNS = ["check", "target", "value", "passed"]
COVERAGE_COLUMNS = ["measure", "label", "table", "column", "tier", "tournament", "present", "coverage", "status"]
STATUS_COLUMNS = ["table", "rows", "columns", "tournaments", "modified", "synthetic", "in_contracts"]
LIVE_COVERAGE = 0.95
BOTH = ("men", "women")
MEN = ("men",)


@dataclass(frozen=True)
class Contract:
    table: str
    columns: tuple[str, ...]          # required columns
    keys: tuple[str, ...]             # grain: unique together (empty: no single grain)
    tournaments: tuple[str, ...]      # expected tournaments (empty: table has no tournament column)
    owner: str
    stage: str


def _c(table: str, columns: str, keys: str, tournaments: tuple[str, ...], owner: str, stage: str) -> Contract:
    return Contract(table, tuple(columns.split()), tuple(keys.split()), tournaments, owner, stage)


def team_match_contract() -> Contract:
    """Keys plus every registry column whose table is team_match."""
    columns = tuple(dict.fromkeys(measures(table="team_match")["column"]))
    return Contract("team_match", ("tournament", "match_id", "team", "opponent", *columns),
                    ("tournament", "match_id", "team"), BOTH, "Person 1", "features")


P1, P2, P3 = "Person 1", "Person 2", "Person 3"
_CONTRACTS = (
    # Person 1: StatsBomb ingest and events features
    _c("matches", "tournament match_id match_date kick_off competition_stage home_team away_team home_team_id "
       "away_team_id home_score away_score neutral_ground shootout shootout_winner", "tournament match_id", BOTH, P1,
       "statsbomb"),
    _c("events", "tournament match_id id index period type team team_id period_seconds location_x location_y",
       "tournament match_id index", BOTH, P1, "statsbomb"),
    _c("shootouts", "tournament match_id team team_id kicks_taken kicks_scored won", "tournament match_id team", BOTH,
       P1, "statsbomb"),
    _c("team_match_zone", "tournament match_id team zone_x zone_y zone contests_won contests_lost contests_share "
       "team_actions opponent_actions territory_share actions_count", "tournament match_id team zone", BOTH, P1,
       "features"),
    _c("team_tournament_zone", "tournament team zone_x zone_y zone contests_won contests_lost contests_share "
       "team_actions opponent_actions territory_share", "tournament team zone", BOTH, P1, "features"),
    _c("team_tournament", "tournament team matches", "tournament team", BOTH, P1, "features"),
    _c("team_match_phase", "tournament match_id team phase possessions possession_seconds shots xg obv "
       "xg_per_possession share_of_possessions", "tournament match_id team phase", BOTH, P1, "features"),
    _c("team_tournament_phase", "tournament team phase possessions possession_seconds shots xg obv xg_per_possession "
       "share_of_possessions", "tournament team phase", BOTH, P1, "features"),
    _c("shot_timeline", "tournament match_id team period minute elapsed_minute half_time_minute xg goal penalty "
       "own_goal", "", BOTH, P1, "features"),
    _c("player_match_minutes", "tournament match_id team player_id jersey_number position position_group started "
       "minutes on_minute off_minute", "tournament match_id team player_id", BOTH, P1, "features"),
    _c("senior_events", "tournament match_id id index period type team team_id location_x location_y",
       "tournament match_id index", (), P1, "statsbomb"),
    _c("senior_matches", "tournament match_id match_date competition_stage home_team away_team home_score away_score",
       "tournament match_id", (), P1, "statsbomb"),
    # Person 3: match folders, tracking, reports
    _c("physical_player_match", "tournament match_id team fifa_player_id jersey_number total_distance_m "
       "dist_20_25_kmh_m dist_25_plus_kmh_m", "tournament match_id fifa_player_id", MEN, P3, "match_folders"),
    _c("running_player_match", "tournament match_id team person_id position_group state third minutes distance_m "
       "hsr_m sprint_m", "tournament match_id team person_id state third", MEN, P3, "tracking"),
    _c("positions_window", "tournament match_id team period window window_start_min state person_id position_group "
       "x_m y_m seconds", "tournament match_id team period window state person_id", MEN, P3, "tracking"),
    _c("formation_team_window", "tournament match_id team period window window_start_min state formation lines gvf "
       "players back_line_m front_line_m depth_m",
       "tournament match_id team period window state", MEN, P3, "tracking"),
    _c("formation_team_match", "tournament match_id team state main_formation main_share formations_used changes "
       "back_line_m depth_m", "tournament match_id team state", MEN, P3, "tracking"),
    _c("formation_listed", "tournament match_id team starting_formation main_formation formation_changes back_line",
       "tournament match_id team", BOTH, P2, "features"),
    _c("sync_offsets", "match_id period offset_s median_error_yd n_events", "match_id period", (), P3, "tracking"),
    _c("team_match_provider", "tournament match_id team distance_km hsr_km sprint_km distance_km_tracking hsr_km_tracking "
       "hsr_with_ball_km hsr_without_ball_km hsr_out_of_play_km hsr_without_ball_defensive_third_km top3_hsr_share "
       "line_height_m team_length_m team_width_m fifa_build_up_share fifa_progression_share fifa_final_third_share "
       "fifa_long_ball_share fifa_attacking_transition_share fifa_counter_attack_share fifa_press_share "
       "fifa_counter_press_share fifa_defensive_transition_share fifa_recovery_share", "tournament match_id team",
       MEN, P3, "tracking"),
    _c("phase_clips", "tournament match_id team phase period start_s end_s", "", MEN, P3, "match_folders"),
    _c("running_fifa_phase", "tournament match_id team position_group phase seconds distance_m hsr_m sprint_m",
       "tournament match_id team position_group phase", MEN, P3, "tracking"),
    _c("shape_team_second", "tournament match_id team period clock_s state line_height_m team_length_m team_width_m",
       "tournament match_id team period clock_s", MEN, P3, "tracking"),
    _c("event_running_windows", "tournament match_id event_id team type location_x location_y team_hsr_10s_m "
       "team_distance_10s_m", "tournament match_id event_id", MEN, P3, "tracking"),
    _c("player_lookup", "tournament match_id team statsbomb_player_id fifa_player_id tracking_person_id jersey_number "
       "position_group method", "tournament match_id team statsbomb_player_id", MEN, P3, "match_folders"),
    _c("fifa_events", "tournament match_id team category event_type event outcome game_period match_run_time_in_ms "
       "pressure line_break_outcome line_break_direction team_units_broken total_team_units team_shape team_unit "
       "sequence_type x_location_start y_location_start x_location_end y_location_end", "", MEN, P3, "match_folders"),
    _c("report_team_match", "tournament match_id team possession_pct possession_in_contest_pct goals attempts "
       "attempts_on_target passes passes_complete pass_completion_pct line_breaks_completed defensive_line_breaks "
       "receptions_final_third crosses ball_progressions pressures_applied direct_pressures forced_turnovers "
       "second_balls distance_km zone4_km", "tournament match_id team", MEN, P3, "reports"),
    _c("provider_check_team_match", "tournament match_id team measure statsbomb_value fifa_value statsbomb_definition "
       "fifa_definition", "tournament match_id team measure", MEN, P3, "reports"),
    # Person 2: Elo, expectation, models
    _c("elo_pre_tournament", "tournament team elo_name elo senior_matches last_senior_match elo_rank_in_tournament",
       "tournament team", BOTH, P2, "external"),
    _c("tds_team", "team elo_name confederation campaigns_incl_2025 best_finish_rank", "team", (), P2, "external"),
    _c("overperformance", "tournament match_id team opponent elo_gap result p_loss p_draw p_win expected_points "
       "actual_points overperformance", "tournament match_id team", BOTH, P2, "expectation"),
    _c("team_summary", "tournament team elo matches actual_points expected_points overperformance band_low band_high "
       "beyond_luck actual_stage_name expected_stage progress_over_expectation", "tournament team", BOTH, P2,
       "expectation"),
    _c("expected_progress", "tournament team actual_stage actual_stage_name expected_stage progress_over_expectation "
       "p_actual_or_better", "tournament team", BOTH, P2, "expectation"),
    _c("luck_summary", "tournament teams above below expected_by_chance", "tournament", BOTH, P2, "expectation"),
    _c("reliability", "tournament measure label n_teams r_split r_sb passed", "tournament measure", BOTH, P2,
       "reliability"),
    _c("drivers", "tournament measure label coef coef_low coef_high excludes_zero", "tournament measure", BOTH, P2,
       "drivers"),
    _c("drivers_summary", "tournament signifiers_used oos_r2 beats_baseline", "tournament", BOTH, P2, "drivers"),
    _c("style_types", "tournament team style_id style_label probability", "tournament team", BOTH, P2, "styles"),
    _c("style_profiles", "tournament style_id style_label n_teams", "tournament style_id", BOTH, P2, "styles"),
    _c("style_overperformance", "tournament style_id style_label n_teams overperformance_per_match low high "
       "excludes_zero", "tournament style_id", BOTH, P2, "styles"),
    _c("profile_comparison", "tournament measure label raw_difference_z raw_low raw_high adjusted_difference_z "
       "adjusted_low adjusted_high adjusted_excludes_zero survives_adjustment verdict", "tournament measure", BOTH,
       P2, "profiles"),
    _c("efficiency_team_match", "tournament match_id team cost_value conversion_value cost_z conversion_z "
       "efficiency_index", "tournament match_id team", BOTH, P2, "profiles"),
    _c("tactical_profile_team", "tournament team matches dominant efficient resourceful profile "
       "overperformance_per_match", "tournament team", BOTH, P2, "profiles"),
    _c("tactical_profile_summary", "tournament profile n_teams teams overperformance_per_match low high excludes_zero",
       "tournament profile", BOTH, P2, "profiles"),
    _c("tactical_profile_axes", "tournament axis label parts n_teams effect low high spearman spearman_low "
       "spearman_high r_sb repeatable predicts_later", "tournament axis", BOTH, P2, "profiles"),
    _c("tactical_profile_sensitivity", "tournament score variant n_teams spearman low high excludes_zero "
       "same_sign_as_main median_threshold_share leave_one_out_share", "tournament score variant", BOTH, P2,
       "profiles"),
    _c("efficiency_team_tournament", "tournament team matches cost_measure efficiency_index efficiency_rank",
       "tournament team", BOTH, P2, "profiles"),
    _c("sustainability", "tournament outcome term coef ci_low ci_high converged", "tournament outcome term", BOTH, P2,
       "sustainability"),
)
CONTRACTS: dict[str, Contract] = {c.table: c for c in (team_match_contract(), *_CONTRACTS)}


def _row(check: str, target: Any, value: Any, passed: bool | None) -> dict[str, object]:
    return {"check": check, "target": str(target), "value": str(value), "passed": pd.NA if passed is None else passed}


def check_table(df: pd.DataFrame, contract: Contract) -> list[dict[str, object]]:
    """Check rows for one built table: columns, grain, tournaments, rows and extra columns."""
    name = contract.table
    missing = [c for c in contract.columns if c not in df.columns]
    rows = [_row(f"{name}: required columns present" + (f" (missing: {', '.join(missing)})" if missing else ""),
                 0, len(missing), not missing)]
    if contract.keys and not missing:
        keyed = df.dropna(subset=list(contract.keys))  # a row with an empty key (e.g. an unmatched player) has no grain
        duplicates = int(keyed.duplicated(subset=list(contract.keys)).sum())
        rows.append(_row(f"{name}: one row per {', '.join(contract.keys)}", 0, duplicates, duplicates == 0))
    if contract.tournaments:
        present = sorted(df["tournament"].dropna().unique()) if "tournament" in df else []
        rows.append(_row(f"{name}: tournaments", ", ".join(contract.tournaments), ", ".join(present) or "none",
                         set(contract.tournaments) <= set(present)))
    rows.append(_row(f"{name}: rows", "information", len(df), None))
    extra = [c for c in df.columns if c not in contract.columns]
    if extra:
        rows.append(_row(f"{name}: extra columns", "information", len(extra), None))
    return rows


def check_contracts(cfg: Config, tables: Sequence[str] | None = None) -> pd.DataFrame:
    """Every contract (or the named ones); a table not built yet is one information row."""
    contracts = {**CONTRACTS, "team_match": team_match_contract()}
    names = list(tables) if tables is not None else list(contracts)
    unknown = [n for n in names if n not in contracts]
    if unknown:
        raise ValueError(f"No contract for: {', '.join(unknown)}.")
    rows: list[dict[str, object]] = []
    for name in names:
        contract = contracts[name]
        if not table_exists(cfg, name):
            rows.append(_row(f"{name}: not built yet ({contract.owner}, {contract.stage} stage)", "built", "missing",
                             None))
        else:
            rows.extend(check_table(read_table(cfg, name), contract))
    table = pd.DataFrame(rows, columns=CHECK_COLUMNS)
    table["passed"] = table["passed"].astype("boolean")
    return table


def measure_coverage(cfg: Config) -> pd.DataFrame:
    """Per registry measure and tournament: share of that tournament's rows with a value, and a status
    (live >= 95%, partial, missing). Men-only measures are not expected for women and get no row."""
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for reg in load_measures().itertuples(index=False):
        if reg.table not in cache:
            cache[reg.table] = read_table(cfg, reg.table) if table_exists(cfg, reg.table) else None
        table = cache[reg.table]
        for tournament in (MEN if reg.tier == "men" else BOTH):
            present = table is not None and reg.column in table.columns
            share = 0.0
            if present:
                rows_t = table[table["tournament"] == tournament] if "tournament" in table else table
                values = pd.to_numeric(rows_t[reg.column], errors="coerce")
                share = float(values.notna().mean()) if len(values) else 0.0
            status = "live" if share >= LIVE_COVERAGE else "partial" if share > 0 else "missing"
            rows.append({"measure": reg.measure, "label": reg.label, "table": reg.table, "column": reg.column,
                         "tier": reg.tier, "tournament": tournament, "present": present, "coverage": share,
                         "status": status})
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_status(cfg: Config) -> pd.DataFrame:
    """One row per Parquet table in the build folder."""
    if not cfg.build_dir.is_dir():
        return pd.DataFrame(columns=STATUS_COLUMNS)
    info = read_table(cfg, "build_info") if table_exists(cfg, "build_info") else pd.DataFrame()
    synthetic = bool(len(info)) and "source" in info and bool(info["source"].eq("synthetic").all())
    rows = []
    for path in sorted(cfg.build_dir.glob("*.parquet")):
        df = pd.read_parquet(path)
        tournaments = sorted(df["tournament"].dropna().astype(str).unique()) if "tournament" in df else []
        rows.append({"table": path.stem, "rows": len(df), "columns": df.shape[1], "tournaments": ", ".join(tournaments),
                     "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(
                         timespec="seconds"),
                     "synthetic": synthetic, "in_contracts": path.stem in CONTRACTS})
    return pd.DataFrame(rows, columns=STATUS_COLUMNS)


def contract_failures(checks: pd.DataFrame) -> pd.DataFrame:
    """Rows that break a contract (missing columns, duplicate keys, missing tournaments)."""
    return checks[checks["passed"].eq(False).fillna(False)]


def run(cfg: Config) -> None:
    write_table(cfg, "checks_contracts", check_contracts(cfg))
    write_table(cfg, "measure_coverage", measure_coverage(cfg))
    write_table(cfg, "build_status", build_status(cfg))
