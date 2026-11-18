"""Tests for the FIFA post-match report reader (three_trees.ingest.reports), on made-up page text."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from three_trees.ingest.reports import parse_key_stats, parse_phases, report_path, report_rows

# Made-up page text in the layout of the real reports (page 3 and page 4).
KEY_STATS_PAGE = """3 November 2025 - Some Stadium - Pitch 1 - Some Ground - 18:45
Match Summary - Key Statistics
2 1
Aland Korea Republic
Possession
Total 40% 15% 45% Total
2 Goals 1
12 (5) Attempts at Goal (On Target) 8 (3)
400 (350) Total Passes (Complete) 500 (430)
88 % Pass Completion % 86 %
150 Completed Line Breaks 170
40 Defensive Line Breaks 35
60 Receptions in the Final Third 75
14 Crosses 20
30 Ball Progressions 25
300 (120) Defensive Pressures Applied (Direct Pressures) 250 (90)
45 Forced Turnovers 50
33 Second Balls 29
106.2 km Total Distance Covered 105.9 km
7.1 km Zone 4 - Low Speed Sprinting: 20-25 km/h 6.8 km"""

PHASES_PAGE = """3 November 2025 - Some Stadium - Pitch 1 - Some Ground - 18:45
Aland Phases of Play Korea Republic
IN POSSESSION
10% Build Up Unopposed 12%
8% Build Up Opposed 9%
OUT OF POSSESSION
20% Low Block 5%
6% Counter-press 7%"""


def test_parse_key_stats_reads_both_columns() -> None:
    sides = parse_key_stats(KEY_STATS_PAGE)
    home, away = sides["home"], sides["away"]
    assert (home["goals"], away["goals"]) == (2, 1)
    assert (home["attempts"], home["attempts_on_target"], away["attempts"], away["attempts_on_target"]) == (12, 5, 8, 3)
    assert (home["pressures_applied"], home["direct_pressures"]) == (300, 120)
    assert (home["possession_pct"], home["possession_in_contest_pct"], away["possession_pct"]) == (40, 15, 45)
    assert home["pass_completion_pct"] == 88 and away["distance_km"] == pytest.approx(105.9)
    assert home["zone4_km"] == pytest.approx(7.1) and away["zone4_km"] == pytest.approx(6.8)


def test_parse_phases_reads_team_names_with_spaces() -> None:
    home, away, sides = parse_phases(PHASES_PAGE)
    assert (home, away) == ("Aland", "Korea Republic")
    assert sides["home"]["phase_low_block_pct"] == 20 and sides["away"]["phase_counter_press_pct"] == 7


def test_report_rows_pair_names_with_statsbomb() -> None:
    home, away, phases = parse_phases(PHASES_PAGE)
    key = parse_key_stats(KEY_STATS_PAGE)
    report = {"home": home, "away": away, "values": {s: {**key[s], **phases[s]} for s in ("home", "away")}}
    rows = report_rows(report, 7, ["Aland U17", "Korea Republic U17"], "men").set_index("team")
    assert rows.loc["Aland U17", "goals"] == 2 and rows.loc["Korea Republic U17", "goals"] == 1
    assert set(rows["tournament"]) == {"men"} and set(rows["match_id"]) == {7}


def test_report_path_prefers_v2(tmp_path: Path) -> None:
    (tmp_path / "M02 RSA V BOL POST MATCH REPORT.pdf").write_bytes(b"")
    (tmp_path / "M02 RSA V BOL POST MATCH REPORT V2.pdf").write_bytes(b"")
    assert report_path(tmp_path).name == "M02 RSA V BOL POST MATCH REPORT V2.pdf"


# ---------------------------------------------------------------------------
# Provider check (three_trees.tracking.provider)
# ---------------------------------------------------------------------------

def test_provider_check_puts_both_providers_side_by_side() -> None:
    from three_trees.tracking.provider import provider_check, provider_summary, statsbomb_team_match

    # Made-up events: Aland has possession 1 (0-30 s), Bland possession 2 (30-40 s), all in period 1.
    events = pd.DataFrame({
        "tournament": "men", "match_id": 1, "period": 1,
        "team": ["Aland", "Aland", "Aland", "Bland", "Bland", "Bland"],
        "possession_team": ["Aland U17"] * 3 + ["Bland U17"] * 3, "possession": [1, 1, 1, 2, 2, 2],
        "period_seconds": [0.0, 10.0, 30.0, 30.0, 35.0, 40.0],
        "type": ["Pass", "Pass", "Shot", "Pass", "Pressure", "Shot"],
        "pass_outcome": [None, "Incomplete", None, None, None, None],
        "shot_outcome": [None, None, "Goal", None, None, "Off T"],
    })
    sb = statsbomb_team_match(events).set_index("team")
    assert sb.loc["Aland", "sb_possession_pct"] == pytest.approx(75.0)  # 30 s of 40 s
    assert sb.loc["Aland", "sb_pass_completion_pct"] == pytest.approx(50.0)
    assert (sb.loc["Aland", "sb_shots_on_target"], sb.loc["Bland", "sb_shots_on_target"]) == (1, 0)

    report = pd.DataFrame({"tournament": "men", "match_id": 1, "team": ["Aland", "Bland"], "possession_pct": [60.0, 30.0],
                           "attempts": [1.0, 1.0], "attempts_on_target": [1.0, 0.0], "passes": [2.0, 1.0],
                           "passes_complete": [1.0, 1.0], "pass_completion_pct": [50.0, 100.0],
                           "pressures_applied": [0.0, 3.0]})
    check = provider_check(sb.reset_index(), report, None)
    assert set(check["measure"]) == {"possession_pct", "shots", "shots_on_target", "passes", "passes_complete",
                                     "pass_completion_pct", "pressures"}  # no distance without team_match_provider
    row = check[(check["team"] == "Bland") & (check["measure"] == "pressures")].iloc[0]
    assert (row["statsbomb_value"], row["fifa_value"]) == (1.0, 3.0)
    assert row["statsbomb_definition"].startswith("StatsBomb") and row["fifa_definition"].startswith("FIFA")
    summary = provider_summary(check).set_index("measure")
    assert summary.loc["shots", "team_matches"] == 2
