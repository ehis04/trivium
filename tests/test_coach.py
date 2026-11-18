from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from trivium import synthetic
from trivium.charts.tactical import team_scores
from trivium.coach import NOTABLE_Z, scouting_report, scouting_report_from_tables
from trivium.measures import measure
from trivium.storage import write_table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "app"))

RULES = {
    "security": {
        "high": "Get into their defensive third often, even with simple entries.",
        "low": "Be patient and only shoot from good positions.",
    },
    "chances_share": {
        "high": "Cut off their supply and stay compact around your box.",
        "low": "Be brave on the ball.",
    },
    "territory_share": {
        "high": "Plan to beat their press and keep an outlet behind it.",
        "low": "Push your line up and squeeze the space.",
    },
    "contests_share": {
        "high": "Avoid turning the match into a battle of duels and second balls.",
        "low": "Compete physically and play for the second ball.",
    },
    "conversion": {
        "high": "Stop the final pass and defend against cut-backs.",
        "low": "Let them have territory, then defend the box.",
    },
    "defensive_work_per_dominance": {
        "high": "Keep the ball, move it quickly and make them chase; expect them to tire late.",
        "low": "Press them early.",
    },
    "chance_zone_spread": {
        "low": "Protect the few areas they create chances from.",
    },
    "pressures_per_opp_possession_min": {
        "high": "Prepare your build-up to beat the first press; the goalkeeper and centre-backs need a long option.",
    },
    "possession_share": {
        "high": "Defend compactly and counter.",
        "low": "Be patient and use the width.",
    },
    "directness": {
        "high": "Win the first header and the second ball.",
    },
    "fifa_block_share": {
        "high": "Be patient, use the width, and look for crosses and cut-backs.",
    },
    "fifa_press_share": {
        "high": "Go long early to get beyond their press.",
    },
    "fifa_counter_attack_share": {
        "high": "Keep two defenders and a midfielder behind the ball.",
    },
}


def _standardised_values(z_score: float) -> list[float]:
    other_count = 4
    pair_spread = np.sqrt((other_count - z_score**2 * 5 / other_count) / 2)
    baseline = -z_score / other_count
    return [z_score, baseline + pair_spread, baseline - pair_spread, baseline, baseline]


def _tables(
    standout: str | None = None,
    raw_side: str = "high",
    z_score: float = 1.0,
    tournament: str = "men",
) -> dict[str, pd.DataFrame]:
    teams = ["Team A", "Team B", "Team C", "Team D", "Team E"]
    table = pd.DataFrame({"team": teams, "tournament": tournament, "matches": [3] * len(teams)})
    measure_names = set(RULES)
    measure_names.add("top2_xg_xa_share")
    for name in measure_names:
        definition = measure(name)
        column = str(definition["column"])
        table[column] = [0.0] * len(teams)
    if standout is not None:
        definition = measure(standout)
        target_raw_z = z_score if raw_side == "high" else -z_score
        table[str(definition["column"])] = _standardised_values(target_raw_z)

    return {"team_tournament": table}


def _note_for(report: dict, area: str, expected: str) -> dict[str, str]:
    return next(
        note for note in report["notes"]
        if note["area"] == area and expected in note["recommendation"]
    )


@pytest.mark.parametrize(
    ("name", "raw_side", "expected"),
    [
        ("security", "high", RULES["security"]["high"]),
        ("security", "low", RULES["security"]["low"]),
        ("chances_share", "high", RULES["chances_share"]["high"]),
        ("chances_share", "low", RULES["chances_share"]["low"]),
        ("territory_share", "high", RULES["territory_share"]["high"]),
        ("territory_share", "low", RULES["territory_share"]["low"]),
        ("contests_share", "high", RULES["contests_share"]["high"]),
        ("contests_share", "low", RULES["contests_share"]["low"]),
        ("conversion", "high", RULES["conversion"]["high"]),
        ("conversion", "low", RULES["conversion"]["low"]),
        ("defensive_work_per_dominance", "high", RULES["defensive_work_per_dominance"]["high"]),
        ("defensive_work_per_dominance", "low", RULES["defensive_work_per_dominance"]["low"]),
        ("chance_zone_spread", "low", RULES["chance_zone_spread"]["low"]),
        ("pressures_per_opp_possession_min", "high", RULES["pressures_per_opp_possession_min"]["high"]),
        ("possession_share", "high", RULES["possession_share"]["high"]),
        ("possession_share", "low", RULES["possession_share"]["low"]),
        ("directness", "high", RULES["directness"]["high"]),
        ("fifa_block_share", "high", RULES["fifa_block_share"]["high"]),
        ("fifa_press_share", "high", RULES["fifa_press_share"]["high"]),
        ("fifa_counter_attack_share", "high", RULES["fifa_counter_attack_share"]["high"]),
    ],
)
def test_measure_rules_use_registry_orientation_and_recommendations(
    name: str,
    raw_side: str,
    expected: str,
) -> None:
    report = scouting_report_from_tables(_tables(name, raw_side), "Team A", "men")

    assert _note_for(report, "Phases of play" if name.startswith("fifa_") else
                    "Spending and cost" if name in {
                        "defensive_work_per_dominance", "pressures_per_opp_possession_min", "possession_share",
                        "directness",
                    } else "Chances" if name in {"security", "conversion", "chance_zone_spread"} else "Control",
                    expected)["recommendation"] == expected


@pytest.mark.parametrize("threshold", [NOTABLE_Z, NOTABLE_Z - 0.001])
def test_notes_fire_at_quarter_cutoff_only(threshold: float) -> None:
    tables = _tables("chances_share", z_score=threshold)

    report = scouting_report_from_tables(tables, "Team A", "men")

    has_chances_note = any(note["observation"] == "They win a large share of the match's chances."
                           for note in report["notes"])
    assert has_chances_note is (threshold >= NOTABLE_Z)


def test_evidence_names_quarter_and_usable_team_count() -> None:
    report = scouting_report_from_tables(_tables("chances_share"), "Team A", "men")
    evidence = next(note["evidence"] for note in report["notes"] if note["observation"].startswith("They win"))

    assert evidence.startswith("Among the top third of the 5 teams")


def test_game_plan_uses_three_largest_absolute_z_scores() -> None:
    tables = _tables()
    table = tables["team_tournament"]
    for name, z_score in (
        ("chances_share", 0.8),
        ("territory_share", 1.2),
        ("contests_share", -1.0),
        ("conversion", 1.5),
    ):
        table[str(measure(name)["column"])] = _standardised_values(z_score)
    report = scouting_report_from_tables(tables, "Team A", "men")

    assert report["game_plan"] == [
        RULES["conversion"]["high"],
        RULES["territory_share"]["high"],
        RULES["contests_share"]["low"],
    ]


def test_resourcefulness_profile_advises_when_costly_or_resourceful() -> None:
    tables = _tables()
    profiles = pd.DataFrame(
        {
            "team": ["Team A"],
            "tournament": ["men"],
            "profile": ["Dominant · Efficient · Costly"],
            "resourceful": [-0.8],
        }
    )
    tables["tactical_profile_team"] = profiles
    costly = scouting_report_from_tables(tables, "Team A", "men")
    assert RULES["defensive_work_per_dominance"]["high"] in costly["game_plan"]

    tables["tactical_profile_team"] = profiles.assign(resourceful=0.8)
    efficient = scouting_report_from_tables(tables, "Team A", "men")
    assert any("control games cheaply" in note["recommendation"] for note in efficient["notes"])


def test_top_two_concentration_and_low_zone_spread_recommendations() -> None:
    tables = _tables()
    top2 = str(measure("top2_xg_xa_share")["column"])
    spread = str(measure("chance_zone_spread")["column"])
    tables["team_tournament"][top2] = _standardised_values(1.0)
    tables["team_tournament"][spread] = _standardised_values(-1.0)

    report = scouting_report_from_tables(tables, "Team A", "men")

    assert any("few contributors" in note["observation"] for note in report["notes"])
    assert RULES["chance_zone_spread"]["low"] in report["game_plan"]


def test_women_report_has_no_men_only_phase_notes() -> None:
    tables = _tables("fifa_block_share", tournament="women")

    report = scouting_report_from_tables(tables, "Team A", "women")

    assert not any(note["area"] == "Phases of play" for note in report["notes"])


@pytest.mark.parametrize(
    ("formation", "expected"),
    [
        ("3-5-2", "Use the space between their wing-back and wide centre-back, and switch play quickly."),
        ("4-3-3", "Overload their full-backs two against one."),
    ],
)
def test_formation_note_recommends_attacking_matchup(formation: str, expected: str) -> None:
    tables = _tables()
    tables["formation_listed"] = pd.DataFrame(
        {"team": ["Team A"], "tournament": ["men"], "main_formation": [formation]}
    )

    report = scouting_report_from_tables(tables, "Team A", "men")

    assert _note_for(report, "Shape", expected)["recommendation"] == expected


def test_report_interface_uses_profile_summary_and_answer_evidence() -> None:
    tables = _tables("chances_share")
    tables["tactical_profile_team"] = pd.DataFrame(
        {
            "team": ["Team A"],
            "tournament": ["men"],
            "profile": ["Dominant · Efficient · Costly"],
            "dominant": [1.0],
            "efficient": [0.5],
            "resourceful": [-0.5],
        }
    )
    tables["tactical_profile_summary"] = pd.DataFrame(
        {
            "team": ["Team A"],
            "tournament": ["men"],
            "profile": ["Dominant · Efficient · Costly"],
            "overperformance_per_match": [0.5],
            "low": [0.1],
            "high": [0.9],
            "n_teams": [4],
            "excludes_zero": [True],
        }
    )
    tables["profile_comparison"] = pd.DataFrame(
        {
            "tournament": ["men", "men"],
            "measure": ["chances_share", "security"],
            "label": ["Chances share", "Security"],
            "adjusted_difference_z": [0.5, -0.2],
            "adjusted_low": [0.1, -0.6],
            "adjusted_high": [0.9, 0.2],
            "adjusted_excludes_zero": [True, False],
            "direction": ["higher", "lower"],
            "verdict": ["survives adjustment", "no clear difference"],
        }
    )
    tables["finding_gate"] = pd.DataFrame(
        {
            "tournament": ["men"],
            "measure": ["chances_share"],
            "passes_gate": [True],
        }
    )

    report = scouting_report_from_tables(tables, "Team A", "men")

    assert set(report) == {"team", "tournament", "profile", "summary", "game_plan", "notes"}
    assert report["profile"] == "Dominant · Efficient · Costly"
    assert report["summary"].startswith("Team A are dominant, efficient, costly.")
    assert "won on average 0.50 points per match more than their strength predicted (4 teams)" in report["summary"]
    assert "Dominance:" not in report["summary"]
    note = _note_for(report, "Control", "Cut off")
    assert "difference remained after accounting for senior strength" in note["evidence"]
    assert "evidence checks passed" in note["evidence"]


def test_scouting_report_reads_tables_from_config(tmp_path: Path) -> None:
    cfg = synthetic.demo_config(tmp_path)
    tables = _tables("chances_share")
    for name, frame in tables.items():
        write_table(cfg, name, frame)

    report = scouting_report(cfg, "Team A", "men")

    assert report["team"] == "Team A"
    assert report["game_plan"]


@pytest.fixture(scope="module")
def coach_demo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("coach-view-demo") / "data"
    cfg = synthetic.write_demo(root, scale="small", luck_sims=200, progress_sims=200)
    write_table(
        cfg,
        "tactical_profile_team",
        pd.DataFrame(
            {
                "team": ["Team A1", "Team A1"],
                "tournament": ["men", "women"],
                "profile": ["Dominant · Efficient · Costly", "Less dominant · Efficient · Resourceful"],
                "dominant": [0.8, -0.4],
                "efficient": [0.6, 0.3],
                "resourceful": [-0.4, 0.7],
            }
        ),
    )
    write_table(
        cfg,
        "tactical_profile_summary",
        pd.DataFrame(
            {
                "profile": ["Dominant · Efficient · Costly", "Less dominant · Efficient · Resourceful"],
                "tournament": ["men", "women"],
                "overperformance_per_match": [0.2, 0.1],
                "low": [-0.1, -0.2],
                "high": [0.5, 0.4],
                "n_teams": [3, 2],
                "excludes_zero": [False, False],
            }
        ),
    )
    return cfg.data_root


@pytest.fixture(autouse=True)
def demo_environment(coach_demo_root: Path) -> Iterator[None]:
    previous = os.environ.get("TRIVIUM_DATA_ROOT")
    os.environ["TRIVIUM_DATA_ROOT"] = str(coach_demo_root)
    yield
    if previous is None:
        os.environ.pop("TRIVIUM_DATA_ROOT", None)
    else:
        os.environ["TRIVIUM_DATA_ROOT"] = previous


def test_coach_view_runs_with_demo_build_for_both_tournaments(coach_demo_root: Path) -> None:
    page = REPO_ROOT / "app" / "pages" / "1_Coach_view.py"
    app = AppTest.from_file(str(page), default_timeout=60).run()

    assert not app.exception
    assert app.get("download_button")
    assert any(item.value == "Three things to do against them" for item in app.subheader)

    app.sidebar.selectbox(key="tournament").set_value("women").run()

    assert not app.exception
    assert app.get("download_button")
    assert not any(
        "FIFA phase profile" in element.value
        for element in app.markdown
        if isinstance(element.value, str)
    )


def test_profile_chart_uses_the_existing_team_scores_chart() -> None:
    teams = pd.DataFrame(
        {
            "tournament": ["men"],
            "team": ["Team A"],
            "dominant": [1.0],
            "efficient": [-0.5],
            "resourceful": [0.25],
        }
    )

    figure = team_scores(teams, "men", "Team A")

    assert list(figure.data[0].y) == ["Dominant", "Efficient", "Resourceful"]
