from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from trivium.config import Config
from trivium.definitions import POINT_DEFINITIONS, POINTS, RESEARCH_QUESTION
from trivium.models import answer
from trivium.query import fallback, functions
from trivium.storage import read_table, write_table
from trivium.synthetic import demo_config, write_demo


@pytest.fixture(scope="module")
def demo_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    return write_demo(
        tmp_path_factory.mktemp("query_answer"), scale="small", run_models=True,
        luck_sims=200, progress_sims=200,
    )


def _comparison() -> pd.DataFrame:
    rows = []
    for tournament in ("men", "women"):
        for measure, label, value, clear in (
            ("chances_share", "Chances share", 0.4 if tournament == "men" else -0.3, True),
            ("contests_share", "Contests won share", 0.1, False),
            ("defensive_work_per_dominance", "Defensive work per point of dominance", -0.5,
             tournament == "women"),
            ("minutes_outside_top11_share", "Minutes outside the most-used 11", 0.0, False),
            ("security", "Security", -0.7, tournament == "men"),
            ("conversion", "Conversion", 0.0, False),
            ("unmapped_measure", "Unmapped measure", 0.8, True),
        ):
            low, high = (value - 0.1, value + 0.1) if clear else (-1.0, 1.0)
            rows.append({
                "tournament": tournament, "measure": measure, "label": label,
                "raw_difference_z": value, "raw_low": low, "raw_high": high,
                "adjusted_difference_z": value, "adjusted_low": low, "adjusted_high": high,
                "adjusted_excludes_zero": clear,
                "verdict": "survives adjustment" if clear else "no clear difference",
            })
    return pd.DataFrame(rows)


def _controlled_cfg(tmp_path: Path) -> Config:
    cfg = demo_config(tmp_path)
    write_table(cfg, "profile_comparison", _comparison())
    write_table(cfg, "finding_gate", pd.DataFrame({
        "tournament": ["men", "men", "men", "women", "men"],
        "finding_type": ["tactical profile", "tactical profile", "driver", "tactical profile",
                         "tactical profile"],
        "passes_gate": pd.Series([True, False, True, True, pd.NA], dtype="boolean"),
        "wording": ["Approved men's profile.", "Rejected men's profile.", "A different type of finding.",
                    "Approved women's profile.", "Untested men's profile."],
    }))
    return cfg


def _install_coach(
    monkeypatch: pytest.MonkeyPatch, report: dict[str, object],
) -> list[tuple[Config, str, str]]:
    calls = []
    module = ModuleType("trivium.coach")

    def scouting_report(cfg: Config, team: str, tournament: str) -> dict[str, object]:
        calls.append((cfg, team, tournament))
        return report

    module.scouting_report = scouting_report
    monkeypatch.setitem(sys.modules, "trivium.coach", module)
    return calls


def _report() -> dict[str, object]:
    return {
        "team": "Team A1", "tournament": "men", "profile": "Dominant, resourceful, efficient",
        "summary": "This synthetic team controls territory and limits entries.",
        "game_plan": ["Keep the ball away from the crowded side.", "Protect the central entry.",
                      "Offer a short passing option.", "An additional idea."],
        "notes": [{"area": "Territory", "observation": "The team keeps control in the centre.",
                   "recommendation": "Use the wider passing option.",
                   "evidence": "Synthetic team averages."}],
    }


@pytest.mark.parametrize("tournament", ["men", "women"])
def test_research_answer_on_demo_uses_shared_question_and_evidence(
    demo_cfg: Config, tournament: str,
) -> None:
    comparison = read_table(demo_cfg, "profile_comparison")
    expected = answer.point_evidence(comparison)
    expected = expected.loc[expected["tournament"].eq(tournament)].reset_index(drop=True)
    result = functions.research_answer(demo_cfg, tournament)
    assert result.available and result.chart is not None
    assert result.numbers["question"] == RESEARCH_QUESTION
    text = fallback.template(result)
    assert RESEARCH_QUESTION in text
    assert [point["point"] for point in result.numbers["points"]] == list(POINTS)
    assert text.index(RESEARCH_QUESTION) < text.index(result.numbers["points"][0]["sentence"])
    pd.testing.assert_frame_equal(result.table.reset_index(drop=True), expected, check_dtype=False)
    assert functions.ensure_aggregated(result) is result


@pytest.mark.parametrize("tournament", ["men", "women"])
def test_research_answer_uses_both_tournaments_for_verdicts_and_only_local_wording(
    tmp_path: Path, tournament: str,
) -> None:
    cfg = _controlled_cfg(tmp_path)
    result = functions.research_answer(cfg, tournament)
    verdicts = answer.point_verdicts(_comparison()).set_index("point")
    points = {row["point"]: row for row in result.numbers["points"]}
    for point in POINTS:
        actual = points[point]
        assert actual["definition"] == POINT_DEFINITIONS[point]
        assert actual["verdict"] == verdicts.loc[point, "verdict"]
        assert actual["clear"] == verdicts.loc[point, f"{tournament}_clear"]
        assert actual["not_clear"] == verdicts.loc[point, f"{tournament}_not_clear"]
        assert actual["sentence"] in fallback.template(result)
    assert result.numbers["wording"] == [f"Approved {tournament}'s profile."]
    assert "Unmapped measure" not in result.table["label"].tolist()
    text = fallback.template(result)
    assert "Rejected" not in text and "Untested" not in text and "different type" not in text
    assert result.numbers["chart_caption"]


def test_changing_synthetic_evidence_changes_every_point_verdict(tmp_path: Path) -> None:
    cfg = _controlled_cfg(tmp_path)
    before = functions.research_answer(cfg, "men")
    comparison = _comparison()
    comparison["adjusted_excludes_zero"] = False
    write_table(cfg, "profile_comparison", comparison)
    after = functions.research_answer(cfg, "men")
    assert before.numbers["points"] != after.numbers["points"]
    assert all(point["verdict"] == "no clear difference" for point in after.numbers["points"])
    assert all(not point["clear"] for point in after.numbers["points"])


@pytest.mark.parametrize("point", list(POINTS))
def test_point_evidence_on_demo_filters_one_point(demo_cfg: Config, point: str) -> None:
    result = functions.point_evidence(demo_cfg, point.lower(), "women")
    expected = answer.point_evidence(read_table(demo_cfg, "profile_comparison"))
    expected = expected.loc[expected["tournament"].eq("women") & expected["point"].eq(point)]
    assert result.available
    assert result.numbers["point"] == point
    assert result.numbers["definition"] == POINT_DEFINITIONS[point]
    pd.testing.assert_frame_equal(result.table.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_dtype=False)
    assert result.numbers["sentence"] in fallback.template(result)
    assert functions.ensure_aggregated(result) is result


def test_point_evidence_preserves_clear_direction_and_unclear_measures(tmp_path: Path) -> None:
    cfg = _controlled_cfg(tmp_path)
    men = functions.point_evidence(cfg, "dOmInAnCe", "men")
    women = functions.point_evidence(cfg, "Dominance", "women")
    assert "Chances share (higher)" in men.numbers["clear"]
    assert "Chances share (lower)" in women.numbers["clear"]
    assert "Contests won share" in men.numbers["not_clear"]
    assert men.numbers["verdict"] == "clear in both tournaments"
    assert set(men.table["tournament"]) == {"men"}


def test_unknown_point_gives_a_plain_query_error(tmp_path: Path) -> None:
    cfg = _controlled_cfg(tmp_path)
    with pytest.raises(functions.QueryError, match="point|Dominance"):
        functions.point_evidence(cfg, "Unrecognised point", "men")


@pytest.mark.parametrize(
    ("name", "args"),
    [("research_answer", {"tournament": "men"}),
     ("point_evidence", {"point": "Dominance", "tournament": "women"})],
)
def test_missing_comparison_returns_a_plain_unavailable_answer(
    tmp_path: Path, name: str, args: dict[str, object],
) -> None:
    result = functions.run_function(demo_config(tmp_path), name, args)
    assert not result.available and result.table.empty
    assert "not built" in fallback.template(result).lower()
    assert functions.ensure_aggregated(result) is result


def test_how_to_play_against_calls_coach_and_limits_game_plan(
    demo_cfg: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    calls = _install_coach(monkeypatch, report)
    result = functions.how_to_play_against(demo_cfg, "Team A1", "men")
    assert calls == [(demo_cfg, "Team A1", "men")]
    assert result.available
    assert result.numbers["profile"] == report["profile"]
    assert result.numbers["summary"] == report["summary"]
    assert result.numbers["game_plan"] == report["game_plan"][:3]
    assert list(result.table.columns) == ["area", "observation", "recommendation", "evidence"]
    assert result.table.to_dict("records") == report["notes"]
    text = fallback.template(result)
    assert report["summary"] in text
    assert report["game_plan"][0] in text
    assert "An additional idea." not in text
    assert functions.ensure_aggregated(result) is result


def test_how_to_play_against_propagates_tournament_to_coach(
    demo_cfg: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report() | {"tournament": "women"}
    calls = _install_coach(monkeypatch, report)
    result = functions.how_to_play_against(demo_cfg, "Team A1", "women")
    assert calls == [(demo_cfg, "Team A1", "women")]
    assert result.numbers["tournament"] == "women"


def test_missing_coach_module_is_unavailable_before_resolving_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "trivium.coach", None)
    result = functions.how_to_play_against(demo_config(tmp_path), "Team A1", "men")
    assert not result.available and result.table.empty
    assert "not built" in fallback.template(result).lower()
    assert functions.ensure_aggregated(result) is result


def test_nested_player_identifiers_in_coach_notes_are_rejected(
    demo_cfg: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report()
    report["notes"][0]["evidence"] = {"person_id": 7}
    _install_coach(monkeypatch, report)
    forbidden = "(?i)player|person|aggregat|identifier|team-level"
    with pytest.raises((functions.QueryError, ValueError), match=forbidden):
        functions.how_to_play_against(demo_cfg, "Team A1", "men")


@pytest.mark.parametrize("plan", ["Attack", None, {"instruction": "Attack"}, [None]])
def test_malformed_coach_game_plan_gives_a_plain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: object,
) -> None:
    report = _report() | {"game_plan": plan}
    _install_coach(monkeypatch, report)
    with pytest.raises(functions.QueryError, match="game plan"):
        functions.how_to_play_against(demo_config(tmp_path), "Team A1", "men")


@pytest.mark.parametrize("field", ["profile", "summary"])
def test_coach_profile_and_summary_must_be_plain_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    _install_coach(monkeypatch, _report() | {field: {"unexpected": "value"}})
    with pytest.raises(functions.QueryError, match="plain text"):
        functions.how_to_play_against(demo_config(tmp_path), "Team A1", "men")


def test_missing_gate_keeps_evidence_and_reports_the_missing_claims(tmp_path: Path) -> None:
    cfg = demo_config(tmp_path)
    write_table(cfg, "profile_comparison", _comparison())
    result = functions.research_answer(cfg, "men")
    assert result.available and not result.table.empty
    assert result.numbers["wording"] == []
    assert "finding_gate is not built in this build" in fallback.template(result)


def test_comparison_chart_explanation_and_womens_colour(tmp_path: Path) -> None:
    cfg = _controlled_cfg(tmp_path)
    result = functions.research_answer(cfg, "women")
    clear = [trace for trace in result.chart.data if str(trace.name).startswith("Above minus below")]
    assert clear and all(trace.marker.color == "#CC79A7" for trace in clear)
    assert all(trace.error_x.color == "#CC79A7" for trace in clear)
    assert "hollow grey" in result.numbers["chart_caption"]
    assert "pink" in result.numbers["chart_caption"]
    assert "crossing zero" in result.numbers["chart_caption"]


@pytest.mark.parametrize(
    ("question", "name", "extra"),
    [
        ("What is the research question?", "research_answer", {}),
        ("What is the answer?", "research_answer", {}),
        ("Explain the three points", "research_answer", {}),
        ("Dominance", "point_evidence", {"point": "Dominance"}),
        ("Resourcefulness for the women", "point_evidence", {"point": "Resourcefulness"}),
        ("Efficiency", "point_evidence", {"point": "Efficiency"}),
        ("How to play against Team A1", "how_to_play_against", {"team": "Team A1"}),
        ("How do we beat Team A2?", "how_to_play_against", {"team": "Team A2"}),
        ("Scouting for Team A1", "how_to_play_against", {"team": "Team A1"}),
        ("Game plan against Team A1", "how_to_play_against", {"team": "Team A1"}),
        ("Tactical profile for Team A1", "tactical_profile", {"team": "Team A1"}),
        ("Which profiles that win?", "profiles_that_win", {}),
        ("Which dominant teams beat their strength?", "profiles_that_win", {}),
        ("What findings answer the research question?", "research_answer", {}),
        ("What findings support a game plan against Team A1?", "how_to_play_against", {"team": "Team A1"}),
    ],
)
def test_research_and_coaching_routes_preserve_existing_profile_routes(
    demo_cfg: Config, question: str, name: str, extra: dict[str, object],
) -> None:
    routed = fallback.route(demo_cfg, question)
    assert routed is not None and routed[0] == name
    for key, value in extra.items():
        assert routed[1][key] == value
    assert routed[1]["tournament"] == ("women" if "women" in question else "men")


def test_new_functions_have_strict_tool_schemas() -> None:
    tools = {tool["name"]: tool for tool in functions.TOOLS}
    expected = {"research_answer", "point_evidence", "how_to_play_against"}
    assert expected.issubset(functions.CATALOGUE) and expected.issubset(tools)
    for name in expected:
        schema = tools[name]["input_schema"]
        assert tools[name]["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_point_follow_up_keeps_the_point_and_ignores_a_stale_team(demo_cfg: Config) -> None:
    previous = {"function": "point_evidence", "point": "Efficiency", "team": "Team A1", "tournament": "men"}
    routed = fallback.route(demo_cfg, "What about the women?", previous)
    assert routed == ("point_evidence", {"point": "Efficiency", "tournament": "women"})
    previous["function"] = "research_answer"
    routed = fallback.route(demo_cfg, "What about the women?", previous)
    assert routed == ("research_answer", {"tournament": "women"})


def test_short_team_names_are_recognised(monkeypatch) -> None:
    from trivium.query import fallback

    teams = ["Republic of Ireland", "Panama", "Korea DPR", "Korea Republic", "United States"]
    monkeypatch.setattr(fallback, "_get_tournament_teams", lambda cfg, tournament: teams)
    assert fallback._extract_teams(None, "What is Ireland's tactical profile?", "men") == ["Republic of Ireland"]
    assert fallback._extract_teams(None, "How do we beat South Korea?", "men") == ["Korea Republic"]
    assert fallback._extract_teams(None, "Korea DPR against USA", "men") == ["Korea DPR", "United States"]
    assert fallback._extract_teams(None, "Show us the answer", "men") == []
