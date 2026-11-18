from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trivium.config import Config
from trivium.measures import load_measures
from trivium.query import fallback, functions
from trivium.storage import read_table, write_table
from trivium.synthetic import demo_config, write_demo


@pytest.fixture(scope="module")
def demo_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    return write_demo(
        tmp_path_factory.mktemp("query_round2"), scale="small", run_models=True,
        luck_sims=200, progress_sims=200,
    )


def _copy_tables(tmp_path: Path, cfg: Config, names: tuple[str, ...]) -> Config:
    result = demo_config(tmp_path)
    for name in names:
        write_table(result, name, read_table(cfg, name))
    return result


def _display(result: functions.Result) -> str:
    return result.table.to_string(index=False) + " " + fallback.template(result)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("xg_timeline", {"team": "Team A1", "opponent": "Team A2", "tournament": "men"}),
        ("fifa_phases", {"team": "Team A1"}),
        ("team_shape", {"team": "Team A1"}),
        ("findings", {"tournament": "men"}),
        ("senior_check", {}),
        ("data_coverage", {}),
    ],
)
def test_missing_round2_tables_give_plain_unavailable_answers(
    tmp_path: Path, name: str, args: dict[str, object],
) -> None:
    result = functions.run_function(demo_config(tmp_path), name, args)
    assert not result.available
    assert result.table.empty
    assert "not built" in fallback.template(result).lower()
    assert functions.ensure_aggregated(result) is result


def test_xg_timeline_has_both_team_totals_and_goal_times(demo_cfg: Config) -> None:
    shots = read_table(demo_cfg, "shot_timeline")
    matches = read_table(demo_cfg, "matches")
    match = matches[
        matches["tournament"].eq("men")
        & matches["home_team"].eq("Team A1") & matches["away_team"].eq("Team A2")
    ].iloc[0]
    expected = shots[shots["tournament"].eq("men") & shots["match_id"].eq(match["match_id"])]
    result = functions.xg_timeline(demo_cfg, "Team A1", "Team A2", "men")
    assert result.available and result.chart is not None
    assert functions.ensure_aggregated(result) is result
    rendered = _display(result)
    assert "Team A1" in rendered and "Team A2" in rendered
    text = fallback.template(result)
    for team in ("Team A1", "Team A2"):
        total = expected.loc[expected["team"].eq(team), "xg"].sum()
        assert f"{total:.2f}" in text
    goals = expected.loc[expected["goal"]]
    for minute in goals["minute"]:
        assert str(int(minute) + 1) in text
    assert len(result.table) <= functions.MAX_ROWS
    assert not set(result.table.columns).intersection(functions.FORBIDDEN_COLUMNS)


def test_match_review_dominance_chart_uses_requested_team_and_tournament(
    demo_cfg: Config, tmp_path: Path,
) -> None:
    cfg = _copy_tables(tmp_path, demo_cfg, ("matches", "expected_results", "team_match"))
    expected = read_table(cfg, "expected_results")
    expected.loc[expected["tournament"].eq("women"), "expected_points"] = 2.87
    write_table(cfg, "expected_results", expected)
    team_match = read_table(cfg, "team_match")
    team_match.loc[team_match["tournament"].eq("women"), "territory_share"] = 0.87
    write_table(cfg, "team_match", team_match)
    result = functions.match_review(cfg, "Team A2", "Team A1", "women")
    assert result.available and result.chart is not None
    assert result.numbers["expected_points_team"] == pytest.approx(2.87)
    bars = [trace for trace in result.chart.data if trace.type == "bar"]
    assert bars
    assert any(float(value) == pytest.approx(37.0) for trace in bars for value in trace.x)
    assert "Team A2" in str(result.chart.layout.xaxis.title.text)
    assert functions.ensure_aggregated(result) is result


def test_fifa_phases_report_registry_labels_and_tournament_comparison(demo_cfg: Config) -> None:
    result = functions.fifa_phases(demo_cfg, "Team A1")
    assert result.available and not result.table.empty
    assert functions.ensure_aggregated(result) is result
    registry = load_measures()
    phase = registry["measure"].str.startswith("fifa_") & registry["measure"].str.endswith("_share")
    labels = registry.loc[phase]
    displayed = result.table.to_string(index=False)
    assert any(label in displayed for label in labels["label"])
    assert "tournament" in displayed.lower()
    assert result.chart is not None


def test_team_shape_returns_only_the_three_without_ball_team_measures(demo_cfg: Config) -> None:
    result = functions.team_shape(demo_cfg, "Team A1")
    assert result.available and len(result.table) == 3
    assert functions.ensure_aggregated(result) is result
    displayed = result.table.to_string(index=False).lower()
    assert "line height" in displayed and "team length" in displayed and "team width" in displayed
    assert "tournament" in displayed
    assert result.chart is not None


def test_team_shape_fallback_filters_without_ball_and_weights_match_means(
    demo_cfg: Config, tmp_path: Path,
) -> None:
    cfg = _copy_tables(tmp_path, demo_cfg, ("matches",))
    teams = ["Team A1"] * 5 + ["Team A2"] * 2
    values = [10.0, 10.0, 10.0, 30.0, 1000.0, 50.0, 50.0]
    shape = pd.DataFrame({
        "tournament": "men", "match_id": [1, 1, 1, 2, 1, 1, 2], "team": teams,
        "state": ["without_ball"] * 4 + ["with_ball", "without_ball", "without_ball"],
        "line_height_m": values, "team_length_m": values, "team_width_m": values,
    })
    write_table(cfg, "shape_team_second", shape)
    result = functions.team_shape(cfg, "Team A1")
    assert result.available
    assert result.table["team_mean"].eq(20.0).all()
    assert result.table["tournament_mean"].eq(35.0).all()


def test_findings_filter_gate_and_tournament_and_return_plain_wording(tmp_path: Path) -> None:
    cfg = demo_config(tmp_path)
    gate = pd.DataFrame({
        "tournament": ["men", "men", "men", "women"],
        "measure": ["territory_share", "security", "conversion", "territory_share"],
        "label": ["Field tilt", "Security", "Conversion", "Field tilt"],
        "passes_gate": pd.Series([True, False, pd.NA, True], dtype="boolean"),
        "wording": ["Useful finding for men.", "Rejected finding.", "Unknown finding.", "Women finding."],
    })
    write_table(cfg, "finding_gate", gate)
    result = functions.findings(cfg, "men")
    assert result.available and len(result.table) == 1
    displayed = _display(result)
    assert "Useful finding for men." in displayed
    assert "Rejected finding" not in displayed and "Unknown finding" not in displayed
    assert "Women finding" not in displayed
    assert functions.ensure_aggregated(result) is result


def test_findings_empty_pass_set_is_an_available_null_result(tmp_path: Path) -> None:
    cfg = demo_config(tmp_path)
    write_table(cfg, "finding_gate", pd.DataFrame({
        "tournament": ["men"], "passes_gate": [False], "wording": ["Not established."],
    }))
    result = functions.findings(cfg, "men")
    assert result.available and result.table.empty
    assert "no" in fallback.template(result).lower()


def test_senior_check_summarises_editions_and_includes_optional_signifiers(tmp_path: Path) -> None:
    cfg = demo_config(tmp_path)
    write_table(cfg, "senior_backtest", pd.DataFrame({
        "tournament": ["men", "men", "women"], "year": [2018, 2022, 2019],
        "model_beats_base_rates": [1.0, 0.0, 1.0],
        "rps_model": [0.1, 0.3, 0.1], "rps_base_rates": [0.2, 0.2, 0.2],
    }))
    write_table(cfg, "checks_senior", pd.DataFrame({
        "check": ["senior men: editions where the model beats base rates (out-of-sample)"],
        "target": [2.0], "value": [1.0], "passed": [False],
    }))
    result = functions.senior_check(cfg)
    assert result.available
    text = fallback.template(result)
    assert "men" in text.lower() and "women" in text.lower()
    assert "1" in text and "2" in text
    assert functions.ensure_aggregated(result) is result
    write_table(cfg, "senior_signifiers", pd.DataFrame({
        "tournament": ["men"], "measure": ["territory_share"], "label": ["Field tilt"],
        "r_sb": [0.7], "passed_reliability": [True], "r_overperformance": [0.2],
        "r_low": [0.1], "r_high": [0.3], "excludes_zero": [True],
    }))
    with_signifiers = functions.senior_check(cfg)
    assert with_signifiers.available
    assert "senior_signifiers" in with_signifiers.sources
    assert functions.ensure_aggregated(with_signifiers) is with_signifiers


def test_data_coverage_uses_check_counts_and_preserves_string_targets(tmp_path: Path) -> None:
    cfg = demo_config(tmp_path)
    write_table(cfg, "checks_statsbomb", pd.DataFrame({
        "check": ["men: matches", "men: rows (events + shootout events)",
                  "women: matches", "women: rows (events + shootout events)"],
        "target": ["12", "1200", "6", "600"], "value": [12.0, 1200.0, 6.0, 600.0],
        "passed": [True] * 4,
    }))
    write_table(cfg, "checks_match_folders", pd.DataFrame({
        "check": ["folders with a physical CSV"], "target": ["11 of 12"],
        "value": [11.0], "passed": [False],
    }))
    write_table(cfg, "checks_tracking", pd.DataFrame({
        "check": ["folders with a tracking zip"], "target": ["10 of 12"],
        "value": [10.0], "passed": [False],
    }))
    result = functions.data_coverage(cfg)
    assert result.available and not result.table.empty
    assert functions.ensure_aggregated(result) is result
    text = _display(result)
    for count in ("1200", "600", "11", "10"):
        assert count in text
    assert "physical" in text.lower() and "tracking" in text.lower()


@pytest.mark.parametrize(
    ("question", "name"),
    [
        ("Show the xG timeline for Team A1 against Team A2", "xg_timeline"),
        ("FIFA phases of play for Team A1", "fifa_phases"),
        ("Team shape without the ball for Team A1", "team_shape"),
        ("What findings can we say for the women?", "findings"),
        ("Show the senior model check", "senior_check"),
        ("What is our data coverage?", "data_coverage"),
    ],
)
def test_round2_questions_route_to_catalogue(demo_cfg: Config, question: str, name: str) -> None:
    result = fallback.route(demo_cfg, question)
    assert result is not None and result[0] == name


def test_new_tool_schemas_match_catalogue() -> None:
    expected = {"xg_timeline", "fifa_phases", "team_shape", "findings", "senior_check", "data_coverage"}
    schemas = {tool["name"]: tool for tool in functions.TOOLS}
    assert expected.issubset(schemas) and expected.issubset(functions.CATALOGUE)
    for name in expected:
        schema = schemas[name]["input_schema"]
        assert schemas[name]["strict"] is (name not in functions.NON_STRICT_TOOLS)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_tactical_profile_and_profiles_that_win(demo_cfg: Config) -> None:
    from trivium.query import fallback, functions

    teams = functions.read_table(demo_cfg, "tactical_profile_team")
    team = teams.loc[teams["tournament"] == "men", "team"].iloc[0]
    result = functions.run_function(demo_cfg, "tactical_profile", {"team": team, "tournament": "men"})
    assert result.available and result.numbers["profile"] and len(result.table) == 3
    assert "points per match against its Elo expectation" in fallback.template(result)
    wins = functions.run_function(demo_cfg, "profiles_that_win", {"tournament": "men"})
    assert wins.available and wins.numbers["best_mean"] >= wins.numbers["worst_mean"]
    assert "did best against their strength" in fallback.template(wins)
    names = {tool["name"] for tool in functions.TOOLS}
    assert {"tactical_profile", "profiles_that_win"} <= names
