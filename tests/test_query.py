"""Tests for the query box: catalogue functions, tool schema, Claude path with fake client,
offline keyword routing accuracy, and the Streamlit Ask page.
"""

from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace
from typing import Any

import anthropic
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from trivium.config import Config
from trivium.measures import load_measures
from trivium.query import fallback, functions, llm
from trivium.query.functions import (
    FORBIDDEN_COLUMNS,
    MAX_ROWS,
    TOOLS,
    QueryError,
    Result,
    ensure_aggregated,
    run_function,
)
from trivium.synthetic import demo_config, write_demo


@pytest.fixture(scope="module")
def demo_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Module-scoped small synthetic demo build."""
    out = tmp_path_factory.mktemp("demo")
    write_demo(out, scale="small", luck_sims=200, progress_sims=200)
    old_env = os.environ.get("TRIVIUM_DATA_ROOT")
    os.environ["TRIVIUM_DATA_ROOT"] = str(out)
    cfg = demo_config(out)
    yield cfg
    if old_env is None:
        os.environ.pop("TRIVIUM_DATA_ROOT", None)
    else:
        os.environ["TRIVIUM_DATA_ROOT"] = old_env


# ---------------------------------------------------------------- catalogue tests


def test_catalogue_functions_run_and_ensure_aggregated(demo_cfg: Config) -> None:
    # 1. list_measures
    res = functions.list_measures(demo_cfg, tournament="men")
    assert res.available
    assert isinstance(res, Result)
    assert not res.table.empty
    ensure_aggregated(res)

    # 2. team_summary
    res = functions.team_summary(demo_cfg, team="Team A1", tournament="men")
    assert res.available
    assert res.numbers["team"] == "Team A1"
    assert res.chart is not None
    ensure_aggregated(res)

    # 3. rank_teams
    res = functions.rank_teams(demo_cfg, measure="overperformance", tournament="men", n=5, order="best")
    assert res.available
    assert len(res.table) == 5
    assert res.chart is not None
    ensure_aggregated(res)

    # 4. compare_teams
    res = functions.compare_teams(demo_cfg, team_a="Team A1", team_b="Team A2", tournament="men")
    assert res.available
    assert not res.table.empty
    assert res.chart is not None
    ensure_aggregated(res)

    # 5. explain_overperformance
    res = functions.explain_overperformance(demo_cfg, team="Team A1", tournament="men")
    assert res.available
    assert not res.table.empty
    assert res.chart is not None
    ensure_aggregated(res)

    # 6. match_review
    res = functions.match_review(demo_cfg, team="Team A1", opponent="Team A2", tournament="men")
    assert res.available
    assert not res.table.empty
    assert res.chart is not None
    ensure_aggregated(res)

    # 7. running_profile (men)
    res = functions.running_profile(demo_cfg, team="Team A1", tournament="men", position_group="Midfielder")
    assert res.available
    assert not res.table.empty
    ensure_aggregated(res)

    # 8. zone_map
    res = functions.zone_map(demo_cfg, team="Team A1", tournament="men")
    assert res.available
    assert len(res.table) == 18
    assert res.chart is not None
    ensure_aggregated(res)

    # 9. Conversion by phase reads team_tournament_phase, which the synthetic build now has
    res = functions.run_function(demo_cfg, "conversion_by_phase", {"team": "Team A1", "tournament": "men"})
    assert res.available and not res.table.empty
    ensure_aggregated(res)

    # 10-13. Functions whose tables need stages not run on this build return available=False gracefully
    for fn_name, args in [
        ("style_of", {"team": "Team A1", "tournament": "men"}),
        ("teams_by_style", {"style": "Direct", "tournament": "men"}),
        ("efficiency_index", {"tournament": "men", "n": 5}),
        ("compare_progressing_vs_exiting", {"tournament": "men"}),
    ]:
        res = functions.run_function(demo_cfg, fn_name, args)
        assert not res.available
        assert res.note != ""
        ensure_aggregated(res)


def test_missing_table_handling_on_existing_table(demo_cfg: Config, tmp_path: pytest.TempPathFactory) -> None:
    # Copy build to tmp_path and delete team_summary.parquet
    test_dir = tmp_path / "corrupted_build"
    shutil.copytree(demo_cfg.build_dir, test_dir)
    (test_dir / "team_summary.parquet").unlink()

    corrupt_cfg = demo_config(test_dir)
    res = functions.team_summary(corrupt_cfg, team="Team A1", tournament="men")
    assert not res.available
    assert "team_summary" in res.note
    assert res.table.empty


def test_women_asking_for_men_only_raises_query_error(demo_cfg: Config) -> None:
    with pytest.raises(QueryError, match="men's tournament only"):
        functions.running_profile(demo_cfg, team="Team A1", tournament="women")

    with pytest.raises(QueryError, match="only available for the men's tournament"):
        functions.rank_teams(demo_cfg, measure="distance_km", tournament="women")


def test_ensure_aggregated_violations() -> None:
    # Violation 1: > MAX_ROWS
    huge_df = pd.DataFrame({"x": range(MAX_ROWS + 1)})
    bad_res1 = Result(
        function="test", args={}, title="t", table=huge_df, numbers={}, chart=None, sources=(), available=True, note=""
    )
    with pytest.raises(ValueError, match="exceeding MAX_ROWS"):
        ensure_aggregated(bad_res1)

    # Violation 2: Forbidden column
    forbidden_df = pd.DataFrame({"person_id": [123, 456], "val": [1, 2]})
    bad_res2 = Result(
        function="test", args={}, title="t", table=forbidden_df, numbers={}, chart=None, sources=(), available=True, note=""
    )
    with pytest.raises(ValueError, match="Forbidden player-level columns"):
        ensure_aggregated(bad_res2)


# ---------------------------------------------------------------- TOOLS schema tests


def test_tools_schema_validation() -> None:
    assert len(TOOLS) == 24
    reg_measures = set(load_measures()["measure"].unique())

    from trivium.query.functions import MAX_STRICT_TOOLS, NON_STRICT_TOOLS

    assert sum(tool["strict"] for tool in TOOLS) <= MAX_STRICT_TOOLS
    for tool in TOOLS:
        assert tool["strict"] is (tool["name"] not in NON_STRICT_TOOLS)
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        props = schema["properties"]
        required = schema.get("required", [])
        for req in required:
            assert req in props

        # Verify rank_teams measure enum
        if tool["name"] == "rank_teams":
            tool_measures = set(props["measure"]["enum"])
            assert tool_measures == reg_measures


# ---------------------------------------------------------------- Fake Client & Claude Path tests


class FakeMessagesBeta:
    def __init__(self, responses: list[Any], calls_record: list[dict[str, Any]]):
        self._responses = responses
        self._calls_record = calls_record

    def create(self, **kwargs: Any) -> Any:
        self._calls_record.append(kwargs)
        if not self._responses:
            raise RuntimeError("No mock response queued")
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeAnthropicClient:
    def __init__(self, responses: list[Any]):
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=FakeMessagesBeta(responses, self.calls))

    def with_options(self, **kwargs: Any) -> FakeAnthropicClient:
        return self


def test_claude_tool_call_flow(demo_cfg: Config) -> None:
    # First response calls tool
    first_resp = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="thinking", thinking="Thinking about team summary"),
            SimpleNamespace(
                type="tool_use",
                id="call_abc123",
                name="team_summary",
                input={"team": "Team A1", "tournament": "men"},
            ),
        ],
    )
    # Second response writes explanation
    second_resp = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Team A1 gained +2.5 points above expectation.")]
    )

    fake_client = FakeAnthropicClient([first_resp, second_resp])
    ans = llm.answer(demo_cfg, "How did Team A1 perform?", client=fake_client, use_claude=True)

    assert ans.mode == "claude"
    assert ans.text == "Team A1 gained +2.5 points above expectation."
    assert ans.function == "team_summary"
    assert ans.result is not None
    assert ans.error == ""

    # Verify second call received the first assistant message unchanged
    assert len(fake_client.calls) == 2
    second_call_messages = fake_client.calls[1]["messages"]
    assert second_call_messages[1]["role"] == "assistant"
    assert second_call_messages[1]["content"] == first_resp.content
    # Same top-level system prompt and tools in both calls (keeps the thinking blocks valid);
    # the writer instructions are an appended system message.
    assert fake_client.calls[1]["system"] == fake_client.calls[0]["system"]
    assert fake_client.calls[1]["tools"] == fake_client.calls[0]["tools"]
    assert second_call_messages[-1] == {"role": "system", "content": llm.WRITER_SYSTEM}
    assert fake_client.calls[1]["tool_choice"] == {"type": "none"}

    # Verify no forbidden columns in payload sent to Claude
    tool_result_payload = second_call_messages[2]["content"][0]["content"]
    assert second_call_messages[2]["role"] == "user"
    payload_data = json.loads(tool_result_payload)
    for row in payload_data.get("rows", []):
        for col in row:
            assert col.lower() not in FORBIDDEN_COLUMNS


def test_claude_no_tool_call_falls_back(demo_cfg: Config) -> None:
    first_resp = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="I cannot help with that.")],
    )
    fake_client = FakeAnthropicClient([first_resp])
    ans = llm.answer(demo_cfg, "Tell me about Team A1", client=fake_client, use_claude=True)
    assert ans.mode == "offline"
    assert "No tool selected" in ans.error


def test_claude_refusal_falls_back(demo_cfg: Config) -> None:
    first_resp = SimpleNamespace(
        stop_reason="refusal",
        content=[SimpleNamespace(type="text", text="I refuse to answer.")],
    )
    fake_client = FakeAnthropicClient([first_resp])
    ans = llm.answer(demo_cfg, "Tell me about Team A1", client=fake_client, use_claude=True)
    assert ans.mode == "offline"
    assert "refused" in ans.error.lower()


_mock_req = SimpleNamespace()
_mock_resp401 = SimpleNamespace(request=_mock_req, status_code=401, headers={})
_mock_resp429 = SimpleNamespace(request=_mock_req, status_code=429, headers={})
_mock_resp500 = SimpleNamespace(request=_mock_req, status_code=500, headers={})


@pytest.mark.parametrize(
    "error_exc",
    [
        anthropic.AuthenticationError(message="bad key", response=_mock_resp401, body=None),
        anthropic.RateLimitError(message="rate limited", response=_mock_resp429, body=None),
        anthropic.APITimeoutError(request=_mock_req),
        anthropic.APIConnectionError(request=_mock_req),
        anthropic.APIStatusError(message="server error", response=_mock_resp500, body=None),
        anthropic.AnthropicError("generic SDK error"),
    ],
)
def test_claude_sdk_errors_fall_back(demo_cfg: Config, error_exc: Exception) -> None:
    fake_client = FakeAnthropicClient([error_exc])
    ans = llm.answer(demo_cfg, "How did Team A1 perform?", client=fake_client, use_claude=True)
    assert ans.mode == "offline"
    assert ans.error != ""


# ---------------------------------------------------------------- Offline Routing Tests


ROUTING_BENCHMARK = [
    # team_summary
    ("Team A1", "team_summary", {"team": "Team A1", "tournament": "men"}),
    ("Tell me about Team B1 in the women's tournament", "team_summary", {"team": "Team B1", "tournament": "women"}),
    ("Summary for Team C2", "team_summary", {"team": "Team C2", "tournament": "men"}),
    # rank_teams
    ("Which men's teams beat their strength most?", "rank_teams", {"measure": "overperformance", "tournament": "men", "order": "best"}),
    ("Top 5 teams by field tilt", "rank_teams", {"measure": "territory_share", "tournament": "men", "n": 5, "order": "best"}),
    ("Worst teams by conversion", "rank_teams", {"measure": "conversion", "tournament": "men", "order": "worst"}),
    ("Rank women teams by possession share", "rank_teams", {"measure": "possession_share", "tournament": "women"}),
    ("Which team has the highest security?", "rank_teams", {"measure": "security", "tournament": "men", "order": "highest"}),
    ("Who had the lowest non-penalty xg conceded?", "rank_teams", {"measure": "np_xg_against", "tournament": "men", "order": "lowest"}),
    ("Top 10 teams in non-penalty xg", "rank_teams", {"measure": "np_xg", "tournament": "men", "n": 10}),
    ("Rank teams by chances share", "rank_teams", {"measure": "chances_share", "tournament": "men"}),
    # compare_teams
    ("Compare Team A1 and Team B2", "compare_teams", {"team_a": "Team A1", "team_b": "Team B2", "tournament": "men"}),
    ("Team A1 vs Team B1", "compare_teams", {"team_a": "Team A1", "team_b": "Team B1", "tournament": "men"}),
    ("Compare Team A1 and Team A2 in women's tournament", "compare_teams", {"team_a": "Team A1", "team_b": "Team A2", "tournament": "women"}),
    # match_review
    ("Review the match between Team A1 and Team A2", "match_review", {"team": "Team A1", "opponent": "Team A2", "tournament": "men"}),
    ("What was the result when Team A1 played Team A2?", "match_review", {"team": "Team A1", "opponent": "Team A2", "tournament": "men"}),
    ("Head to head game Team A1 vs Team A2", "match_review", {"team": "Team A1", "opponent": "Team A2", "tournament": "men"}),
    # explain_overperformance
    ("Why did Team A1 over-perform?", "explain_overperformance", {"team": "Team A1", "tournament": "men"}),
    ("Explain why Team A2 beat expectation", "explain_overperformance", {"team": "Team A2", "tournament": "men"}),
    ("Was Team B1 overperformance due to luck?", "explain_overperformance", {"team": "Team B1", "tournament": "men"}),
    ("Why did Team A1 overperform in women's tournament?", "explain_overperformance", {"team": "Team A1", "tournament": "women"}),
    # running_profile
    ("What is the running profile of Team A1?", "running_profile", {"team": "Team A1", "tournament": "men"}),
    ("Distance and high-speed running for Team B2", "running_profile", {"team": "Team B2", "tournament": "men"}),
    ("Sprint distance for Team A1 Midfielders", "running_profile", {"team": "Team A1", "tournament": "men", "position_group": "Midfielder"}),
    # zone_map
    ("Show the zone territory map for Team A1", "zone_map", {"team": "Team A1", "tournament": "men"}),
    ("Where did Team B1 control the pitch?", "zone_map", {"team": "Team B1", "tournament": "men"}),
    ("Pitch heatmap for Team A1 against Team A2", "zone_map", {"team": "Team A1", "opponent": "Team A2", "tournament": "men"}),
    # style
    ("What is the style of Team A1?", "style_of", {"team": "Team A1", "tournament": "men"}),
    ("List teams with Direct Transition style", "teams_by_style", {"tournament": "men"}),
    # list_measures & others
    ("What can I ask?", "list_measures", {"tournament": "men"}),
    ("List available measures for women", "list_measures", {"tournament": "women"}),
    ("Show the efficiency index leaderboard", "efficiency_index", {"tournament": "men"}),
    ("Compare progressing vs exiting teams", "compare_progressing_vs_exiting", {"tournament": "men"}),
]


def test_offline_routing_benchmark(demo_cfg: Config) -> None:
    correct = 0
    total = len(ROUTING_BENCHMARK)

    for q, expected_fn, expected_args in ROUTING_BENCHMARK:
        routed = fallback.route(demo_cfg, q)
        if routed is not None:
            fn, args = routed
            if fn == expected_fn:
                match = True
                for k, v in expected_args.items():
                    if k in args and args[k] != v:
                        match = False
                if match:
                    correct += 1

    accuracy = correct / total
    assert accuracy >= 0.90, f"Routing accuracy {accuracy:.1%} is below 90% ({correct}/{total})"


def test_offline_follow_up_turns(demo_cfg: Config) -> None:
    # Turn 1: Team summary
    ans1 = fallback.offline_answer(demo_cfg, "Why did Team A1 over-perform?")
    assert ans1.function == "explain_overperformance"
    assert ans1.args["team"] == "Team A1"
    ctx = {"team": ans1.args["team"], "tournament": ans1.args["tournament"], "function": ans1.function}

    # Turn 2: Follow-up reusing team ("and their style?")
    ans2 = fallback.offline_answer(demo_cfg, "and their style?", previous_context=ctx)
    assert ans2.function == "style_of"
    assert ans2.args["team"] == "Team A1"
    ctx["function"] = ans2.function

    # Turn 3: Follow-up changing tournament ("what about the women?")
    ans3 = fallback.offline_answer(demo_cfg, "what about the women?", previous_context=ctx)
    assert ans3.function == "style_of"
    assert ans3.args["tournament"] == "women"


# ---------------------------------------------------------------- AppTest for 6_Ask.py


def test_app_ask_page_offline(demo_cfg: Config) -> None:
    from pathlib import Path

    app_path = Path(__file__).resolve().parents[1] / "app" / "pages" / "6_Ask.py"
    at = AppTest.from_file(str(app_path), default_timeout=15)
    at.run()
    assert not at.exception

    # Enter question
    at.text_input[0].input("Which men's teams beat their strength most?").run()
    assert not at.exception

    # Verify answers rendered
    markdown_texts = [m.value for m in at.markdown]
    has_answer = any("### Answer" in text for text in markdown_texts)
    assert has_answer
    assert len(at.dataframe) >= 1


def test_claude_empty_answer_falls_back(demo_cfg: Config) -> None:
    first_resp = SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", id="call_1", name="team_summary",
                                 input={"team": "Team A1", "tournament": "men"})],
    )
    second_resp = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="thinking", thinking="")])
    ans = llm.answer(demo_cfg, "How did Team A1 do?", client=FakeAnthropicClient([first_resp, second_resp]))
    assert ans.mode == "offline"
    assert "no answer text" in ans.error
    assert ans.text


@pytest.fixture(scope="module")
def styled_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    from trivium.models import styles

    cfg = write_demo(tmp_path_factory.mktemp("styled"), scale="small", luck_sims=200, progress_sims=200)
    styles.run(cfg)
    return cfg


def test_teams_by_style_keys_by_tournament_and_matches_words(styled_cfg: Config) -> None:
    from trivium.storage import read_table

    labels = read_table(styled_cfg, "style_types").query("tournament == 'men'")["style_label"].unique()
    target = next(label for label in labels if "high possession" in label)
    result = functions.teams_by_style(styled_cfg, "high possession", "men")
    assert "high possession" in result.args["style"]
    assert result.table["team"].is_unique
    expected = read_table(styled_cfg, "style_types").query("tournament == 'men' and style_label == @target")
    if result.args["style"] == target:
        assert len(result.table) == len(expected)


@pytest.mark.parametrize(
    "question",
    ["Which traits separate progressing teams from teams that went out?",
     "What did teams that progressed do differently from those knocked out?"],
)
def test_progressing_vs_exiting_routes(demo_cfg: Config, question: str) -> None:
    assert fallback.route(demo_cfg, question)[0] == "compare_progressing_vs_exiting"
