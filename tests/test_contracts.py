from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trivium import cli, contracts, measures, synthetic
from trivium.storage import read_table, write_table


def _contract() -> contracts.Contract:
    return contracts.Contract("demo", ("tournament", "match_id", "team", "x"), ("tournament", "match_id", "team"),
                              ("men", "women"), "Person 2", "demo")


def _table() -> pd.DataFrame:
    return pd.DataFrame({"tournament": ["men", "women"], "match_id": [1, 1], "team": ["A", "A"], "x": [1.0, 2.0]})


def test_check_table_passes_and_reports_extra_columns() -> None:
    rows = pd.DataFrame(contracts.check_table(_table().assign(extra=1), _contract()))
    assert rows["passed"].dropna().all()
    assert rows["check"].str.contains("extra columns").any()


def test_check_table_failures() -> None:
    missing = pd.DataFrame(contracts.check_table(_table().drop(columns="x"), _contract()))
    assert "missing: x" in missing.loc[~missing["passed"].fillna(True).astype(bool), "check"].iloc[0]
    duplicated = pd.DataFrame(contracts.check_table(pd.concat([_table(), _table()]), _contract()))
    assert not duplicated.set_index("check").loc["demo: one row per tournament, match_id, team", "passed"]
    empty_keys = _table().assign(match_id=[None, None], tournament=["men", "men"])
    empty = pd.DataFrame(contracts.check_table(empty_keys, _contract())).set_index("check")
    assert empty.loc["demo: one row per tournament, match_id, team", "passed"]
    men_only = pd.DataFrame(contracts.check_table(_table().iloc[:1], _contract()))
    assert not men_only.set_index("check").loc["demo: tournaments", "passed"]


def test_tables_without_grain_or_tournament_skip_those_checks() -> None:
    contract = contracts.Contract("demo", ("team",), (), (), "Person 2", "demo")
    rows = pd.DataFrame(contracts.check_table(pd.DataFrame({"team": ["A", "A"]}), contract))
    assert not rows["check"].str.contains("one row per|tournaments").any()


def test_team_match_contract_follows_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = measures.load_measures()
    extra = reg.iloc[[0]].assign(measure="new_measure", column="new_column")
    monkeypatch.setattr(contracts, "measures", lambda **kw: pd.concat([reg, extra])[lambda d: d["table"] == "team_match"])
    assert "new_column" in contracts.team_match_contract().columns


def test_not_built_and_unknown_tables(tmp_path: Path) -> None:
    cfg = synthetic.demo_config(tmp_path)
    checks = contracts.check_contracts(cfg, ["events"])
    assert len(checks) == 1 and "not built yet (Person 1, statsbomb stage)" in checks["check"].iloc[0]
    assert pd.isna(checks["passed"].iloc[0])
    with pytest.raises(ValueError, match="no_such_table"):
        contracts.check_contracts(cfg, ["no_such_table"])


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory):
    return synthetic.write_demo(tmp_path_factory.mktemp("demo"), scale="small", luck_sims=200, progress_sims=200,
                                run_models=True)


def test_every_built_table_meets_its_contract(build) -> None:
    checks = contracts.check_contracts(build)
    failures = contracts.contract_failures(checks)
    assert failures.empty, failures["check"].tolist()
    not_built = checks[checks["check"].str.contains("not built yet")]["check"].str.split(":").str[0]
    assert set(not_built) == {"events", "shootouts", "senior_events", "senior_matches", "physical_player_match",
                              "sync_offsets"}


def test_measure_coverage_per_tournament(build) -> None:
    cov = contracts.measure_coverage(build)
    assert not ((cov["tournament"] == "women") & (cov["tier"] == "men")).any()
    assert set(cov.loc[cov["tier"] == "men", "status"]) == {"live"}
    first_match_only = {"starters_changed", "rest_days"}
    assert set(cov.loc[cov["status"] != "live", "measure"]) <= first_match_only


def test_measure_coverage_missing_column(tmp_path: Path) -> None:
    cfg = synthetic.write_demo(tmp_path, scale="small", run_expectation=False)
    write_table(cfg, "team_match", read_table(cfg, "team_match").drop(columns="security"))
    cov = contracts.measure_coverage(cfg).set_index(["measure", "tournament"])
    assert cov.loc[("security", "men"), "status"] == "missing"
    assert not cov.loc[("security", "men"), "present"]


def test_build_status(build) -> None:
    status = contracts.build_status(build).set_index("table")
    assert status["synthetic"].all()
    assert status.loc["team_match", "tournaments"] == "men, women"
    assert status.loc["team_match", "in_contracts"] and not status.loc["build_info", "in_contracts"]


def test_cli_check_contracts_and_status(build, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("TRIVIUM_DATA_ROOT", str(build.data_root))
    config = str(Path(__file__).resolve().parents[1] / "config.example.yaml")
    assert cli.main(["check-contracts", "--config", config]) == 0
    assert "0 failed checks" in capsys.readouterr().out
    assert cli.main(["status", "--config", config]) == 0
    out = capsys.readouterr().out
    assert "men:" in out and "measures live" in out

    tm = read_table(build, "team_match")
    write_table(build, "team_match", pd.concat([tm, tm.iloc[:1]], ignore_index=True))
    assert cli.main(["check-contracts", "--config", config, "--table", "team_match"]) == 1
    assert "one row per tournament, match_id, team" in capsys.readouterr().out
    write_table(build, "team_match", tm)
    assert np.isfinite(tm.select_dtypes("number").to_numpy(dtype=float, na_value=0)).all()
