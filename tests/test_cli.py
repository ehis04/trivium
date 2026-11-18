from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from three_trees import cli
from three_trees.config import DATA_ROOT_ENV

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.yaml"


@pytest.mark.parametrize(
    "module", [m for _, m in cli.STAGES] + [cli.VALIDATION, cli.SENIOR_CHECK]
)
def test_entry_modules_expose_run(module: str) -> None:
    assert callable(importlib.import_module(module).run)


def test_check_config_counts_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    statsbomb = tmp_path / "statsbomb"
    statsbomb.mkdir()
    for n in range(1, 14):
        (statsbomb / f"u17_world_cup_2025_events_part{n:03d}.csv").touch()
    (statsbomb / "u17_world_cup_2025_matches.csv").touch()
    (tmp_path / "match_folders" / "M1 Qatar - Italy").mkdir(parents=True)

    code = cli.main(["check-config", "--config", str(EXAMPLE)])
    out = capsys.readouterr().out

    assert "13 event files" in out
    assert "0 event files" in out
    assert "1 match folders" in out
    assert code == 1


@pytest.fixture
def ran(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    modules: list[str] = []
    monkeypatch.setattr(cli, "_run", lambda module, cfg: modules.append(module))
    return modules


def test_build_runs_every_stage_then_validates(tmp_path: Path, ran: list[str]) -> None:
    assert cli.main(["build", "--config", str(EXAMPLE)]) == 0
    assert ran == [m for _, m in cli.STAGES] + [cli.VALIDATION]
    assert (tmp_path / "build").is_dir()


def test_build_only_keeps_pipeline_order_and_skips_validation(ran: list[str]) -> None:
    assert cli.main(["build", "--only", "features", "statsbomb", "--config", str(EXAMPLE)]) == 0
    assert ran == ["three_trees.ingest.statsbomb", "three_trees.features.tables"]


def test_unbuilt_stage_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))

    ran: list[str] = []

    def unbuilt(module: str, cfg: object) -> None:
        if module == "three_trees.tracking.stream":
            raise NotImplementedError("Tracking")
        ran.append(module)

    monkeypatch.setattr(cli, "_run", unbuilt)
    assert cli.main(["build", "--config", str(EXAMPLE)]) == 1
    assert "Not built yet: Tracking (tracking stage skipped)" in capsys.readouterr().err
    assert ran[-1] == cli.VALIDATION and "three_trees.models.sustainability" in ran


def test_config_error_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["validate", "--config", str(tmp_path / "missing.yaml")])
    assert code == 2
    assert "config.example.yaml" in capsys.readouterr().err
