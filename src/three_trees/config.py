"""Configuration: data paths from config.yaml. Every data path must be outside the repo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_ENV = "THREE_TREES_CONFIG"
DATA_ROOT_ENV = "THREE_TREES_DATA_ROOT"
DEFAULT_CONFIG = "config.yaml"
TOURNAMENTS = ("men", "women")
EXTERNAL_URLS = ("results_men", "results_women", "statsbomb_open_data")


class ConfigError(ValueError):
    """Config file missing, incomplete, or pointing inside the repo."""


@dataclass(frozen=True)
class StatsBombPaths:
    events: Path
    matches: Path

    def event_files(self) -> list[Path]:
        """Event part files matching the glob, in name order."""
        return sorted(self.events.parent.glob(self.events.name))


@dataclass(frozen=True)
class Config:
    source: Path
    data_root: Path
    build_dir: Path
    statsbomb: dict[str, StatsBombPaths]
    match_folders_men: Path
    fifa_reports_women: Path
    external_dir: Path
    external_urls: dict[str, str]

    def data_paths(self) -> dict[str, Path]:
        """Every configured local path, keyed by its config name."""
        paths = {"data_root": self.data_root, "build_dir": self.build_dir}
        for tournament, sb in self.statsbomb.items():
            paths[f"statsbomb.{tournament}.events"] = sb.events
            paths[f"statsbomb.{tournament}.matches"] = sb.matches
        paths["match_folders.men"] = self.match_folders_men
        paths["fifa_reports.women"] = self.fifa_reports_women
        paths["external.dir"] = self.external_dir
        return paths


def repo_root() -> Path | None:
    """The git work tree holding this package, or None when installed elsewhere."""
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def load_env_file(path: str | Path | None = None) -> list[str]:
    """KEY=VALUE lines of the local .env (default: <repo>/.env, git-ignored) put into the environment;
    a variable already set in the environment is kept. Returns the names set."""
    env_path = Path(path) if path is not None else (repo_root() or Path.cwd()) / ".env"
    if not env_path.is_file():
        return []
    loaded = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_config(path: str | Path | None = None) -> Config:
    """Read the config file from `path`, else $THREE_TREES_CONFIG, else ./config.yaml."""
    source = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(
            f"No config file at {source}. Copy config.example.yaml to config.yaml and set data_root."
        )
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must be a YAML mapping.")

    data_root = _absolute(os.environ.get(DATA_ROOT_ENV) or _get(raw, "data_root"), source.parent)

    def local(key: str) -> Path:
        return _absolute(_get(raw, key), data_root)

    cfg = Config(
        source=source,
        data_root=data_root,
        build_dir=local("build_dir"),
        statsbomb={
            t: StatsBombPaths(
                events=local(f"statsbomb.{t}.events"),
                matches=local(f"statsbomb.{t}.matches"),
            )
            for t in TOURNAMENTS
        },
        match_folders_men=local("match_folders.men"),
        fifa_reports_women=local("fifa_reports.women"),
        external_dir=local("external.dir"),
        external_urls={name: str(_get(raw, f"external.{name}")) for name in EXTERNAL_URLS},
    )
    _check_outside_repo(cfg)
    return cfg


def _get(raw: dict[str, Any], dotted: str) -> Any:
    value: Any = raw
    for key in dotted.split("."):
        if not isinstance(value, dict) or value.get(key) in (None, ""):
            raise ConfigError(f"Config key '{dotted}' is missing.")
        value = value[key]
    return value


def _absolute(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _check_outside_repo(cfg: Config) -> None:
    root = repo_root()
    if root is None:
        return
    inside = [f"{name} ({p})" for name, p in cfg.data_paths().items() if p.is_relative_to(root)]
    if inside:
        raise ConfigError(
            f"Data paths must be outside the repo ({root}): " + ", ".join(inside)
        )
