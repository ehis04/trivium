"""Table storage: one Parquet file per table in the build folder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from three_trees.config import Config


def table_path(cfg: Config, name: str) -> Path:
    """<build_dir>/<name>.parquet; a name may hold folders, e.g. 'tracking/M01/running'."""
    return cfg.build_dir / f"{name}.parquet"


def write_table(cfg: Config, name: str, df: pd.DataFrame) -> Path:
    """Write the table whole or not at all: a failed write leaves any previous file in place."""
    path = table_path(cfg, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def read_table(cfg: Config, name: str) -> pd.DataFrame:
    path = table_path(cfg, name)
    if not path.is_file():
        raise FileNotFoundError(f"Table '{name}' is not built yet ({path}).")
    return pd.read_parquet(path)


def table_exists(cfg: Config, name: str) -> bool:
    return table_path(cfg, name).is_file()
