"""DuckDB over the build folder: an in-memory database with one view per Parquet table in
<build_dir> (top level only; per-match parts in sub-folders are not exposed). Nothing is
copied and the Parquet files are only read.
"""

from __future__ import annotations

import re

import duckdb
import pandas as pd

from three_trees.config import Config

TABLE_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")


def tables(cfg: Config) -> list[str]:
    """Names of the Parquet tables in the build folder, sorted."""
    if not cfg.build_dir.is_dir():
        return []
    return sorted(p.stem for p in cfg.build_dir.glob("*.parquet") if TABLE_NAME.match(p.stem))


def connect(cfg: Config) -> duckdb.DuckDBPyConnection:
    """In-memory connection with a view per table, named as the table."""
    con = duckdb.connect(":memory:")
    for name in tables(cfg):
        path = str(cfg.build_dir / f"{name}.parquet").replace("'", "''")
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{path}')")
    return con


def query(cfg: Config, sql: str, params: list | dict | None = None) -> pd.DataFrame:
    """Run one SQL statement against the build folder and return a DataFrame."""
    with connect(cfg) as con:
        return con.execute(sql, params or []).df()
