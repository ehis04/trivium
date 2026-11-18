"""Cached access to finished Parquet tables and user-facing missing-table notices."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from three_trees.config import load_config
from three_trees.storage import read_table, table_path


@st.cache_data(show_spinner=False)
def _read_cached(build_dir: str, name: str, modified_ns: int) -> pd.DataFrame:
    cfg = load_config()
    if str(cfg.build_dir) != build_dir:
        raise RuntimeError("The configured build folder changed while loading a table.")
    return read_table(cfg, name)


def load(name: str) -> pd.DataFrame | None:
    """Load one built table; return None if it does not exist in this build."""
    cfg = load_config()
    path = table_path(cfg, name)
    if not path.is_file():
        return None
    modified_ns = path.stat().st_mtime_ns
    return _read_cached(str(cfg.build_dir), name, modified_ns)


def require(name: str, built_by: str) -> pd.DataFrame | None:
    """Load a required table or show which pipeline stage must create it."""
    table = load(name)
    if table is None:
        st.info(f"Not built yet: needs table `{name}` from the `{built_by}` stage.")
    return table
