"""Shared page setup, build-status banner and plain-English number formatting."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.data import load
from three_trees.config import load_env_file


def page_setup(title: str) -> None:
    """Set the wide page layout, title and synthetic-data notice when appropriate; read the local .env
    (ANTHROPIC_API_KEY for the Ask page)."""
    st.set_page_config(page_title=f"{title} | Pathways", layout="wide")
    load_env_file()
    if st.session_state.get("presenter_mode", False):
        st.markdown(
            "<style>html {font-size: 1.15rem;} [data-testid='stDataFrame'] {display: none;}</style>",
            unsafe_allow_html=True,
        )
    st.title(title)
    build_info = load("build_info")
    if build_info is not None and not build_info.empty and "source" in build_info:
        if build_info["source"].eq("synthetic").all():
            st.warning("Synthetic demo data: fake teams and random values, not real results.")


def show_table(table: pd.DataFrame, **kwargs: object) -> None:
    """Display a table unless presenter mode is hiding detail for the projector."""
    if not st.session_state.get("presenter_mode", False):
        st.dataframe(table, **kwargs)


def format_share(value: object, decimals: int = 0) -> str:
    """Format a fractional share as a percentage."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "Not available" if pd.isna(number) else f"{number:.{decimals}%}"


def format_points(value: object) -> str:
    """Format points to two decimal places."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "Not available" if pd.isna(number) else f"{number:.2f}"


def format_elo(value: object) -> str:
    """Format an Elo rating as an integer."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "Not available" if pd.isna(number) else f"{number:.0f}"


def stage_name(value: object) -> str:
    """Turn a progression-stage key into a coach-readable label."""
    labels = {
        "group": "Group stage",
        "round_of_32": "Round of 32",
        "round_of_16": "Round of 16",
        "quarter_final": "Quarter-final",
        "semi_final": "Semi-final",
        "final": "Final",
        "champion": "Champion",
    }
    return labels.get(str(value), str(value).replace("_", " ").title())


def expected_stage_label(value: object, tournament: str) -> str:
    """Describe the nearest stage to an expected mean stage index."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return "Not available"
    stages = [
        "Group stage",
        *(["Round of 32"] if tournament == "men" else []),
        "Round of 16",
        "Quarter-final",
        "Semi-final",
        "Final",
        "Champion",
    ]
    index = min(max(int(round(float(number))), 0), len(stages) - 1)
    return f"{stages[index]} (mean index {number:.2f})"


def display_label(label: str, unit: str, higher_is_better: object) -> str:
    """Include registry units and an explicit direction when one is defined."""
    direction = " · lower is better" if pd.notna(higher_is_better) and not bool(higher_is_better) else ""
    unit_text = f" ({unit})" if unit and unit != "share" else ""
    return f"{label}{unit_text}{direction}"
