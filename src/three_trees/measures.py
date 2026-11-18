"""Measure registry: every team-match measure used by the models, dashboard and query box,
read from lookup/measures.csv (name, label, unit, direction, source table and column, tier,
group, signifier number, plain-English definition).

Code refers to measures by `measure`; the `column` field maps a measure to its column in the
source table, so a renamed column is a one-line change in the CSV.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files

import pandas as pd

REGISTRY_COLUMNS = [
    "measure", "label", "unit", "higher_is_better", "table", "column", "tier", "group", "signifier", "definition",
]
TIERS = ("both", "men")


@cache
def _registry() -> pd.DataFrame:
    with (files("three_trees") / "lookup" / "measures.csv").open(encoding="utf-8", newline="") as f:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
    df["higher_is_better"] = df["higher_is_better"].map({"true": True, "false": False, "": None})
    df["signifier"] = pd.to_numeric(df["signifier"].replace("", None), errors="coerce").astype("Int64")
    return df[REGISTRY_COLUMNS]


def load_measures() -> pd.DataFrame:
    """The whole registry (a copy)."""
    return _registry().copy()


def measures(
    group: str | None = None,
    tournament: str | None = None,
    table: str | None = None,
    signifiers_only: bool = False,
) -> pd.DataFrame:
    """Registry rows filtered by group, by what exists for a tournament (women: tier "both" only),
    by source table, or to the six signifiers."""
    df = load_measures()
    if group is not None:
        df = df[df["group"] == group]
    if tournament == "women":
        df = df[df["tier"] == "both"]
    if table is not None:
        df = df[df["table"] == table]
    if signifiers_only:
        df = df[df["signifier"].notna()]
    return df.reset_index(drop=True)


def measure(name: str) -> pd.Series:
    df = _registry()
    row = df[df["measure"] == name]
    if row.empty:
        raise ValueError(f"Unknown measure: {name!r}.")
    return row.iloc[0].copy()


def label(name: str) -> str:
    return str(measure(name)["label"])


def available(table: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Registry rows whose column exists in `table` with at least one value."""
    present = [c for c in registry["column"] if c in table.columns and table[c].notna().any()]
    return registry[registry["column"].isin(present)].reset_index(drop=True)
