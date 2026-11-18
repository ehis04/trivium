"""Team names: StatsBomb names without age-group suffixes, and their names in the senior results."""

from __future__ import annotations

import csv
from functools import cache
from importlib.resources import files

SUFFIX_WORDS = frozenset({"U17", "W"})


def clean_team_name(name: str) -> str:
    """'Qatar U17' -> 'Qatar'. Also 'Spain U17 W', 'Canada W U17' and 'Colombia U17<tab>W'."""
    words = name.split()
    while words and words[-1] in SUFFIX_WORDS:
        words.pop()
    return " ".join(words)


@cache
def _aliases() -> dict[str, str]:
    with (files("three_trees") / "lookup" / "countries.csv").open(encoding="utf-8", newline="") as f:
        return {row["name"]: row["elo_name"] for row in csv.DictReader(f)}


def elo_name(name: str) -> str:
    """The team's name in the martj42 senior international results."""
    clean = clean_team_name(name)
    return _aliases().get(clean, clean)
