"""StatsBomb listed formations and their playing-time summaries."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from three_trees.config import Config
from three_trees.ingest.statsbomb import parse_literal

FORMATION_LISTED_COLUMNS = [
    "tournament", "match_id", "team", "starting_formation", "main_formation", "formation_changes", "back_line",
]
FORMATION_EVENTS = frozenset({"Starting XI", "Tactical Shift"})


def _event_name(value: Any) -> Any:
    return value.get("name") if isinstance(value, dict) else value


def _formation(value: Any) -> str | None:
    """Convert StatsBomb's numeric formation code to a hyphenated shape."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        text = str(value)
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip().replace(" ", "")
    if re.fullmatch(r"\d{3,5}", text):
        return "-".join(text)
    parts = text.split("-")
    return text if len(parts) >= 3 and all(part.isdigit() for part in parts) else None


def _tactics_formation(value: Any) -> str | None:
    tactics = value if isinstance(value, dict) else parse_literal(value)
    return _formation(tactics.get("formation")) if isinstance(tactics, dict) else None


def listed_formations(events: pd.DataFrame) -> pd.DataFrame:
    """Return each team's starting shape, most-used shape, changes and back line."""
    required = {"tournament", "match_id", "team", "period", "period_seconds", "type", "tactics"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events is missing columns: {', '.join(sorted(missing))}.")

    match_period_ends = (
        events.loc[events["period"].isin([1, 2])]
        .assign(_seconds=pd.to_numeric(events["period_seconds"], errors="coerce"))
        .groupby(["tournament", "match_id", "period"], dropna=False)["_seconds"].max()
    )
    rows = []
    for (tournament, match_id, team), team_events in events.groupby(
        ["tournament", "match_id", "team"], sort=False, dropna=False
    ):
        actions = []
        starting = None
        filtered = team_events.loc[team_events["period"].isin([1, 2])].copy()
        filtered["seconds_for_formation"] = pd.to_numeric(
            filtered["period_seconds"], errors="coerce"
        ).fillna(0.0)
        filtered["formation_event_name"] = filtered["type"].map(_event_name)
        filtered = filtered.loc[filtered["formation_event_name"].isin(FORMATION_EVENTS)]
        for order, (_, event) in enumerate(filtered.sort_values(
            ["period", "seconds_for_formation"], kind="stable"
        ).iterrows()):
            shape = _tactics_formation(event["tactics"])
            if shape is None:
                continue
            kind = event["formation_event_name"]
            period = int(event["period"])
            second = float(event["seconds_for_formation"])
            actions.append((period, second, order, kind, shape))
            if kind == "Starting XI" and starting is None:
                starting = shape

        current = None
        changes = 0
        durations: dict[str, float] = defaultdict(float)
        first_seen: list[str] = []
        for period in (1, 2):
            period_actions = [action for action in actions if action[0] == period]
            if current is not None and (not period_actions or period_actions[0][1] > 0):
                period_actions.insert(0, (period, 0.0, -1, "Carry", current))
            period_end = match_period_ends.get((tournament, match_id, period), 2700.0)
            period_end = 2700.0 if pd.isna(period_end) else max(0.0, float(period_end))
            for index, (_, second, _, kind, shape) in enumerate(period_actions):
                if kind == "Tactical Shift" and current is not None and shape != current:
                    changes += 1
                current = shape
                if shape not in first_seen:
                    first_seen.append(shape)
                end = period_actions[index + 1][1] if index + 1 < len(period_actions) else period_end
                durations[shape] += max(0.0, end - second)

        main = max(first_seen, key=lambda shape: durations[shape]) if first_seen else None
        rows.append({
            "tournament": tournament,
            "match_id": match_id,
            "team": team,
            "starting_formation": starting,
            "main_formation": main,
            "formation_changes": changes,
            "back_line": f"back {main.split('-')[0]}" if main else None,
        })
    return pd.DataFrame(rows, columns=FORMATION_LISTED_COLUMNS)


def build_listed_formations(cfg: Config) -> pd.DataFrame:
    """Build and write formation_listed from the already-ingested StatsBomb events."""
    from three_trees.storage import read_table, write_table

    result = listed_formations(read_table(cfg, "events"))
    write_table(cfg, "formation_listed", result)
    print(f"formation_listed: {len(result):,} team-matches")
    return result
