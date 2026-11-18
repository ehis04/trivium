"""Detect team formations from average player depth in tracking windows."""

from __future__ import annotations

import numpy as np
import pandas as pd


WINDOW_KEYS = ["tournament", "match_id", "team", "period", "window", "window_start_min", "state"]
FORMATION_WINDOW_COLUMNS = [
    *WINDOW_KEYS, "formation", "lines", "gvf", "players", "back_line_m", "front_line_m", "depth_m",
]
FORMATION_MATCH_COLUMNS = [
    "tournament", "match_id", "team", "state", "main_formation", "main_share", "formations_used", "changes",
    "back_line_m", "depth_m",
]
_WINDOW_GROUP = ["tournament", "match_id", "team", "period", "window", "window_start_min", "state"]
_MATCH_GROUP = ["tournament", "match_id", "team", "state"]
OUTFIELD_PLAYERS = 10


def _values(values: object) -> np.ndarray:
    """Convert finite one-dimensional depth values to a float array."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("Formation depths must be one-dimensional.")
    if not np.isfinite(data).all():
        raise ValueError("Formation depths must all be finite.")
    return data


def jenks_labels(values: object, n_classes: int) -> np.ndarray:
    """Return exact contiguous Jenks classes, numbered from deepest to most advanced."""
    data = _values(values)
    if isinstance(n_classes, bool) or not isinstance(n_classes, (int, np.integer)) or n_classes < 1:
        raise ValueError("n_classes must be a positive integer.")
    if data.size == 0:
        return np.empty(0, dtype=int)

    unique, inverse, counts = np.unique(data, return_inverse=True, return_counts=True)
    classes = min(int(n_classes), len(unique))
    if classes == 1:
        return np.zeros(data.size, dtype=int)

    weighted = unique * counts
    prefix_counts = np.r_[0, np.cumsum(counts)]
    prefix_sums = np.r_[0.0, np.cumsum(weighted)]
    prefix_squares = np.r_[0.0, np.cumsum(unique**2 * counts)]
    dp = np.full((classes + 1, len(unique) + 1), np.inf)
    previous = np.zeros((classes + 1, len(unique) + 1), dtype=int)
    dp[0, 0] = 0.0
    for n_groups in range(1, classes + 1):
        for end in range(n_groups, len(unique) + 1):
            starts = np.arange(n_groups - 1, end)
            segment_counts = prefix_counts[end] - prefix_counts[starts]
            segment_sums = prefix_sums[end] - prefix_sums[starts]
            segment_squares = prefix_squares[end] - prefix_squares[starts]
            within = np.maximum(0.0, segment_squares - segment_sums**2 / segment_counts)
            candidates = dp[n_groups - 1, starts] + within
            best = int(np.argmin(candidates))
            dp[n_groups, end] = candidates[best]
            previous[n_groups, end] = starts[best]

    cuts = []
    end = len(unique)
    for n_groups in range(classes, 1, -1):
        end = int(previous[n_groups, end])
        cuts.append(end)
    bounds = (0, *reversed(cuts), len(unique))
    unique_labels = np.empty(len(unique), dtype=int)
    for label, (left, right) in enumerate(zip(bounds, bounds[1:])):
        unique_labels[left:right] = label
    return unique_labels[inverse]


def gvf(values: object, labels: object) -> float:
    """Return the Jenks goodness of variance fit for the supplied depth classes."""
    data = _values(values)
    groups = np.asarray(labels)
    if groups.ndim != 1 or groups.size != data.size:
        raise ValueError("Formation labels must be one-dimensional and match the depth values.")
    if data.size == 0:
        return 0.0
    sdam = float(np.sum((data - data.mean()) ** 2))
    if sdam == 0:
        return 0.0
    sdcm = sum(float(np.sum(((part := data[groups == label]) - part.mean()) ** 2))
               for label in pd.unique(groups))
    return (sdam - sdcm) / sdam


def best_line_count(values: object, choices: tuple[int, ...] = (3, 4, 5), min_gain: float = 0.04) -> int:
    """Choose line count by the notebook's GVF step rule and small-input rules."""
    data = _values(values)
    if data.size == 0:
        return 0
    unique_count = np.unique(data).size
    if unique_count <= 3:
        return int(unique_count)
    if data.size < 6:
        return 3
    options = tuple(sorted(set(choices)))
    if not options or options[0] < 1:
        raise ValueError("choices must contain positive line counts.")
    baseline = options[0]
    previous = gvf(data, jenks_labels(data, baseline))
    best = baseline
    for count in options[1:]:
        score = gvf(data, jenks_labels(data, count))
        if score - previous >= min_gain:
            best, previous = count, score
        else:
            break
    return min(best, unique_count)


def formation_string(values: object) -> str:
    """Return the player counts in each depth line, from deepest to most advanced."""
    data = _values(values)
    if data.size == 0:
        return ""
    labels = jenks_labels(data, best_line_count(data))
    return "-".join(str(int(np.count_nonzero(labels == label))) for label in np.unique(labels))


def _formation_details(depths: np.ndarray) -> tuple[str, int, float, float, float]:
    lines = best_line_count(depths)
    labels = jenks_labels(depths, lines)
    groups = np.unique(labels)
    counts = [int(np.count_nonzero(labels == label)) for label in groups]
    means = np.array([depths[labels == label].mean() for label in groups], dtype=float)
    back, front = float(means[0]), float(means[-1])
    return "-".join(map(str, counts)), lines, gvf(depths, labels), back, front - back


def team_window_formations(
    positions_window: pd.DataFrame, min_players: int = 10, min_seconds: float = 60,
) -> pd.DataFrame:
    """Detect one team formation per possession-state window, dropping short or understrength samples."""
    if min_players < 1:
        raise ValueError("min_players must be positive.")
    if min_seconds < 0:
        raise ValueError("min_seconds must not be negative.")
    required = {*_WINDOW_GROUP, "person_id", "x_m", "seconds"}
    missing = required - set(positions_window.columns)
    if missing:
        raise ValueError(f"positions_window is missing columns: {', '.join(sorted(missing))}.")
    rows = []
    usable = positions_window.loc[
        (pd.to_numeric(positions_window["seconds"], errors="coerce") >= min_seconds)
        & pd.to_numeric(positions_window["x_m"], errors="coerce").notna()
    ]
    for keys, group in usable.groupby(_WINDOW_GROUP, sort=False, dropna=False):
        # A substitution inside the window can leave more than ten outfield players: keep the ten with most time.
        players = (group.sort_values("seconds", ascending=False, kind="stable")
                   .drop_duplicates("person_id", keep="first").head(OUTFIELD_PLAYERS))
        depths = players["x_m"].to_numpy(dtype=float)
        if len(depths) < min_players:
            continue
        formation, lines, score, back, depth = _formation_details(depths)
        rows.append(dict(zip(_WINDOW_GROUP, keys)) | {
            "formation": formation,
            "lines": lines,
            "gvf": score,
            "players": len(depths),
            "back_line_m": back,
            "front_line_m": back + depth,
            "depth_m": depth,
        })
    return pd.DataFrame(rows, columns=FORMATION_WINDOW_COLUMNS)


def team_match_formations(formation_team_window: pd.DataFrame) -> pd.DataFrame:
    """Summarise each team's most-used shape and shape changes over the two periods."""
    required = {*_WINDOW_GROUP, "formation", "back_line_m", "depth_m"}
    missing = required - set(formation_team_window.columns)
    if missing:
        raise ValueError(f"formation_team_window is missing columns: {', '.join(sorted(missing))}.")
    rows = []
    ordered = formation_team_window.sort_values(["period", "window", "window_start_min"], kind="stable")
    for keys, group in ordered.groupby(_MATCH_GROUP, sort=False, dropna=False):
        first_seen = group["formation"].drop_duplicates().tolist()
        counts = group["formation"].value_counts()
        main = next(shape for shape in first_seen if counts[shape] == counts.max())
        changes = sum(
            int(part["formation"].ne(part["formation"].shift()).iloc[1:].sum())
            for _, part in group.groupby("period", sort=False)
        )
        rows.append(dict(zip(_MATCH_GROUP, keys)) | {
            "main_formation": main,
            "main_share": float(counts[main] / len(group)),
            "formations_used": int(group["formation"].nunique()),
            "changes": changes,
            "back_line_m": float(group["back_line_m"].mean()),
            "depth_m": float(group["depth_m"].mean()),
        })
    return pd.DataFrame(rows, columns=FORMATION_MATCH_COLUMNS)
