"""Style types from team means of registry style measures and field tilt.
Diagonal Gaussian mixtures with three or four components are compared by
BIC, with silhouette reported separately. Features are standardised across
team profiles; missing profile values use that feature's tournament mean.
The groups describe style and do not imply a cause of over-performance.

Output tables: style_types, style_profiles, style_overperformance,
style_selection, checks_styles.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from trivium import measures as registry
from trivium.config import Config
from trivium.models.uncertainty import bootstrap_by_team
from trivium.storage import read_table, write_table

SEED = 20251103
K_RANGE = (3, 4)
N_INIT = 20
N_BOOT = 2_000
TYPE_COLUMNS = ["tournament", "team", "style_id", "style_label", "probability"]
PROFILE_COLUMNS = ["tournament", "style_id", "style_label", "n_teams"]
SELECTION_COLUMNS = ["tournament", "k", "bic", "silhouette", "chosen"]
OVERPERFORMANCE_COLUMNS = [
    "tournament", "style_id", "style_label", "n_teams", "overperformance_per_match", "low", "high",
    "excludes_zero",
]
CHECK_COLUMNS = ["check", "target", "value", "passed"]


def _style_measures(tournament: str) -> pd.DataFrame:
    """Return the allowed style registry rows, including the field-tilt measure."""
    candidates = registry.measures(tournament=tournament, table="team_match")
    return candidates.loc[
        candidates["group"].eq("style") | candidates["measure"].eq("territory_share")
    ].reset_index(drop=True)


def team_profiles(team_match: pd.DataFrame, tournament: str) -> pd.DataFrame:
    """Return raw team means of available style measures using semantic registry names."""
    keys = ["tournament", "team"]
    missing = sorted(set(keys) - set(team_match.columns))
    if missing:
        raise ValueError(f"Missing team profile columns: {', '.join(missing)}.")
    work = team_match.loc[team_match["tournament"].eq(tournament)].copy()
    if work["team"].isna().any():
        raise ValueError("Style team names must be present.")
    candidates = registry.available(work, _style_measures(tournament))
    profiles = work[keys].copy()
    for row in candidates.itertuples(index=False):
        profiles[row.measure] = pd.to_numeric(work[row.column], errors="coerce")
    names = candidates["measure"].tolist()
    if not names:
        return profiles.drop_duplicates().sort_values(keys, ignore_index=True)
    values = profiles.groupby(keys, as_index=False, sort=True)[names].mean()
    return values.sort_values(keys, ignore_index=True)


def _empty_tables(measure_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the three style-fit schemas when no clustering can be estimated."""
    profile_columns = [*PROFILE_COLUMNS, *[
        column for name in measure_names for column in (f"{name}_z", f"{name}_mean")
    ]]
    return (
        pd.DataFrame(columns=TYPE_COLUMNS),
        pd.DataFrame(columns=profile_columns),
        pd.DataFrame(columns=SELECTION_COLUMNS),
    )


def style_label(centroid_z: pd.Series, n_features: int = 2) -> str:
    """Describe the n most distinctive standardised features using registry labels."""
    values = pd.to_numeric(centroid_z, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Style centroid values must be finite numbers.")
    if values.empty or not values.abs().gt(0).any():
        return "average style"
    labels = registry.load_measures().set_index("measure")["label"]
    unknown = sorted(set(values.index) - set(labels.index))
    if unknown:
        raise ValueError(f"Unknown style measures: {', '.join(unknown)}.")
    names = sorted(values.index, key=lambda name: (-abs(float(values[name])), name))[:n_features]
    return ", ".join(
        f"{'high' if values[name] >= 0 else 'low'} {str(labels[name]).lower()}" for name in names
    )


def unique_style_labels(centroids: dict[int, pd.Series], style_ids: dict[int, int]) -> dict[int, str]:
    """Labels from the two most distinctive features; clashing labels add the next feature until
    they differ, and any still equal end with the style number."""
    n = {component: 2 for component in centroids}
    labels = {component: style_label(c, n[component]) for component, c in centroids.items()}
    while True:
        counts = pd.Series(labels).value_counts()
        clashing = [c for c, label in labels.items() if counts[label] > 1 and n[c] < len(centroids[c])]
        if not clashing:
            break
        for component in clashing:
            n[component] += 1
            labels[component] = style_label(centroids[component], n[component])
    counts = pd.Series(labels).value_counts()
    return {c: f"{label} (type {style_ids[c]})" if counts[label] > 1 else label for c, label in labels.items()}


def fit_styles(
    profiles: pd.DataFrame,
    k_range: tuple[int, ...] = K_RANGE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return team styles, centroids and BIC selection, mean-imputing missing profile cells."""
    if not k_range or any(
        isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1 for k in k_range
    ):
        raise ValueError("Style component counts must be positive integers.")
    keys = ["tournament", "team"]
    missing = sorted(set(keys) - set(profiles.columns))
    if missing:
        raise ValueError(f"Missing team profile columns: {', '.join(missing)}.")
    if profiles.empty:
        return _empty_tables([])
    if profiles[keys].isna().any().any() or profiles["tournament"].nunique() != 1:
        raise ValueError("Fit styles for exactly one tournament with non-missing team names.")
    if profiles.duplicated(keys).any():
        raise ValueError("Style profiles require exactly one row per tournament and team.")
    tournament = str(profiles["tournament"].iloc[0])
    work = profiles.sort_values(keys, ignore_index=True)
    raw = pd.DataFrame(index=work.index)
    for row in _style_measures(tournament).itertuples(index=False):
        column = row.measure if row.measure in work else row.column
        if column in work:
            values = pd.to_numeric(work[column], errors="coerce")
            if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Style measure {row.measure!r} must not contain infinite values.")
            if values.notna().any():
                raw[row.measure] = values
    names = raw.columns.tolist()
    empty_types, empty_profiles, selection = _empty_tables(names)
    if raw.empty or len(work) < 2:
        return empty_types, empty_profiles, selection
    means = raw.mean()
    missing_cells = int(raw.isna().sum().sum())
    imputed = raw.fillna(means)
    scale = imputed.std(ddof=0)
    if not scale.gt(0).any():
        return empty_types, empty_profiles, selection
    scale = scale.mask(scale.eq(0), 1.0)
    z = ((imputed - means) / scale).to_numpy(dtype=float)
    unique_profiles = len(np.unique(z, axis=0))
    fitted = {}
    records = []
    for k in sorted(set(k_range)):
        record = {"tournament": tournament, "k": int(k), "bic": np.nan,
                  "silhouette": np.nan, "chosen": False}
        if k <= min(len(work), unique_profiles):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                model = GaussianMixture(
                    n_components=int(k), covariance_type="diag", n_init=N_INIT, random_state=seed,
                ).fit(z)
            labels = model.predict(z)
            record["bic"] = float(model.bic(z))
            if 1 < len(np.unique(labels)) < len(work):
                record["silhouette"] = float(silhouette_score(z, labels))
            fitted[int(k)] = model
        records.append(record)
    selection = pd.DataFrame(records, columns=SELECTION_COLUMNS)
    if not fitted:
        return empty_types, empty_profiles, selection
    bic_by_k = selection.set_index("k")["bic"]
    chosen = min(fitted, key=lambda k: (float(bic_by_k[k]), k))
    selection.loc[selection["k"].eq(chosen), "chosen"] = True
    model = fitted[chosen]
    components = model.predict(z)
    probabilities = model.predict_proba(z)[np.arange(len(work)), components]
    counts = np.bincount(components, minlength=chosen)
    order = sorted(
        np.flatnonzero(counts),
        key=lambda component: (-int(counts[component]), tuple(model.means_[component])),
    )
    style_ids = {int(component): index + 1 for index, component in enumerate(order)}
    style_names = unique_style_labels(
        {int(component): pd.Series(model.means_[component], index=names) for component in order}, style_ids
    )
    types = work[keys].copy()
    types["style_id"] = [style_ids[int(component)] for component in components]
    types["style_label"] = [style_names[int(component)] for component in components]
    types["probability"] = probabilities
    records = []
    for component in order:
        record = {"tournament": tournament, "style_id": style_ids[int(component)],
                  "style_label": style_names[int(component)], "n_teams": int(counts[component])}
        for index, name in enumerate(names):
            record[f"{name}_z"] = float(model.means_[component, index])
            record[f"{name}_mean"] = float(means[name] + model.means_[component, index] * scale[name])
        records.append(record)
    centres = pd.DataFrame(records, columns=empty_profiles.columns)
    selection.attrs["imputed_cells"] = missing_cells
    selection.attrs["converged"] = bool(model.converged_)
    return types[TYPE_COLUMNS], centres, selection


def run(cfg: Config) -> None:
    """Write style assignments, profiles, model selection and team-bootstrap comparisons."""
    team_match = read_table(cfg, "team_match")
    overperformance = read_table(cfg, "overperformance")
    keys = ["tournament", "match_id", "team"]
    if team_match.duplicated(keys).any() or overperformance.duplicated(keys).any():
        raise ValueError("Style over-performance requires one row per tournament, match and team.")
    target = team_match[keys].merge(
        overperformance[[*keys, "overperformance"]], on=keys, how="left", validate="one_to_one",
    )
    target["overperformance"] = pd.to_numeric(target["overperformance"], errors="coerce")
    target = target.loc[np.isfinite(target["overperformance"])]
    target = target.groupby(["tournament", "team"], as_index=False)["overperformance"].mean()
    types_parts, profile_parts, selections, comparisons, checks = [], [], [], [], []
    for tournament in sorted(team_match["tournament"].dropna().unique()):
        profiles = team_profiles(team_match, tournament)
        types, centres, selection = fit_styles(profiles)
        types_parts.append(types)
        profile_parts.append(centres)
        selections.append(selection)
        available = not types.empty
        checks.append({
            "check": f"{tournament}: style model available", "target": 1.0,
            "value": float(available), "passed": available,
        })
        if not available:
            continue
        missing_teams = len(profiles) - len(types)
        checks.extend([
            {"check": f"{tournament}: teams missing a style", "target": 0.0,
             "value": float(missing_teams), "passed": missing_teams == 0},
            {"check": f"{tournament}: style model converged", "target": 1.0,
             "value": float(selection.attrs["converged"]), "passed": bool(selection.attrs["converged"])},
            {"check": f"{tournament}: missing style values replaced with tournament means", "target": np.nan,
             "value": float(selection.attrs["imputed_cells"]), "passed": True},
        ])
        joined = types.merge(target, on=["tournament", "team"], how="left", validate="one_to_one")
        for style_id, style in joined.groupby("style_id", sort=True):
            observed = style.dropna(subset="overperformance")
            interval = {"estimate": np.nan, "low": np.nan, "high": np.nan, "excludes_zero": 0.0}
            if not observed.empty:
                interval = bootstrap_by_team(
                    observed, lambda sample: float(sample["overperformance"].mean()),
                    n_boot=N_BOOT, seed=SEED + int(style_id),
                )
            comparisons.append({
                "tournament": tournament,
                "style_id": int(style_id),
                "style_label": str(style["style_label"].iloc[0]),
                "n_teams": len(style),
                "overperformance_per_match": float(interval["estimate"]),
                "low": float(interval["low"]),
                "high": float(interval["high"]),
                "excludes_zero": bool(interval["excludes_zero"]),
            })
    empty_types, empty_profiles, empty_selection = _empty_tables([])
    tables = {
        "style_types": pd.concat(types_parts, ignore_index=True) if types_parts else empty_types,
        "style_profiles": pd.concat(profile_parts, ignore_index=True) if profile_parts else empty_profiles,
        "style_selection": pd.concat(selections, ignore_index=True) if selections else empty_selection,
        "style_overperformance": pd.DataFrame(comparisons, columns=OVERPERFORMANCE_COLUMNS),
        "checks_styles": pd.DataFrame(checks, columns=CHECK_COLUMNS),
    }
    for name, table in tables.items():
        write_table(cfg, name, table)
    print(
        f"Styles: {len(tables['style_types'])} team assignments "
        f"across {len(comparisons)} tournament styles."
    )
