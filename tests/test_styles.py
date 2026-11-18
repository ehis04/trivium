from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from trivium import measures as registry
from trivium.models import styles as st
from trivium.storage import read_table, write_table
from trivium.synthetic import demo_config


def _profiles(n_each: int = 12, seed: int = 31) -> tuple[pd.DataFrame, np.ndarray]:
    """Three distinct synthetic styles on registered, team-level measurements."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.2, 0.25, 1.0, 0.8], [0.5, 0.5, 3.0, 0.5], [0.8, 0.75, 5.0, 0.2]])
    truth = np.repeat(np.arange(3), n_each)
    values = centres[truth] + rng.normal(0, [0.005, 0.005, 0.02, 0.005], (len(truth), 4))
    profiles = pd.DataFrame(values, columns=[
        "territory_share", "possession_share", "pressures_per_opp_possession_min", "directness",
    ])
    profiles.insert(0, "team", [f"Team {i:02d}" for i in range(len(profiles))])
    profiles.insert(0, "tournament", "men")
    return profiles, truth


def _team_match(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = profiles.loc[profiles.index.repeat(2)].reset_index(drop=True)
    rows["match_id"] = np.tile([1, 2], len(profiles))
    rows["opponent"] = "Other team"
    return rows


def test_team_profiles_uses_registered_measures_and_team_means() -> None:
    profiles, _ = _profiles()
    rows = _team_match(profiles)
    rows.loc[rows["match_id"].eq(1), "possession_share"] += 0.04
    rows["not_a_registered_measure"] = 123.0
    women = rows.assign(tournament="women", territory_share=0.01)
    result = st.team_profiles(pd.concat([rows, women], ignore_index=True), "men").set_index("team")
    assert len(result) == len(profiles)
    assert result["tournament"].eq("men").all()
    assert "not_a_registered_measure" not in result
    np.testing.assert_allclose(result["possession_share"], profiles["possession_share"] + 0.02)
    np.testing.assert_allclose(result["territory_share"], profiles["territory_share"])


def test_team_profiles_resolves_registry_source_column_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = registry.load_measures()
    metadata.loc[metadata["measure"].eq("possession_share"), "column"] = "provider_possession"
    monkeypatch.setattr(registry, "load_measures", lambda: metadata.copy())
    profiles, _ = _profiles()
    rows = _team_match(profiles).rename(columns={"possession_share": "provider_possession"})
    result = st.team_profiles(rows, "men")
    assert "possession_share" in result
    assert "provider_possession" not in result
    np.testing.assert_allclose(result["possession_share"], profiles["possession_share"])


def test_three_clear_clusters_are_recovered_and_every_team_has_one_style() -> None:
    profiles, truth = _profiles()
    types, centroids, selection = st.fit_styles(profiles, k_range=(3,), seed=17)
    assigned = types.set_index("team").loc[profiles["team"], "style_id"]
    assert adjusted_rand_score(truth, assigned) >= 0.9
    assert not types.duplicated(["tournament", "team"]).any()
    assert len(types) == len(profiles)
    assert types["probability"].between(0, 1).all()
    assert types["probability"].gt(0.99).all()
    assert set(types["style_id"]) == {1, 2, 3}
    assert centroids["n_teams"].sum() == len(profiles)
    assert centroids.sort_values("style_id")["n_teams"].is_monotonic_decreasing
    assert selection["chosen"].sum() == 1
    assert selection.loc[selection["chosen"], "k"].iloc[0] == 3
    assert selection.loc[selection["chosen"], "silhouette"].iloc[0] > 0.9
    for name in ("territory_share", "possession_share", "pressures_per_opp_possession_min", "directness"):
        assert f"{name}_z" in centroids and f"{name}_mean" in centroids


def test_style_ids_follow_assigned_size_with_unequal_clusters() -> None:
    profiles, truth = _profiles()
    keep = np.concatenate([np.flatnonzero(truth == i)[:count] for i, count in enumerate((12, 9, 6))])
    types, centroids, _ = st.fit_styles(profiles.iloc[keep], k_range=(3,))
    assert centroids.sort_values("style_id")["n_teams"].tolist() == [12, 9, 6]
    assert types.groupby("style_id").size().tolist() == [12, 9, 6]


def test_bic_selects_the_lowest_candidate_and_seed_is_exact() -> None:
    profiles, _ = _profiles()
    first = st.fit_styles(profiles, seed=27)
    repeated = st.fit_styles(profiles, seed=27)
    for left, right in zip(first, repeated):
        pd.testing.assert_frame_equal(left, right)
    selection = first[2]
    assert set(selection["k"]) == {3, 4}
    assert selection.loc[selection["chosen"], "bic"].iloc[0] == pytest.approx(selection["bic"].min())


def test_style_label_uses_two_strongest_registered_labels() -> None:
    centroid = pd.Series({"possession_share": 2.0, "territory_share": -3.0, "directness": 0.2})
    assert st.style_label(centroid) == "low field tilt, high possession share"
    assert st.style_label(centroid) == st.style_label(centroid.iloc[::-1])


def test_clashing_style_labels_are_made_unique() -> None:
    a = pd.Series({"possession_share": 2.0, "territory_share": 1.5, "directness": 0.4})
    b = pd.Series({"possession_share": 1.8, "territory_share": 1.2, "directness": -0.3})
    c = pd.Series({"possession_share": -1.0, "territory_share": -2.0, "directness": 0.1})
    labels = st.unique_style_labels({0: a, 1: b, 2: c}, {0: 1, 1: 2, 2: 3})
    assert len(set(labels.values())) == 3
    assert labels[0] == "high possession share, high field tilt, high directness"
    assert labels[1] == "high possession share, high field tilt, low directness"
    assert labels[2] == "low field tilt, low possession share"
    same = st.unique_style_labels({0: a, 1: a.copy()}, {0: 1, 1: 2})
    assert same[0].endswith("(type 1)") and same[1].endswith("(type 2)")


def test_partial_missing_profiles_still_assign_every_team() -> None:
    profiles, _ = _profiles()
    profiles.loc[0, "possession_share"] = np.nan
    types, _, _ = st.fit_styles(profiles, k_range=(3,))
    assert set(types["team"]) == set(profiles["team"])
    assert types["probability"].notna().all()


def test_fitting_mixed_tournaments_raises() -> None:
    profiles, _ = _profiles()
    profiles.loc[0, "tournament"] = "women"
    with pytest.raises(ValueError, match="one tournament"):
        st.fit_styles(profiles)


def test_no_available_style_features_returns_empty_tables() -> None:
    profiles, _ = _profiles()
    result = st.fit_styles(profiles[["tournament", "team"]])
    assert all(table.empty for table in result)
    assert "style_id" in result[0] and "chosen" in result[2]


def test_stage_writes_styles_and_bootstraps_teams_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles, truth = _profiles()
    men = _team_match(profiles)
    extra = men.loc[men["team"].eq("Team 00")].copy()
    extra["match_id"] += 2
    men = pd.concat([men, extra], ignore_index=True)
    women = men.assign(tournament="women")
    team_match = pd.concat([men, women], ignore_index=True)
    effects = dict(zip(profiles["team"], (truth - 1) * 2.0))
    effects["Team 00"] = 10.0
    performance = team_match[["tournament", "match_id", "team"]].copy()
    performance["overperformance"] = performance["team"].map(effects)
    cfg = demo_config(tmp_path)
    write_table(cfg, "team_match", team_match)
    write_table(cfg, "overperformance", performance)
    monkeypatch.setattr(st, "N_BOOT", 80)
    st.run(cfg)
    types = read_table(cfg, "style_types")
    assert len(types) == 2 * len(profiles)
    assert not types.duplicated(["tournament", "team"]).any()
    assert set(types["tournament"]) == {"men", "women"}
    overperformance = read_table(cfg, "style_overperformance")
    assert overperformance["n_teams"].sum() == len(types)
    team_effects = types.assign(value=types["team"].map(effects))
    expected_means = team_effects.groupby(["tournament", "style_id"])["value"].mean()
    observed_means = overperformance.set_index(["tournament", "style_id"])["overperformance_per_match"]
    np.testing.assert_allclose(observed_means.sort_index(), expected_means.sort_index())
    assert (overperformance["low"] <= overperformance["high"]).all()
    strong = overperformance["overperformance_per_match"].abs().gt(1)
    assert overperformance.loc[strong, "excludes_zero"].all()
    selection = read_table(cfg, "style_selection")
    assert selection.groupby("tournament")["chosen"].sum().eq(1).all()
    checks = read_table(cfg, "checks_styles")
    assert list(checks.columns) == ["check", "target", "value", "passed"]
    assert checks["passed"].all(), checks.to_string(index=False)
