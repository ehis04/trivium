from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from three_trees.models import expectation as xp
from three_trees.models import progression as pg


def _probabilities(gaps: np.ndarray) -> np.ndarray:
    """Symmetric ordered-logit probabilities with a known positive Elo effect."""
    gaps = np.asarray(gaps, dtype=float) / 180.0
    loss = 1 / (1 + np.exp(gaps + 0.7))
    win = 1 / (1 + np.exp(0.7 - gaps))
    return np.stack([loss, 1 - loss - win, win], axis=-1)


def _tournament(n_groups: int = 4, tournament: str = "men") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four-team groups and a complete knockout, with Elo-related synthetic scores."""
    rng = np.random.default_rng(19)
    teams = [f"T{i:02d}" for i in range(4 * n_groups)]
    ratings = dict(zip(teams, np.linspace(1300.0, 2200.0, len(teams))))
    records = []

    def fixture(home: str, away: str, stage: str) -> str:
        probabilities = _probabilities(np.array(ratings[home] - ratings[away]))
        outcome = int(rng.choice(3, p=probabilities))
        hs, as_ = ((0, 1), (1, 1), (1, 0))[outcome]
        shootout = stage != "Group Stage" and outcome == xp.DRAW
        winner = str(rng.choice([home, away])) if shootout else home if hs > as_ else away
        records.append(
            {
                "tournament": tournament,
                "match_id": len(records) + 1,
                "match_date": pd.Timestamp("2025-11-03"),
                "competition_stage": stage,
                "home_team": home,
                "away_team": away,
                "home_score": hs,
                "away_score": as_,
                "neutral_ground": True,
                "shootout": shootout,
                "shootout_winner": winner if shootout else None,
            }
        )
        return winner

    qualifiers = []
    thirds = []
    for group_index in range(n_groups):
        group = teams[group_index::n_groups]
        points = dict.fromkeys(group, 0)
        for home, away in combinations(group, 2):
            fixture(home, away, "Group Stage")
            last = records[-1]
            if last["home_score"] == last["away_score"]:
                points[home] += 1
                points[away] += 1
            else:
                points[home if last["home_score"] > last["away_score"] else away] += 3
        ranked = sorted(group, key=lambda team: (points[team], team), reverse=True)
        qualifiers.extend(ranked[:2])
        thirds.append((points[ranked[2]], ranked[2]))
    n_best_thirds = 8 if n_groups == 12 else 4 if n_groups == 6 else 0
    qualifiers.extend(team for _, team in sorted(thirds, reverse=True)[:n_best_thirds])
    qualifiers = list(rng.permutation(qualifiers))
    round_labels = {
        32: "Round of 32",
        16: "Round of 16",
        8: "Quarter-finals",
        4: "Semi-finals",
        2: "Final",
    }
    semi_losers = []
    while len(qualifiers) > 1:
        winners = []
        for home, away in zip(qualifiers[::2], qualifiers[1::2]):
            winner = fixture(home, away, round_labels[len(qualifiers)])
            winners.append(winner)
            if len(qualifiers) == 4:
                semi_losers.append(away if winner == home else home)
        qualifiers = winners
    fixture(*semi_losers, "3rd Place Final")
    elo = pd.DataFrame({"tournament": tournament, "team": teams, "elo": list(ratings.values())})
    return pd.DataFrame(records), elo


def _known_fit(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return _probabilities(test["elo_gap"].to_numpy())


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Group Stage", "group"),
        ("GROUP phase", "group"),
        ("3rd Place Final", "third_place"),
        ("Third-place play-off", "third_place"),
        ("Semi-finals", "semi_final"),
        ("SEMI FINAL", "semi_final"),
        ("Quarter-finals", "quarter_final"),
        ("Round of 32", "round_of_32"),
        ("Round of 16", "round_of_16"),
        ("FINAL", "final"),
    ],
)
def test_stage_key_keyword_forms(label: str, expected: str) -> None:
    assert pg.stage_key(label) == expected


def test_stage_key_names_unknown_label() -> None:
    with pytest.raises(ValueError, match="qualification"):
        pg.stage_key("qualification")


@pytest.mark.parametrize(
    ("n_groups", "first_round_size", "n_best_thirds"), [(4, 8, 0), (6, 16, 4), (12, 32, 8)],
)
def test_structure_derives_groups_and_rounds(
    n_groups: int, first_round_size: int, n_best_thirds: int,
) -> None:
    matches, elo = _tournament(n_groups)
    structure = pg.tournament_structure(matches.sample(frac=1, random_state=7))
    assert structure.teams == tuple(sorted(elo["team"]))
    assert structure.groups == {
        chr(ord("A") + index): tuple(elo["team"].iloc[index::n_groups])
        for index in range(n_groups)
    }
    assert len(structure.group_fixtures) == 6 * n_groups
    assert structure.first_round_size == first_round_size
    assert structure.n_best_thirds == n_best_thirds
    expected_rounds = ("quarter_final", "semi_final", "final")
    if n_groups in (6, 12):
        expected_rounds = ("round_of_16", *expected_rounds)
    if n_groups == 12:
        expected_rounds = ("round_of_32", *expected_rounds)
    assert structure.rounds == expected_rounds
    assert structure.stages == ("group", *expected_rounds, "champion")


def test_structure_rejects_incomplete_group() -> None:
    matches, _ = _tournament()
    with pytest.raises(ValueError):
        pg.tournament_structure(matches.drop(index=0))


def test_structure_rejects_missing_knockout_round() -> None:
    matches, _ = _tournament()
    with pytest.raises(ValueError):
        pg.tournament_structure(matches[matches["competition_stage"] != "Semi-finals"])


def test_structure_rejects_repeated_group_pairing() -> None:
    matches, _ = _tournament()
    matches.loc[1, ["home_team", "away_team"]] = matches.loc[0, ["home_team", "away_team"]].to_numpy()
    with pytest.raises(ValueError):
        pg.tournament_structure(matches)


def test_actual_stage_uses_final_shootout_and_ignores_third_place() -> None:
    matches, _ = _tournament()
    final_index = matches.index[matches["competition_stage"] == "Final"][0]
    champion = matches.loc[final_index, "away_team"]
    runner_up = matches.loc[final_index, "home_team"]
    matches.loc[final_index, ["home_score", "away_score"]] = [1, 1]
    matches.loc[final_index, "shootout"] = True
    matches.loc[final_index, "shootout_winner"] = champion
    structure = pg.tournament_structure(matches)
    actual = pg.actual_stage(matches, structure).set_index("team")
    assert actual.loc[champion, "actual_stage_name"] == "champion"
    assert actual.loc[champion, "actual_stage"] == len(structure.stages) - 1
    assert actual.loc[runner_up, "actual_stage_name"] == "final"
    third = matches[matches["competition_stage"] == "3rd Place Final"].iloc[0]
    for team in (third["home_team"], third["away_team"]):
        assert actual.loc[team, "actual_stage_name"] == "semi_final"
    without_third = matches[matches["competition_stage"] != "3rd Place Final"]
    pd.testing.assert_frame_equal(pg.actual_stage(without_third, structure).set_index("team"), actual)


def test_actual_stage_uses_score_for_champion() -> None:
    matches, _ = _tournament()
    final_index = matches.index[matches["competition_stage"] == "Final"][0]
    matches.loc[final_index, ["home_score", "away_score"]] = [2, 0]
    matches.loc[final_index, "shootout"] = False
    actual = pg.actual_stage(matches, pg.tournament_structure(matches)).set_index("team")
    assert actual.loc[matches.loc[final_index, "home_team"], "actual_stage_name"] == "champion"


def test_actual_stage_rejects_drawn_final_without_winner() -> None:
    matches, _ = _tournament()
    final_index = matches.index[matches["competition_stage"] == "Final"][0]
    matches.loc[final_index, ["home_score", "away_score"]] = [1, 1]
    matches.loc[final_index, "shootout_winner"] = None
    with pytest.raises(ValueError, match="final"):
        pg.actual_stage(matches, pg.tournament_structure(matches))


def test_outcome_cube_fits_once_and_aligns_sorted_teams(monkeypatch: pytest.MonkeyPatch) -> None:
    matches, elo = _tournament()
    structure = pg.tournament_structure(matches)
    train = xp.team_match_rows(matches, elo)
    calls = []

    def spy(train_rows: pd.DataFrame, grid: pd.DataFrame) -> np.ndarray:
        pd.testing.assert_frame_equal(train_rows, train)
        calls.append(grid.copy())
        return _known_fit(train_rows, grid)

    monkeypatch.setattr(pg, "ordered_logit", lambda: spy)
    cube = pg.outcome_cube(train, elo.sample(frac=1, random_state=3), structure)
    ratings = elo.set_index("team").loc[list(structure.teams), "elo"].to_numpy()
    assert len(calls) == 1
    assert len(calls[0]) == len(structure.teams) ** 2
    assert cube.shape == (len(structure.teams), len(structure.teams), 3)
    np.testing.assert_allclose(cube, _probabilities(ratings[:, None] - ratings[None, :]))


@pytest.mark.parametrize("n_groups", [4, 6, 12])
def test_simulate_preserves_bracket_counts_with_random_cube(n_groups: int) -> None:
    structure = pg.tournament_structure(_tournament(n_groups)[0])
    n_teams = len(structure.teams)
    cube = np.random.default_rng(41).dirichlet([1, 1, 1], size=(n_teams, n_teams))
    simulated = pg.simulate(structure, cube, 400, np.random.default_rng(73))
    assert simulated.shape == (400, n_teams)
    assert np.issubdtype(simulated.dtype, np.integer)
    assert np.all(simulated >= 0)
    for stage in range(1, len(structure.stages)):
        expected_count = structure.first_round_size // 2 ** (stage - 1)
        np.testing.assert_array_equal((simulated >= stage).sum(axis=1), expected_count)


def test_simulate_unbeatable_team_is_always_champion() -> None:
    structure = pg.tournament_structure(_tournament()[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    cube[0, 1:, :] = [0, 0, 1]
    cube[1:, 0, :] = [1, 0, 0]
    simulated = pg.simulate(structure, cube, 400, np.random.default_rng(12))
    assert np.all(simulated[:, 0] == len(structure.stages) - 1)


def test_simulate_repeated_seed_is_exact() -> None:
    structure = pg.tournament_structure(_tournament()[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    first = pg.simulate(structure, cube, 250, np.random.default_rng(35))
    repeated = pg.simulate(structure, cube, 250, np.random.default_rng(35))
    np.testing.assert_array_equal(first, repeated)


@pytest.mark.parametrize("n_sims", [0, -1, 1.5, True])
def test_simulate_rejects_invalid_simulation_count(n_sims: int) -> None:
    structure = pg.tournament_structure(_tournament()[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    with pytest.raises(ValueError):
        pg.simulate(structure, cube, n_sims, np.random.default_rng(1))


@pytest.mark.parametrize("probabilities", [[-0.1, 0.1, 1], [0.5, 0.5, 0.5], [np.nan, 0, 1]])
def test_simulate_rejects_invalid_probabilities(probabilities: list[float]) -> None:
    structure = pg.tournament_structure(_tournament()[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    cube[0, 1] = probabilities
    with pytest.raises(ValueError):
        pg.simulate(structure, cube, 10, np.random.default_rng(1))


def test_expected_progress_real_fit_probability_identity_and_strength() -> None:
    matches, elo = _tournament()
    expected = pg.expected_progress(matches, elo, n_sims=500, seed=37)
    assert len(expected) == len(elo)
    structure = pg.tournament_structure(matches)
    columns = [f"p_reach_{stage}" for stage in structure.stages[1:]]
    assert expected["expected_stage"].to_numpy() == pytest.approx(expected[columns].sum(axis=1))
    assert np.all(np.diff(expected[columns].to_numpy(), axis=1) <= 0)
    assert np.all((expected[columns].to_numpy() >= 0) & (expected[columns].to_numpy() <= 1))
    assert expected["progress_over_expectation"].to_numpy() == pytest.approx(
        expected["actual_stage"] - expected["expected_stage"]
    )
    assert expected["elo"].corr(expected["expected_stage"]) > 0.8
    assert expected.loc[expected["actual_stage"] == 0, "p_actual_or_better"].eq(1).all()
    for _, row in expected[expected["actual_stage"] > 0].iterrows():
        assert row["p_actual_or_better"] == row[f"p_reach_{row['actual_stage_name']}"]


def test_expected_progress_leaves_both_sides_of_own_matches_out(monkeypatch: pytest.MonkeyPatch) -> None:
    matches, elo = _tournament()
    teams = set(elo["team"])
    omitted_teams = []

    def spy(train: pd.DataFrame, grid: pd.DataFrame) -> np.ndarray:
        omitted = teams - set(train["team"])
        assert len(omitted) == 1
        team = omitted.pop()
        assert not train["opponent"].eq(team).any()
        own_match_ids = set(matches.loc[
            matches["home_team"].eq(team) | matches["away_team"].eq(team), "match_id"
        ])
        assert not own_match_ids.intersection(train["match_id"])
        assert len(grid) == len(teams) ** 2
        omitted_teams.append(team)
        return _known_fit(train, grid)

    monkeypatch.setattr(pg, "ordered_logit", lambda: spy)
    pg.expected_progress(matches, elo, n_sims=100, seed=53)
    assert sorted(omitted_teams) == sorted(teams)


def test_expected_progress_same_seed_survives_input_reordering(monkeypatch: pytest.MonkeyPatch) -> None:
    matches, elo = _tournament()
    monkeypatch.setattr(pg, "ordered_logit", lambda: _known_fit)
    first = pg.expected_progress(matches, elo, n_sims=250, seed=61)
    reordered = pg.expected_progress(
        matches.sample(frac=1, random_state=2), elo.iloc[::-1], n_sims=250, seed=61,
    )
    pd.testing.assert_frame_equal(first, reordered)


def test_expected_progress_multiple_tournaments_fill_absent_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    men_matches, men_elo = _tournament()
    women_matches, women_elo = _tournament(6, "women")
    matches = pd.concat([men_matches, women_matches], ignore_index=True)
    elo = pd.concat([men_elo, women_elo], ignore_index=True)
    monkeypatch.setattr(pg, "ordered_logit", lambda: _known_fit)
    expected = pg.expected_progress(matches, elo, n_sims=100)
    assert len(expected) == 40
    assert not expected.duplicated(["tournament", "team"]).any()
    assert expected.loc[expected["tournament"] == "men", "p_reach_round_of_16"].isna().all()
    assert expected.loc[expected["tournament"] == "women", "p_reach_round_of_16"].notna().all()


def test_checks_progression_shared_schema_and_valid_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    matches, elo = _tournament()
    monkeypatch.setattr(pg, "ordered_logit", lambda: _known_fit)
    expected = pg.expected_progress(matches, elo, n_sims=500)
    checks = pg.checks_progression(expected, matches, elo, n_sims=500)
    assert list(checks.columns) == ["check", "target", "value", "passed"]
    assert len(checks) == 5
    assert checks["check"].str.startswith("men").all()
    assert checks["passed"].all(), checks.to_string(index=False)


@pytest.mark.parametrize(("n_groups", "n_sims"), [(4, 1000), (6, 500), (12, 500)])
def test_first_round_group_separation_removes_clashes(n_groups: int, n_sims: int) -> None:
    structure = pg.tournament_structure(_tournament(n_groups)[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    repaired_stages, repaired_clashes = pg._simulate(
        structure, cube, n_sims, np.random.default_rng(81), separate_groups=True,
    )
    _, unrestricted_clashes = pg._simulate(
        structure, cube, n_sims, np.random.default_rng(81), separate_groups=False,
    )
    assert repaired_clashes.shape == (n_sims,)
    assert not repaired_clashes.any()
    assert unrestricted_clashes.any()
    for stage in range(1, len(structure.stages)):
        expected_count = structure.first_round_size // 2 ** (stage - 1)
        np.testing.assert_array_equal((repaired_stages >= stage).sum(axis=1), expected_count)


def test_group_separation_repairs_all_clashes_and_preserves_qualifiers() -> None:
    paired = np.tile(np.arange(8), (1000, 1))
    group_ids = np.repeat(np.arange(4), 2)
    repaired = pg._separate_first_round(paired.copy(), group_ids, np.random.default_rng(91))
    np.testing.assert_array_equal(np.sort(repaired, axis=1), paired)
    assert np.all(group_ids[repaired[:, ::2]] != group_ids[repaired[:, 1::2]])


@pytest.mark.parametrize("separate_groups", [True, False])
def test_expected_progress_forwards_group_separation_switch(
    monkeypatch: pytest.MonkeyPatch, separate_groups: bool,
) -> None:
    matches, elo = _tournament()
    observed = []

    def spy(
        structure: pg.Structure, cube: np.ndarray, n_sims: int, rng: np.random.Generator,
        separate_groups: bool = True,
        return_opponents: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        observed.append(separate_groups)
        stages = np.zeros((n_sims, len(structure.teams)), dtype=int)
        opponents = np.full((n_sims, len(structure.teams), len(structure.rounds)), -1, dtype=int)
        return (stages, opponents) if return_opponents else stages

    monkeypatch.setattr(pg, "ordered_logit", lambda: _known_fit)
    monkeypatch.setattr(pg, "simulate", spy)
    pg.expected_progress(matches, elo, n_sims=10, separate_groups=separate_groups)
    assert observed == [separate_groups] * len(elo)


def test_simulate_separates_groups_by_default() -> None:
    structure = pg.tournament_structure(_tournament()[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    default = pg.simulate(structure, cube, 200, np.random.default_rng(101))
    explicit = pg.simulate(structure, cube, 200, np.random.default_rng(101), separate_groups=True)
    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize("n_groups", [4, 6, 12])
def test_simulated_opponents_are_reciprocal_and_only_present_when_round_played(n_groups: int) -> None:
    structure = pg.tournament_structure(_tournament(n_groups)[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    stages, opponents = pg.simulate(
        structure, cube, 300, np.random.default_rng(111), return_opponents=True,
    )
    assert opponents.shape == (300, n_teams, len(structure.rounds))
    assert np.issubdtype(opponents.dtype, np.integer)
    team_ids = np.broadcast_to(np.arange(n_teams), (300, n_teams))
    sim_ids = np.arange(300)[:, None]
    for round_index in range(len(structure.rounds)):
        played = stages >= round_index + 1
        observed = opponents[:, :, round_index]
        assert np.all(observed[~played] == -1)
        assert np.all((observed[played] >= 0) & (observed[played] < n_teams))
        assert np.all(observed[played] != team_ids[played])
        reciprocal = opponents[sim_ids, np.maximum(observed, 0), round_index]
        np.testing.assert_array_equal(reciprocal[played], team_ids[played])


def test_returning_opponents_preserves_stages_and_rng_and_is_reproducible() -> None:
    structure = pg.tournament_structure(_tournament()[0])
    n_teams = len(structure.teams)
    cube = np.full((n_teams, n_teams, 3), 1 / 3)
    plain_rng = np.random.default_rng(121)
    detailed_rng = np.random.default_rng(121)
    plain = pg.simulate(structure, cube, 300, plain_rng)
    detailed = pg.simulate(structure, cube, 300, detailed_rng, return_opponents=True)
    repeated = pg.simulate(structure, cube, 300, np.random.default_rng(121), return_opponents=True)
    np.testing.assert_array_equal(plain, detailed[0])
    np.testing.assert_array_equal(plain_rng.random(10), detailed_rng.random(10))
    for first, second in zip(detailed, repeated):
        np.testing.assert_array_equal(first, second)


def test_path_difficulty_known_weak_opponents_and_group_exit_nans(monkeypatch: pytest.MonkeyPatch) -> None:
    matches, elo = _tournament()
    structure = pg.tournament_structure(matches)
    actual = pg.actual_stage(matches, structure)
    target = actual.loc[actual["actual_stage_name"] == "champion", "team"].iloc[0]
    target_index = structure.teams.index(target)
    elo["elo"] = 1500.0
    elo.loc[elo["team"] == target, "elo"] = 2200.0
    n_teams = len(structure.teams)
    cube = np.zeros((n_teams, n_teams, 3))
    cube[:, :, xp.DRAW] = 1.0
    other_indices = [index for index in range(n_teams) if index != target_index]
    cube[target_index, other_indices, :] = [0, 0, 1]
    cube[other_indices, target_index, :] = [1, 0, 0]
    monkeypatch.setattr(pg, "outcome_cube", lambda train, ratings, inferred: cube)
    expected = pg.expected_progress(matches, elo, n_sims=200, seed=131).set_index("team")
    assert expected.loc[target, "knockout_opponent_elo_expected"] == pytest.approx(1500.0)
    assert expected.loc[target, "knockout_opponent_elo_actual"] == pytest.approx(1500.0)
    assert expected.loc[target, "path_easier_by"] == pytest.approx(0.0)
    group_exits = expected["actual_stage"].eq(0)
    assert group_exits.any()
    knockout_columns = [
        "knockout_opponent_elo_actual", "knockout_opponent_elo_expected", "path_easier_by",
    ]
    assert expected.loc[group_exits, knockout_columns].isna().all().all()
    assert expected["group_opponent_elo"].notna().all()
    for team in structure.teams:
        group = next(members for members in structure.groups.values() if team in members)
        opponent_ratings = elo.loc[elo["team"].isin(set(group) - {team}), "elo"]
        assert expected.loc[team, "group_opponent_elo"] == pytest.approx(opponent_ratings.mean())


@pytest.mark.parametrize("reach_second_round", [True, False])
def test_expected_path_weights_actual_rounds_equally_and_requires_reaching_each(
    monkeypatch: pytest.MonkeyPatch, reach_second_round: bool,
) -> None:
    matches, elo = _tournament()
    structure = pg.tournament_structure(matches)
    actual = pg.actual_stage(matches, structure)
    target = actual.loc[actual["actual_stage_name"] == "semi_final", "team"].iloc[0]
    target_index = structure.teams.index(target)
    alternatives = [team for team in structure.teams if team != target]
    low, high = alternatives[0], alternatives[-1]
    ratings = elo.set_index("team")["elo"]

    def controlled(
        inferred: pg.Structure, cube: np.ndarray, n_sims: int, rng: np.random.Generator,
        separate_groups: bool = True, return_opponents: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        stages = np.zeros((n_sims, len(inferred.teams)), dtype=int)
        stages[:, target_index] = 1
        opponents = np.full((n_sims, len(inferred.teams), len(inferred.rounds)), -1, dtype=int)
        opponents[:, target_index, 0] = inferred.teams.index(low)
        if reach_second_round:
            stages[:2, target_index] = 2
            opponents[:2, target_index, 1] = inferred.teams.index(high)
        return (stages, opponents) if return_opponents else stages

    monkeypatch.setattr(pg, "ordered_logit", lambda: _known_fit)
    monkeypatch.setattr(pg, "simulate", controlled)
    expected = pg.expected_progress(matches, elo, n_sims=10).set_index("team")
    if reach_second_round:
        round_mean = (ratings[low] + ratings[high]) / 2
        assert expected.loc[target, "knockout_opponent_elo_expected"] == pytest.approx(round_mean)
        assert expected.loc[target, "path_easier_by"] == pytest.approx(
            round_mean - expected.loc[target, "knockout_opponent_elo_actual"]
        )
    else:
        assert pd.isna(expected.loc[target, "knockout_opponent_elo_expected"])
        assert pd.isna(expected.loc[target, "path_easier_by"])


def test_actual_path_difficulty_excludes_third_place_match(monkeypatch: pytest.MonkeyPatch) -> None:
    matches, elo = _tournament()
    monkeypatch.setattr(pg, "ordered_logit", lambda: _known_fit)
    with_third_place = pg.expected_progress(matches, elo, n_sims=100, seed=141)
    without_third_place = pg.expected_progress(
        matches[matches["competition_stage"] != "3rd Place Final"], elo, n_sims=100, seed=141,
    )
    pd.testing.assert_frame_equal(with_third_place, without_third_place)
