"""Expected stages from whole-tournament simulations, fitted separately for each
team with its matches excluded. Group results use ordered-logit probabilities
from the Elo gap alone. Knockout pairings are random: FIFA's slot rules for
third-placed teams are not modelled. Group ties are broken at random and
shootouts are coin flips. Group rivals are kept apart in the first knockout
round by default. The third-place match does not affect progression.
Path difficulty compares opponent Elo over the knockout rounds actually
played, weighting each round equally and conditioning on reaching it in the
simulations. Unvisited rounds have no estimated opponent Elo.

Output tables: expected_progress, checks_progression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from three_trees.models.expectation import POINTS, ordered_logit, team_match_rows

SEED = 20251103
PAIRING_REPAIR_PASSES = 2
KNOCKOUT_ORDER = ("round_of_32", "round_of_16", "quarter_final", "semi_final", "final")
PROGRESS_COLUMNS = [
    "tournament", "team", "elo", "actual_stage", "actual_stage_name", "expected_stage",
    "progress_over_expectation", "p_actual_or_better",
    *[f"p_reach_{stage}" for stage in (*KNOCKOUT_ORDER, "champion")],
    "group_opponent_elo", "knockout_opponent_elo_actual", "knockout_opponent_elo_expected", "path_easier_by",
]
CHECK_COLUMNS = ["check", "target", "value", "passed"]


@dataclass(frozen=True)
class Structure:
    teams: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]
    group_fixtures: tuple[tuple[str, str], ...]
    rounds: tuple[str, ...]
    first_round_size: int
    n_best_thirds: int
    stages: tuple[str, ...]


def stage_key(label: str) -> str:
    """Return the stage key for a provider label, checking specific keywords first."""
    if isinstance(label, str):
        text = label.casefold()
        for keywords, key in (
            (("group",), "group"),
            (("3rd", "third"), "third_place"),
            (("semi",), "semi_final"),
            (("quarter",), "quarter_final"),
            (("32",), "round_of_32"),
            (("16",), "round_of_16"),
            (("final",), "final"),
        ):
            if any(word in text for word in keywords):
                return key
    raise ValueError(f"Unknown competition stage label: {label!r}.")


def tournament_structure(matches_t: pd.DataFrame) -> Structure:
    """Infer four-team groups and a consecutive knockout bracket from one tournament."""
    if matches_t.empty or matches_t["tournament"].nunique(dropna=False) != 1:
        raise ValueError("Tournament structure requires matches from exactly one tournament.")
    if matches_t["match_id"].isna().any() or matches_t["match_id"].duplicated().any():
        raise ValueError("Tournament match IDs must be present and unique.")
    if matches_t[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Tournament teams must have names.")
    same = matches_t["home_team"].eq(matches_t["away_team"])
    if same.any():
        raise ValueError(f"Teams cannot play themselves: {matches_t.loc[same, 'home_team'].tolist()}.")
    keys = matches_t["competition_stage"].map(stage_key)
    group_rows = matches_t.loc[keys.eq("group")].sort_values("match_id", ignore_index=True)
    graph: dict[str, set[str]] = {}
    for home, away in group_rows[["home_team", "away_team"]].itertuples(index=False, name=None):
        graph.setdefault(home, set()).add(away)
        graph.setdefault(away, set()).add(home)
    components = []
    unseen = set(graph)
    while unseen:
        pending = [min(unseen)]
        component: set[str] = set()
        while pending:
            team = pending.pop()
            if team not in component:
                component.add(team)
                pending.extend(graph[team] - component)
        unseen -= component
        fixtures = group_rows.loc[group_rows["home_team"].isin(component)]
        pairs = {frozenset(pair) for pair in fixtures[["home_team", "away_team"]].itertuples(
            index=False, name=None,
        )}
        if len(component) != 4 or len(fixtures) != 6 or len(pairs) != 6:
            raise ValueError(f"Group {sorted(component)} must have four teams and six distinct fixtures.")
        components.append((fixtures["match_id"].min(), tuple(sorted(component))))
    groups = {chr(ord("A") + i): teams for i, (_, teams) in enumerate(sorted(components))}
    teams = tuple(sorted(set(matches_t["home_team"]) | set(matches_t["away_team"])))
    missing = sorted(set(teams) - set(graph))
    if not groups or missing:
        raise ValueError(f"Every team must belong to a four-team group; missing teams: {missing}.")
    rounds = tuple(stage for stage in KNOCKOUT_ORDER if keys.eq(stage).any())
    if not rounds or rounds != KNOCKOUT_ORDER[KNOCKOUT_ORDER.index(rounds[0]):]:
        raise ValueError(f"Knockout rounds must be consecutive and end with final: {rounds}.")
    first_round_size = 2 * int(keys.eq(rounds[0]).sum())
    for i, stage in enumerate(rounds):
        fixtures = matches_t.loc[keys.eq(stage)]
        entrants = pd.concat([fixtures["home_team"], fixtures["away_team"]], ignore_index=True)
        if len(fixtures) != 2 ** (len(rounds) - i - 1) or entrants.duplicated().any():
            raise ValueError(f"Knockout round {stage} has invalid fixtures or repeated teams.")
    n_best_thirds = first_round_size - 2 * len(groups)
    if not 0 <= n_best_thirds <= len(groups):
        raise ValueError(f"Invalid number of best third-placed qualifiers: {n_best_thirds}.")
    return Structure(
        teams=teams,
        groups=groups,
        group_fixtures=tuple(group_rows[["home_team", "away_team"]].itertuples(index=False, name=None)),
        rounds=rounds,
        first_round_size=first_round_size,
        n_best_thirds=n_best_thirds,
        stages=("group", *rounds, "champion"),
    )


def actual_stage(matches_t: pd.DataFrame, structure: Structure) -> pd.DataFrame:
    """Return the furthest stage each team reached, including the final's score or shootout winner."""
    keys = matches_t["competition_stage"].map(stage_key)
    reached = {team: 0 for team in structure.teams}
    for index, stage in enumerate(structure.rounds, start=1):
        fixtures = matches_t.loc[keys.eq(stage)]
        for team in set(fixtures["home_team"]) | set(fixtures["away_team"]):
            if team not in reached:
                raise ValueError(f"Knockout team {team!r} is missing from the tournament structure.")
            reached[team] = index
    finals = matches_t.loc[keys.eq("final")]
    if len(finals) != 1:
        raise ValueError("Actual stages require exactly one final.")
    final = finals.iloc[0]
    home, away = final["home_team"], final["away_team"]
    hs, aws = final["home_score"], final["away_score"]
    if pd.isna(hs) or pd.isna(aws):
        raise ValueError(f"Final scores are missing for {home} and {away}.")
    if hs == aws:
        winner = final["shootout_winner"]
        if pd.isna(winner) or winner not in (home, away):
            raise ValueError(f"Drawn final between {home} and {away} needs a valid shootout winner.")
    else:
        winner = home if hs > aws else away
    reached[winner] = len(structure.stages) - 1
    return pd.DataFrame({
        "team": structure.teams,
        "actual_stage": [reached[team] for team in structure.teams],
        "actual_stage_name": [structure.stages[reached[team]] for team in structure.teams],
    })


def _ratings(elo_t: pd.DataFrame, structure: Structure) -> np.ndarray:
    """Return finite ratings aligned to the structure, rejecting duplicates and missing teams."""
    ratings = elo_t.loc[elo_t["team"].isin(structure.teams), ["team", "elo"]]
    duplicates = ratings.loc[ratings["team"].duplicated(), "team"].tolist()
    if duplicates:
        raise ValueError(f"Duplicate Elo ratings for teams: {duplicates}.")
    aligned = pd.to_numeric(ratings.set_index("team")["elo"].reindex(structure.teams), errors="coerce")
    invalid = aligned.index[~np.isfinite(aligned.to_numpy(dtype=float))].tolist()
    if invalid:
        raise ValueError(f"Missing or non-finite Elo ratings for teams: {invalid}.")
    return aligned.to_numpy(dtype=float)


def outcome_cube(train_rows: pd.DataFrame, elo_t: pd.DataFrame, structure: Structure) -> np.ndarray:
    """Predict every ordered team pairing with one ordered-logit fit on the supplied training rows."""
    ratings = _ratings(elo_t, structure)
    grid = pd.DataFrame({"elo_gap": (ratings[:, None] - ratings[None, :]).ravel()})
    probs = ordered_logit()(train_rows, grid)
    cube = np.asarray(probs, dtype=float).reshape(len(ratings), len(ratings), -1)
    _validate_cube(cube, len(ratings))
    return cube


def _validate_cube(cube: np.ndarray, n_teams: int) -> None:
    """Reject probabilities that cannot describe a loss, draw and win for every pairing."""
    if cube.shape != (n_teams, n_teams, 3):
        raise ValueError(f"Outcome cube must have shape ({n_teams}, {n_teams}, 3).")
    if not np.isfinite(cube).all() or (cube < 0).any() or (cube > 1).any():
        raise ValueError("Outcome cube probabilities must be finite and between zero and one.")
    if not np.allclose(cube.sum(axis=2), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Outcome cube probabilities must sum to one for every pairing.")


def _validate_n_sims(n_sims: int) -> None:
    """Require a positive integer simulation count."""
    if isinstance(n_sims, bool) or not isinstance(n_sims, (int, np.integer)) or n_sims <= 0:
        raise ValueError("The number of simulations must be a positive integer.")


def simulate(
    structure: Structure,
    cube: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    separate_groups: bool = True,
    return_opponents: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return stages and, optionally, opponent indices by knockout round (-1 when absent)."""
    simulated = _simulate(structure, cube, n_sims, rng, separate_groups, return_opponents)
    if return_opponents:
        stages, _, opponents = simulated
        return stages, opponents
    stages, _ = simulated
    return stages


def _separate_first_round(
    paired: np.ndarray,
    group_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Repair first-round group clashes by random swaps that leave both pairs valid."""
    paired = paired.copy()
    n_pairs = paired.shape[1] // 2
    for _ in range(PAIRING_REPAIR_PASSES):
        for pair_index in range(n_pairs):
            groups = group_ids[paired]
            clashing = groups[:, 2 * pair_index] == groups[:, 2 * pair_index + 1]
            rows = np.flatnonzero(clashing)
            if not len(rows):
                continue
            group = groups[rows, 2 * pair_index, None]
            eligible = (groups[rows, ::2] != group) & (groups[rows, 1::2] != group)
            valid_rows = eligible.any(axis=1)
            rows, eligible = rows[valid_rows], eligible[valid_rows]
            scores = np.where(eligible, rng.random(eligible.shape), np.inf)
            partners = scores.argmin(axis=1) * 2 + rng.integers(0, 2, size=len(rows))
            member = 2 * pair_index + 1
            saved = paired[rows, member].copy()
            paired[rows, member] = paired[rows, partners]
            paired[rows, partners] = saved
    return paired


def _simulate(
    structure: Structure,
    cube: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
    separate_groups: bool,
    return_opponents: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stages, residual first-round clash flags and optional knockout opponents."""
    _validate_n_sims(n_sims)
    cube = np.asarray(cube, dtype=float)
    n_teams = len(structure.teams)
    _validate_cube(cube, n_teams)
    indices = {team: i for i, team in enumerate(structure.teams)}
    group_ids = np.empty(n_teams, dtype=int)
    for group_index, group in enumerate(structure.groups.values()):
        group_ids[[indices[team] for team in group]] = group_index
    sim_index = np.arange(n_sims)[:, None]
    points = np.zeros((n_sims, n_teams), dtype=int)
    for home, away in structure.group_fixtures:
        h, a = indices[home], indices[away]
        outcome = (rng.random((n_sims, 1)) >= np.cumsum(cube[h, a])[:2]).sum(axis=1)
        points[:, h] += POINTS[outcome].astype(int)
        points[:, a] += POINTS[2 - outcome].astype(int)
    top_two, thirds = [], []
    for group in structure.groups.values():
        members = np.array([indices[team] for team in group])
        order = np.argsort(-(points[:, members] + rng.random((n_sims, 4)) * 0.5), axis=1)
        ranked = members[order]
        top_two.append(ranked[:, :2])
        thirds.append(ranked[:, 2])
    qualifiers = np.concatenate(top_two, axis=1)
    if structure.n_best_thirds:
        candidates = np.stack(thirds, axis=1)
        rank = np.argsort(
            -(points[sim_index, candidates] + rng.random(candidates.shape) * 0.5), axis=1,
        )
        best = np.take_along_axis(candidates, rank[:, :structure.n_best_thirds], axis=1)
        qualifiers = np.concatenate([qualifiers, best], axis=1)
    advancement = cube[:, :, 2] + 0.5 * cube[:, :, 1]
    knockout = (advancement + (1.0 - advancement.T)) / 2.0
    stages = np.zeros((n_sims, n_teams), dtype=int)
    first_round_clashes = np.zeros(n_sims, dtype=bool)
    opponents = np.full((n_sims, n_teams, len(structure.rounds)), -1, dtype=int) if return_opponents else None
    for stage_index, _ in enumerate(structure.rounds, start=1):
        stages[sim_index, qualifiers] = stage_index
        paired = rng.permuted(qualifiers, axis=1)
        if stage_index == 1:
            if separate_groups:
                paired = _separate_first_round(paired, group_ids, rng)
            first_round_clashes = np.any(group_ids[paired[:, ::2]] == group_ids[paired[:, 1::2]], axis=1)
        left, right = paired[:, ::2], paired[:, 1::2]
        if opponents is not None:
            opponents[sim_index, left, stage_index - 1] = right
            opponents[sim_index, right, stage_index - 1] = left
        qualifiers = np.where(rng.random(left.shape) < knockout[left, right], left, right)
    stages[sim_index, qualifiers] = len(structure.stages) - 1
    if opponents is not None:
        return stages, first_round_clashes, opponents
    return stages, first_round_clashes


def expected_progress(
    matches: pd.DataFrame,
    elo_pre: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int = SEED,
    separate_groups: bool = True,
) -> pd.DataFrame:
    """Return actual and expected stages using each team's own leave-team-out tournament model."""
    _validate_n_sims(n_sims)
    records = []
    for tournament_index, (tournament, matches_t) in enumerate(matches.groupby("tournament", sort=True)):
        structure = tournament_structure(matches_t)
        elo_t = elo_pre.loc[elo_pre["tournament"].eq(tournament)]
        ratings = _ratings(elo_t, structure)
        actual = actual_stage(matches_t, structure).set_index("team")
        rows = team_match_rows(matches_t, elo_t, host_advantage=0.0).reset_index(drop=True)
        indices = {team: i for i, team in enumerate(structure.teams)}
        group_opponent_elo = {}
        for group in structure.groups.values():
            total = sum(ratings[indices[team]] for team in group)
            group_opponent_elo.update({team: (total - ratings[indices[team]]) / 3 for team in group})
        keys = rows["competition_stage"].map(stage_key)
        knockout_rows = rows.loc[keys.isin(structure.rounds), ["team", "opponent"]].copy()
        knockout_rows["round_index"] = keys.loc[knockout_rows.index].map(
            {stage: i for i, stage in enumerate(structure.rounds)},
        )
        for team_index, team in enumerate(structure.teams):
            own = rows["team"].eq(team) | rows["opponent"].eq(team)
            cube = outcome_cube(rows.loc[~own].copy(), elo_t, structure)
            rng = np.random.default_rng([seed, tournament_index, team_index])
            simulated_stages, opponents = simulate(
                structure, cube, n_sims, rng, separate_groups=separate_groups, return_opponents=True,
            )
            stages = simulated_stages[:, team_index]
            actual_index = int(actual.loc[team, "actual_stage"])
            expected = float(stages.mean())
            played = knockout_rows.loc[knockout_rows["team"].eq(team)]
            opponent_elo_actual = np.nan
            opponent_elo_expected = np.nan
            if not played.empty:
                opponent_elo_actual = float(played["opponent"].map(
                    {name: ratings[index] for name, index in indices.items()},
                ).mean())
                round_means = []
                for round_index in played["round_index"]:
                    met = opponents[:, team_index, int(round_index)]
                    met = met[met >= 0]
                    round_means.append(float(ratings[met].mean()) if len(met) else np.nan)
                opponent_elo_expected = float(np.mean(round_means))
            record = {
                "tournament": tournament,
                "team": team,
                "elo": float(ratings[team_index]),
                "actual_stage": actual_index,
                "actual_stage_name": actual.loc[team, "actual_stage_name"],
                "expected_stage": expected,
                "progress_over_expectation": actual_index - expected,
                "p_actual_or_better": float(np.mean(stages >= actual_index)),
                "group_opponent_elo": float(group_opponent_elo[team]),
                "knockout_opponent_elo_actual": opponent_elo_actual,
                "knockout_opponent_elo_expected": opponent_elo_expected,
                "path_easier_by": opponent_elo_expected - opponent_elo_actual,
            }
            record.update({f"p_reach_{stage}": np.nan for stage in (*KNOCKOUT_ORDER, "champion")})
            for index, stage in enumerate(structure.stages[1:], start=1):
                record[f"p_reach_{stage}"] = float(np.mean(stages >= index))
            records.append(record)
    return pd.DataFrame(records, columns=PROGRESS_COLUMNS)


def checks_progression(
    expected: pd.DataFrame,
    matches: pd.DataFrame,
    elo_pre: pd.DataFrame,
    n_sims: int = 2_000,
    seed: int = SEED,
) -> pd.DataFrame:
    """Return bracket conservation, probability totals, Elo correlation and actual-stage checks."""
    _validate_n_sims(n_sims)
    checks = []
    for tournament_index, (tournament, matches_t) in enumerate(matches.groupby("tournament", sort=True)):
        structure = tournament_structure(matches_t)
        elo_t = elo_pre.loc[elo_pre["tournament"].eq(tournament)]
        rows = team_match_rows(matches_t, elo_t, host_advantage=0.0).reset_index(drop=True)
        cube = outcome_cube(rows, elo_t, structure)
        stages, clashes = _simulate(
            structure, cube, n_sims, np.random.default_rng([seed, tournament_index]), True,
        )
        clash_share = float(clashes.mean())
        sizes = np.array([structure.first_round_size // 2 ** i for i in range(len(structure.stages) - 1)])
        counts = np.stack([(stages >= i).sum(axis=1) for i in range(1, len(structure.stages))], axis=1)
        violations = int(np.any(counts != sizes, axis=1).sum())
        table = expected.loc[expected["tournament"].eq(tournament)]
        if table["team"].duplicated().any() or set(table["team"]) != set(structure.teams):
            raise ValueError(f"Expected progression must contain each {tournament} team exactly once.")
        totals = np.array([table[f"p_reach_{stage}"].sum() for stage in structure.stages[1:]])
        difference = float(np.max(np.abs(totals - sizes)))
        correlation = float(table["elo"].corr(table["expected_stage"])) if (
            table["elo"].nunique() > 1 and table["expected_stage"].nunique() > 1
        ) else np.nan
        actual = actual_stage(matches_t, structure)["actual_stage"].to_numpy()
        actual_counts = np.array([(actual >= i).sum() for i in range(1, len(sizes) + 1)])
        actual_difference = float(np.max(np.abs(actual_counts - sizes)))
        checks.extend([
            _check(f"{tournament} simulated bracket violations", 0, violations, violations == 0),
            _check(f"{tournament} maximum reach probability total difference",
                   0, difference, difference <= 2),
            _check(f"{tournament} correlation of Elo and expected stage",
                   0.5, correlation, correlation > 0.5),
            _check(f"{tournament} actual stage count difference",
                   0, actual_difference, actual_difference == 0),
            _check(f"{tournament} same-group first-round tie share", 0, clash_share, clash_share <= 0.01),
        ])
    return pd.DataFrame(checks, columns=CHECK_COLUMNS)


def _check(name: str, target: float, value: float, passed: bool) -> dict[str, object]:
    """Return one row of the shared checks schema."""
    return {"check": name, "target": float(target), "value": float(value), "passed": bool(passed)}
