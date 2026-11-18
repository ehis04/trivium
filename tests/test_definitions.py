from __future__ import annotations

from three_trees import definitions
from three_trees.measures import load_measures


def test_every_mapped_measure_is_in_the_registry_and_points_are_complete() -> None:
    registry = set(load_measures()["measure"])
    assert set(definitions.MEASURE_POINTS) <= registry
    assert {point for point, _ in definitions.MEASURE_POINTS.values()} == set(definitions.POINTS)
    assert set(definitions.POINT_DEFINITIONS) == set(definitions.POINTS)
    assert {point for point, _, _ in definitions.PROFILE_SCORES.values()} == set(definitions.POINTS)


def test_point_and_part_lookup() -> None:
    assert definitions.point_of("chances_share") == "Dominance"
    assert definitions.part_of("defensive_work_per_dominance") == "Cost per point of dominance"
    assert definitions.point_of("security") == "Efficiency" and definitions.point_of("unknown") is None
