from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trivium.features import dominance as dom
from trivium.features import spread as sp

A = 1


def credits(rows: list[tuple]) -> pd.DataFrame:
    """rows: (shooter id, xg, shot x, shot y, passer id or None, pass x, pass y)."""
    out = []
    for shooter, xg, sx, sy, passer, px, py in rows:
        out.append({
            "tournament": "men", "match_id": 100, "team_id": A, "player_id": shooter, "xg": xg,
            "location_x": sx, "location_y": sy, "passer_team_id": A if passer is not None else np.nan,
            "passer_id": passer if passer is not None else np.nan, "pass_x": px, "pass_y": py,
        })
    return pd.DataFrame(out)


def test_entropy_is_zero_for_one_zone() -> None:
    assert sp.normalised_entropy(pd.Series([0.7])) == 0.0
    assert sp.normalised_entropy(pd.Series([0.7, 0.0])) == 0.0


def test_entropy_is_one_for_equal_spread() -> None:
    assert sp.normalised_entropy(pd.Series([0.1] * dom.N_ZONES)) == pytest.approx(1.0)


def test_entropy_of_nothing_is_missing() -> None:
    assert np.isnan(sp.normalised_entropy(pd.Series([0.0, 0.0])))


def test_chance_zone_spread_counts_shot_and_pass_zones() -> None:
    same_zone = credits([(9, 0.2, 110.0, 40.0, None, np.nan, np.nan),
                         (9, 0.3, 112.0, 35.0, None, np.nan, np.nan)])
    assert sp.chance_zone_spread(same_zone)["chance_zone_spread"].iloc[0] == 0.0

    # xG 0.4 in the shot's zone and xA 0.4 in the pass zone: two equal zones -> ln 2 / ln 18.
    split = credits([(9, 0.4, 110.0, 40.0, 7, 30.0, 10.0)])
    expected = np.log(2) / np.log(dom.N_ZONES)
    assert sp.chance_zone_spread(split)["chance_zone_spread"].iloc[0] == pytest.approx(expected)


def test_top2_share() -> None:
    c = credits([(9, 0.5, 110.0, 40.0, 7, 90.0, 40.0), (10, 0.2, 100.0, 40.0, None, np.nan, np.nan),
                 (11, 0.1, 100.0, 40.0, None, np.nan, np.nan)])
    threat = sp.player_threat(c)
    total = 0.5 + 0.2 + 0.1 + 0.5
    share = sp.top2_share(threat)["top2_xg_xa_share"].iloc[0]
    assert share == pytest.approx((0.5 + 0.5) / total)


def test_top2_share_missing_without_threat() -> None:
    threat = pd.DataFrame({"tournament": "men", "match_id": 100, "team_id": A, "player_id": [9],
                           "xg": [0.0], "xa": [0.0], "xg_xa": [0.0]})
    assert np.isnan(sp.top2_share(threat)["top2_xg_xa_share"].iloc[0])
