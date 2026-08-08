"""Tests for travel distance and timezone features."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from abcm.features import travel


def test_team_locations_covers_32_teams():
    """The bundled table must cover every active nflverse team."""
    df = travel.load_team_locations()
    assert len(df) == 32
    expected = {
        "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
        "HOU","IND","JAX","KC","LV","LA","LAC","MIA","MIN","NE","NO","NYG","NYJ",
        "PHI","PIT","SF","SEA","TB","TEN","WAS",
    }
    assert set(df["team_abbr"]) == expected


def test_haversine_known_distance():
    """A roughly known distance: MIA to SEA is ~4400 km."""
    d = travel._haversine_km(25.9579, -80.2388, 47.5952, -122.3316)
    assert 4200 < d < 4600


def test_haversine_same_point_is_zero():
    d = travel._haversine_km(40.0, -74.0, 40.0, -74.0)
    assert d == pytest.approx(0.0, abs=1e-6)


def test_game_travel_features_basic():
    """A home game computes distance from away city to home city."""
    schedules = pd.DataFrame([{
        "game_id": "g1", "season": 2024, "week": 1,
        "away_team": "MIA", "home_team": "BUF",
        "location": "Home",
    }])
    out = travel.game_travel_features(schedules)
    row = out.iloc[0]
    assert row["travel_distance_km"] > 0
    # MIA (-5) to BUF (-5): same timezone, zero shift.
    assert row["timezone_shift_h"] == 0.0
    assert not row["neutral_site"]


def test_timezone_shift_positive_going_east():
    """SEA traveling to an Eastern team: shift is +3 (west -> east)."""
    schedules = pd.DataFrame([{
        "game_id": "g1", "away_team": "SEA", "home_team": "NE",
        "location": "Home",
    }])
    out = travel.game_travel_features(schedules)
    # NE (-5) - SEA (-8) = +3
    assert out.iloc[0]["timezone_shift_h"] == 3.0


def test_neutral_site_flagged():
    """location='Neutral' sets neutral_site True."""
    schedules = pd.DataFrame([{
        "game_id": "g1", "away_team": "JAX", "home_team": "BUF",
        "location": "Neutral",
    }])
    out = travel.game_travel_features(schedules)
    assert bool(out.iloc[0]["neutral_site"]) is True


def test_unknown_team_yields_null_travel():
    """A team not in the reference table gets null travel features."""
    schedules = pd.DataFrame([{
        "game_id": "g1", "away_team": "XXX", "home_team": "BUF",
        "location": "Home",
    }])
    out = travel.game_travel_features(schedules)
    row = out.iloc[0]
    assert pd.isna(row["travel_distance_km"])
    assert pd.isna(row["timezone_shift_h"])


def test_travel_features_preserve_input_rows():
    """No rows are dropped or added."""
    schedules = pd.DataFrame([
        {"game_id": "g1", "away_team": "MIA", "home_team": "BUF", "location": "Home"},
        {"game_id": "g2", "away_team": "SEA", "home_team": "SF", "location": "Home"},
    ])
    out = travel.game_travel_features(schedules)
    assert len(out) == len(schedules)
