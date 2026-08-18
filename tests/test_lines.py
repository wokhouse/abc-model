"""Tests for the line-movement + CLV module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abcm.model import lines


# --- decimal -> implied + devig ------------------------------------------

def test_decimal_to_implied_basic():
    assert lines.decimal_to_implied(2.0) == pytest.approx(0.5)
    assert lines.decimal_to_implied(1.25) == pytest.approx(0.8)


def test_decimal_to_implied_invalid_odds_returns_nan():
    assert np.isnan(lines.decimal_to_implied(0.5))
    assert np.isnan(lines.decimal_to_implied(np.nan))


def test_devig_removes_overround():
    # 1.91 / 1.91 has ~4.7% overround; fair should be 0.5 / 0.5.
    pa, pb = lines.devig(1.91, 1.91)
    assert pa == pytest.approx(0.5, abs=1e-6)
    assert pb == pytest.approx(0.5, abs=1e-6)
    assert pa + pb == pytest.approx(1.0)


def test_devig_asymmetric():
    # 1.20 favorite vs 5.0 dog.
    pa, pb = lines.devig(1.20, 5.0)
    assert pa > pb  # favorite side higher
    assert pa + pb == pytest.approx(1.0)


# --- favorite implied prob ------------------------------------------------

def test_favorite_implied_prob_picks_higher_side():
    fav = lines.favorite_implied_prob(1.30, 3.50)  # 1.30 side is favorite
    assert fav > 0.5


def test_favorite_implied_prob_nan_on_missing():
    assert np.isnan(lines.favorite_implied_prob(np.nan, 2.0))


# --- line movement --------------------------------------------------------

def test_ml_movement_positive_when_favorite_money_came_in():
    # Favorite tightens from 1.67->1.50 (more favored at close).
    mv = lines.ml_movement(home_open=1.67, away_open=2.20,
                           home_close=1.50, away_close=2.60)
    assert mv > 0  # favorite's fair prob grew


def test_ml_movement_negative_when_underdog_money():
    mv = lines.ml_movement(home_open=1.50, away_open=2.60,
                           home_close=1.67, away_close=2.20)
    assert mv < 0  # favorite faded


def test_ml_movement_nan_on_missing_odds():
    assert np.isnan(lines.ml_movement(np.nan, 2.0, 1.9, 2.0))


# --- compute_line_features on a frame ------------------------------------

def test_compute_line_features_adds_movement_columns():
    df = pd.DataFrame([{
        "home_open_ml": 1.67, "away_open_ml": 2.20,
        "home_close_ml": 1.50, "away_close_ml": 2.60,
        "spread_home_open_odds": 2.15, "spread_home_close_odds": 2.05,
        "over_open_odds": 1.76, "over_close_odds": 1.91,
    }])
    out = lines.compute_line_features(df)
    assert "ml_movement" in out and out.loc[0, "ml_movement"] > 0
    assert out.loc[0, "spread_odds_movement"] == pytest.approx(-0.10)
    assert out.loc[0, "total_odds_movement"] == pytest.approx(0.15)


def test_compute_line_features_handles_missing_columns():
    df = pd.DataFrame([{"home_open_ml": np.nan}])
    out = lines.compute_line_features(df)
    assert "ml_movement" in out
    assert np.isnan(out.loc[0, "ml_movement"])
    # spread/total movement columns absent when source columns absent.


# --- CLV ------------------------------------------------------------------

def test_clv_positive_when_model_above_market():
    # Closing 2.0/2.0 -> fair 0.5/0.5. Model says 0.6 -> CLV +0.1.
    clv = lines.clv(0.6, close_home_odds=2.0, close_away_odds=2.0, model_is_home=True)
    assert clv == pytest.approx(0.1, abs=1e-6)


def test_clv_negative_when_model_below_market():
    clv = lines.clv(0.4, close_home_odds=2.0, close_away_odds=2.0, model_is_home=True)
    assert clv == pytest.approx(-0.1, abs=1e-6)


def test_clv_respects_model_is_home_flag():
    # Away side fair prob 0.4 (1.20 fav / 2.5 dog -> home fav).
    clv_away = lines.clv(0.5, close_home_odds=1.20, close_away_odds=2.50, model_is_home=False)
    # away fair prob ~0.324; model 0.5 -> positive.
    assert clv_away > 0


def test_clv_nan_on_missing_odds():
    assert np.isnan(lines.clv(0.6, np.nan, 2.0))
