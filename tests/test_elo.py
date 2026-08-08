"""Tests for the Elo rating computation."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from abcm.features import elo


def _game(game_id, season, gameday, away, home, away_score, home_score, location="Home"):
    return {
        "game_id": game_id, "season": season, "gameday": gameday,
        "away_team": away, "home_team": home,
        "away_score": away_score, "home_score": home_score,
        "location": location,
    }


def test_expected_prob_equal_ratings_is_half_plus_hfa():
    """Equal ratings: home prob > 0.5 by the HFA contribution."""
    p = elo._expected_prob(1500, 1500, hfa=55)
    assert p == pytest.approx(1 / (1 + 10 ** (-55 / 400)))
    assert 0.57 < p < 0.60


def test_expected_prob_higher_home_rating_favored():
    p_high = elo._expected_prob(1600, 1500, hfa=0)
    p_low = elo._expected_prob(1500, 1600, hfa=0)
    assert p_high > 0.5
    assert p_low < 0.5
    assert p_high == pytest.approx(1 - p_low)


def test_k_factor_sane_range():
    """A typical NFL game yields K in the 20-70 range."""
    k = elo._k_factor(margin=3, winner_elo_diff=0)
    assert 20 < k < 35


def test_k_factor_favored_winning_barely_is_small():
    """A big favorite winning by little => small multiplier (penalize)."""
    small = elo._k_factor(margin=3, winner_elo_diff=150)
    big = elo._k_factor(margin=3, winner_elo_diff=-150)  # underdog upset
    assert small < big


def test_compute_elo_winner_gains_loser_drops():
    """After a game the winner's rating rises and the loser's falls."""
    sched = pd.DataFrame([
        _game("g1", 2024, "2024-09-10", "BUF", "MIA", 10, 30),  # MIA wins
    ])
    out = elo.compute_elo(sched)
    row = out.iloc[0]
    assert row["elo_home_post"] > row["elo_home_pre"]   # MIA (home) won
    assert row["elo_away_post"] < row["elo_away_pre"]   # BUF lost
    # Conservation: total Elo is unchanged (zero-sum update).
    pre_total = row["elo_home_pre"] + row["elo_away_pre"]
    post_total = row["elo_home_post"] + row["elo_away_post"]
    assert post_total == pytest.approx(pre_total, abs=1e-6)


def test_compute_elo_upset_moves_ratings_more_than_expected():
    """An underdog winning by a lot shifts ratings more than a favorite squeaking by."""
    base = pd.DataFrame([_game("g", 2024, "2024-09-10", "BUF", "MIA", 7, 3)])
    # Set up a strong favorite: give MIA a big prior by playing earlier games.
    hist = pd.DataFrame([
        _game("h1", 2023, "2023-09-10", "NE", "MIA", 0, 50),
        _game("h2", 2023, "2023-09-17", "NE", "MIA", 0, 50),
        _game("h3", 2023, "2023-09-24", "BUF", "MIA", 0, 50),
    ])
    # Big favorite MIA losing to BUF (upset) vs MIA winning narrowly.
    upset = pd.concat([hist, pd.DataFrame([_game("g", 2024, "2024-09-10", "MIA", "BUF", 30, 0)])], ignore_index=True)
    expected = pd.concat([hist, pd.DataFrame([_game("g", 2024, "2024-09-10", "MIA", "BUF", 0, 3)])], ignore_index=True)
    out_upset = elo.compute_elo(upset).iloc[-1]
    out_expected = elo.compute_elo(expected).iloc[-1]
    # In the upset (BUF wins), BUF's rating gain exceeds the narrow expected win.
    buf_gain_upset = out_upset["elo_away_post"] - out_upset["elo_away_pre"]
    buf_gain_expected = out_expected["elo_away_post"] - out_expected["elo_away_pre"]
    assert buf_gain_upset > buf_gain_expected


def test_compute_elo_season_regression_pulls_toward_mean():
    """Crossing into a new season regresses ratings toward the mean."""
    # One team wins big in 2023, inflating its rating; 2024 should regress it.
    sched = pd.DataFrame([
        _game("g1", 2023, "2023-09-10", "NE", "MIA", 0, 50),
        _game("g2", 2024, "2024-09-10", "BUF", "MIA", 10, 10),  # dummy game to observe
    ])
    out = elo.compute_elo(sched)
    mia_2023_end = out[out["season"] == 2023].iloc[-1]["elo_home_post"]  # MIA home in g1
    mia_2024_pre = out[out["season"] == 2024].iloc[0]["elo_home_pre"]    # MIA home in g2
    # Regression pulls the 2023-end rating back toward the mean (1505).
    assert mia_2024_pre < mia_2023_end
    assert mia_2024_pre > 1505  # still above mean, but regressed


def test_compute_elo_neutral_game_ignores_hfa():
    """A neutral game should not give the 'home' team the HFA edge."""
    sched = pd.DataFrame([
        _game("g1", 2024, "2024-09-10", "BUF", "MIA", 10, 10, location="Neutral"),
    ])
    out = elo.compute_elo(sched)
    # Equal teams, equal score at neutral: prob should be 0.5.
    assert out.iloc[0]["elo_home_prob"] == pytest.approx(0.5, abs=1e-9)


def test_compute_elo_skips_unplayed_games():
    """Games with missing scores are skipped (not rated)."""
    sched = pd.DataFrame([
        _game("g1", 2024, "2024-09-10", "BUF", "MIA", None, None),
    ])
    out = elo.compute_elo(sched)
    assert len(out) == 0
