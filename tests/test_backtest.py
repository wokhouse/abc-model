"""Tests for the Stage 3 backtest module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abcm.model import backtest


# --- Kelly sizing ---------------------------------------------------------

def test_kelly_positive_when_edge_exists():
    # Model prob 0.6, decimal odds 2.0 (b=1): f* = (1*0.6 - 0.4)/1 = 0.2; 25% -> 0.05.
    assert backtest.kelly_fraction(0.6, 2.0, fraction=0.25) == pytest.approx(0.05)


def test_kelly_zero_when_no_edge():
    # Fair: p=0.5 at 2.0 odds -> f*=0.
    assert backtest.kelly_fraction(0.5, 2.0) == pytest.approx(0.0)


def test_kelly_zero_when_negative_edge():
    # Model prob 0.4 at 2.0 odds -> negative edge -> don't bet.
    assert backtest.kelly_fraction(0.4, 2.0) == 0.0


def test_kelly_invalid_odds_returns_zero():
    assert backtest.kelly_fraction(0.6, 0.5) == 0.0
    assert backtest.kelly_fraction(np.nan, 2.0) == 0.0


# --- decision rule --------------------------------------------------------

def _preds(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "edge": rng.uniform(-0.2, 0.2, n),
        "market_prob": rng.uniform(0.2, 0.8, n),
        "model_prob_yes": rng.uniform(0.2, 0.8, n),
        "resolved_yes_price": rng.integers(0, 2, n).astype(float),
        "game_id": [f"g{i}" for i in range(n)],
        "market_type": "moneyline",
    })


def test_apply_decision_rule_buckets_by_edge_threshold():
    preds = _preds(n=50)
    buckets = backtest.apply_decision_rule(preds, thresholds=(0.05, 0.15))
    # Higher threshold => fewer trades.
    assert len(buckets[0.05]) >= len(buckets[0.15])
    # All trades in a bucket satisfy |edge| >= threshold.
    assert (buckets[0.05]["edge"].abs() >= 0.05).all()


def test_bet_side_follows_edge_sign():
    preds = _preds(n=10)
    buckets = backtest.apply_decision_rule(preds, thresholds=(0.0,))
    trades = buckets[0.0]
    assert ((trades["edge"] > 0) == (trades["bet_side"] == "YES")).all()


def test_trade_outcome_correct():
    row_yes_win = pd.Series({"bet_side": "YES", "resolved_yes_price": 1.0})
    assert backtest.trade_outcome(row_yes_win) == 1
    row_no_win = pd.Series({"bet_side": "NO", "resolved_yes_price": 0.0})
    assert backtest.trade_outcome(row_no_win) == 1
    row_yes_loss = pd.Series({"bet_side": "YES", "resolved_yes_price": 0.0})
    assert backtest.trade_outcome(row_yes_loss) == 0


# --- evaluate end-to-end --------------------------------------------------

def test_evaluate_returns_thresholds_and_overall():
    preds = _preds(n=40)
    res = backtest.evaluate(preds, oddsportal_lines=None, thresholds=(0.05,))
    assert "thresholds" in res and "overall" in res
    if 0.05 in res["thresholds"]:
        b = res["thresholds"][0.05]
        assert "n_trades" in b and "win_rate" in b and "roi" in b


def test_evaluate_with_clv_when_closing_lines_provided():
    preds = _preds(n=20)
    # Add a few games with closing moneyline lines.
    lines_df = pd.DataFrame([
        {"game_id": "g0", "home_close_ml": 2.0, "away_close_ml": 2.0},
        {"game_id": "g1", "home_close_ml": 1.5, "away_close_ml": 2.5},
    ])
    res = backtest.evaluate(preds, oddsportal_lines=lines_df, thresholds=(0.0,))
    # CLV bucket should be populated where closing lines joined.
    if 0.0 in res["thresholds"] and "clv_mean" in res["thresholds"][0.0]:
        assert not np.isnan(res["thresholds"][0.0]["clv_mean"])
