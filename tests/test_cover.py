"""Tests for the Stage 1b cover-probability model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abcm.model import cover


def _game(game_id, season, *, home_score, away_score, spread_line, elo_diff=0.0,
          pass_epa_diff=0.0, rush_epa_diff=0.0, pass_cpoe_diff=0.0,
          success_rate_diff=0.0, yards_per_play_diff=0.0, to_lost_diff=0.0,
          rest_diff=0, div_game=0, neutral_site=0, is_outdoor=1, temp=70.0, wind=5.0):
    return {
        "game_id": game_id, "season": season, "home_score": home_score,
        "away_score": away_score, "spread_line": spread_line,
        "elo_diff": elo_diff, "pass_epa_diff": pass_epa_diff,
        "rush_epa_diff": rush_epa_diff, "pass_cpoe_diff": pass_cpoe_diff,
        "success_rate_diff": success_rate_diff, "yards_per_play_diff": yards_per_play_diff,
        "to_lost_diff": to_lost_diff, "rest_diff": rest_diff, "div_game": div_game,
        "neutral_site": neutral_site, "is_outdoor": is_outdoor, "temp": temp, "wind": wind,
    }


def _frame(n=80, base_season=2014, rng=None):
    rng = rng or np.random.default_rng(0)
    rows = []
    for i in range(n):
        spread = float(rng.choice([-7, -3.5, -1.5, 1.5, 3.5, 7]))
        margin = float(rng.normal(0, 10))
        rows.append(_game(
            f"g{i}", base_season + i // 40,
            home_score=int(round(20 + max(0, margin))),
            away_score=int(round(20 + max(0, -margin))),
            spread_line=spread,
            elo_diff=float(rng.normal(0, 80)),
            pass_epa_diff=float(rng.normal(0, 5)),
            rush_epa_diff=float(rng.normal(0, 4)),
            pass_cpoe_diff=float(rng.normal(0, 8)),
            success_rate_diff=float(rng.normal(0, 0.1)),
            yards_per_play_diff=float(rng.normal(0, 1.0)),
            to_lost_diff=float(rng.normal(0, 1.0)),
            rest_diff=int(rng.integers(-7, 8)),
        ))
    return pd.DataFrame(rows)


# --- cover label reconstruction ------------------------------------------
# nflverse convention: positive spread_line = home favored by that many.
# Home covers when (home_score - away_score) >= spread_line.

def test_cover_label_home_favored_covers_when_margin_reaches_line():
    """Home favored by 3.5 (spread_line=+3.5); wins by 7 (margin +7) -> covers."""
    df = pd.DataFrame([_game("g", 2020, home_score=27, away_score=20, spread_line=3.5)])
    _, y, _ = cover._prepare_cover_frame(df)
    assert y.iloc[0] == 1


def test_cover_label_home_favored_fails_to_cover():
    """Home favored by 7 (spread_line=+7); wins by 3 (margin +3) -> no cover."""
    df = pd.DataFrame([_game("g", 2020, home_score=24, away_score=21, spread_line=7.0)])
    _, y, _ = cover._prepare_cover_frame(df)
    assert y.iloc[0] == 0


def test_cover_label_away_favored_home_loses_but_covers():
    """Away favored (spread_line=-4); home loses by 1 (margin -1) -> covers,
    since -1 >= -4 (home got 4 points and only lost by 1)."""
    df = pd.DataFrame([_game("g", 2020, home_score=20, away_score=21, spread_line=-4.0)])
    _, y, _ = cover._prepare_cover_frame(df)
    assert y.iloc[0] == 1


def test_cover_label_push_boundary():
    """Margin exactly equals the line: covers (>=, not strict >). Half-point
    lines in real data avoid literal pushes, but the rule is >=."""
    df = pd.DataFrame([_game("g", 2020, home_score=27, away_score=20, spread_line=7.0)])
    _, y, _ = cover._prepare_cover_frame(df)
    assert y.iloc[0] == 1   # margin +7 >= line +7


def test_cover_label_null_line_gives_null_label():
    """A missing spread_line must yield a null label, not 0."""
    df = pd.DataFrame([_game("g", 2020, home_score=21, away_score=17, spread_line=np.nan)])
    _, y, _ = cover._prepare_cover_frame(df)
    assert pd.isna(y.iloc[0])


# --- feature set & monotonic constraints ----------------------------------

def test_cover_features_includes_spread_line():
    assert "spread_line" in cover.COVER_FEATURES
    assert set(cover.FEATURES).issubset(set(cover.COVER_FEATURES))


def test_cover_xgboost_monotonic_constraints_length_matches():
    mc = cover.make_cover_xgboost().get_params()["monotone_constraints"]
    assert len(mc) == len(cover.COVER_FEATURES)
    # spread_line is inherited as +1 from the outcome model's monotonic priors
    # (a bigger line -> more likely to cover). spread_line should appear exactly once.
    assert cover.COVER_FEATURES.count("spread_line") == 1
    assert mc[cover.COVER_FEATURES.index("spread_line")] == 1


def test_prepare_cover_frame_column_order():
    df = _frame(n=4)
    X, _, _ = cover._prepare_cover_frame(df)
    assert list(X.columns) == cover.COVER_FEATURES


# --- gate logic -----------------------------------------------------------

def test_lsoo_evaluate_returns_metrics_and_gate():
    df = _frame(n=120, base_season=2013)  # 3 seasons x 40
    res = cover.lsoo_evaluate(df, train_years=[2013, 2014], test_years=[2015])
    assert res["n_train"] > 0 and res["n_test"] > 0
    assert {"naive_base", "logreg", "xgboost", "gate_passed"} <= set(res)
    assert 0.0 <= res["logreg"]["brier"] <= 1.0
    assert isinstance(res["gate_passed"], bool)


def test_lsoo_drops_null_line_rows():
    df = _frame(n=60, base_season=2015)
    df.loc[30:44, "spread_line"] = np.nan   # null lines in the test fold
    res = cover.lsoo_evaluate(df, train_years=[2015], test_years=[2016])
    assert res["n_test"] == 15   # 30 test - 15 nulled


# --- final predict + fallback --------------------------------------------

def test_final_predict_assigns_method_and_covers_every_game():
    df = _frame(n=60, base_season=2014)
    df.loc[0, "spread_line"] = np.nan     # base_rate fallback
    bundle, preds = cover.train_final_and_predict(df, train_years=[2014, 2015, 2016])
    assert len(preds) == len(df)
    assert preds["home_cover_prob"].notna().all()
    assert preds["home_cover_prob"].between(0, 1).all()
    assert preds.loc[0, "pred_method"] == "base_rate"
    assert preds.loc[5, "pred_method"].startswith("final")
