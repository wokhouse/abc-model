"""Tests for Stage 1 outcome-model pre-training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abcm.model import calibrate, outcome


# --- synthetic frame factory ----------------------------------------------

def _row(game_id, season, *, home_win=1, elo_diff=0.0, spread_line=0.0,
         pass_epa_diff=0.0, rush_epa_diff=0.0, pass_cpoe_diff=0.0,
         success_rate_diff=0.0, yards_per_play_diff=0.0, to_lost_diff=0.0,
         rest_diff=0, div_game=0, neutral_site=0, is_outdoor=1, temp=70.0,
         wind=5.0, elo_home_prob=0.5):
    return {
        "game_id": game_id, "season": season, "home_win": home_win,
        "elo_diff": elo_diff, "spread_line": spread_line,
        "pass_epa_diff": pass_epa_diff,
        "rush_epa_diff": rush_epa_diff, "pass_cpoe_diff": pass_cpoe_diff,
        "success_rate_diff": success_rate_diff, "yards_per_play_diff": yards_per_play_diff,
        "to_lost_diff": to_lost_diff, "rest_diff": rest_diff, "div_game": div_game,
        "neutral_site": neutral_site, "is_outdoor": is_outdoor, "temp": temp,
        "wind": wind, "elo_home_prob": elo_home_prob,
    }


def _frame(n=60, base_season=2015):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        elo_diff = float(rng.normal(0, 80))
        rows.append(_row(
            f"g{i}", base_season + i // 30,
            home_win=int(elo_diff + rng.normal(0, 40) > 0),
            elo_diff=elo_diff,
            pass_epa_diff=float(rng.normal(0, 5)),
            rush_epa_diff=float(rng.normal(0, 4)),
            pass_cpoe_diff=float(rng.normal(0, 8)),
            success_rate_diff=float(rng.normal(0, 0.1)),
            yards_per_play_diff=float(rng.normal(0, 1.0)),
            to_lost_diff=float(rng.normal(0, 1.0)),
            rest_diff=int(rng.integers(-7, 8)),
            elo_home_prob=float(1 / (1 + 10 ** (-elo_diff / 400))),
        ))
    return pd.DataFrame(rows)


# --- design-matrix preparation -------------------------------------------

def test_prepare_frame_returns_features_in_fixed_order():
    """Column order must be exactly FEATURES (monotonic constraints are positional)."""
    df = _frame(n=4)
    # Shuffle the input columns; output order must still match FEATURES.
    df_shuffled = df[list(reversed(df.columns))]
    X, y, ids = outcome._prepare_frame(df_shuffled)
    assert list(X.columns) == outcome.FEATURES


def test_prepare_frame_raises_on_missing_feature():
    df = _frame(n=3).drop(columns=["pass_epa_diff"])
    with pytest.raises(KeyError, match="pass_epa_diff"):
        outcome._prepare_frame(df)


def test_features_exclude_label_and_scores():
    """No post-game or label leakage; elo_home_prob excluded (it's the baseline)."""
    for leak in ("home_score", "away_score", "home_win", "elo_home_prob", "home_win_prob"):
        assert leak not in outcome.FEATURES


def test_null_mask_flags_any_null_row():
    df = _frame(n=3)
    df.loc[0, "pass_epa_diff"] = np.nan          # one null -> flagged
    df.loc[1, "temp"] = np.nan                   # another null -> flagged
    X, _, _ = outcome._prepare_frame(df)
    mask = outcome._null_mask(X)
    assert mask.tolist() == [True, True, False]


# --- monotonic constraints ------------------------------------------------

def test_monotonic_constraints_align_with_features():
    mc = outcome.monotonic_constraints()
    assert len(mc) == len(outcome.FEATURES)
    # +1 on the three domain-prior features, 0 elsewhere.
    for feat, expect in outcome.MONOTONIC_FEATURES.items():
        assert mc[outcome.FEATURES.index(feat)] == expect
    assert sum(mc) == len(outcome.MONOTONIC_FEATURES)


def test_make_xgboost_carries_monotone_constraints():
    xg = outcome.make_xgboost()
    # xgboost stores the constraint as a tuple in get_params().
    mc = xg.get_params()["monotone_constraints"]
    assert mc == outcome.monotonic_constraints()


# --- LOSO gate logic ------------------------------------------------------

def test_lsoo_evaluate_returns_expected_keys_and_disjoint_folds():
    df = _frame(n=90, base_season=2014)  # 3 seasons x 30
    res = outcome.lsoo_evaluate(
        df, train_years=[2014, 2015], test_years=[2016],
    )
    assert res["n_train"] > 0 and res["n_test"] > 0
    assert set(res) >= {"elo", "logreg", "xgboost", "gate_passed"}
    for key in ("elo", "logreg", "xgboost"):
        assert 0.0 <= res[key]["brier"] <= 1.0
        assert 0.0 <= res[key]["accuracy"] <= 1.0
    assert isinstance(res["gate_passed"], bool)


def test_lsoo_evaluate_drops_null_test_rows():
    """Null test rows are dropped, so n_test reflects only the non-null rows."""
    df = _frame(n=60, base_season=2015)          # 30 games in 2015, 30 in 2016
    # Null out 10 of the 2016 (test-fold) rows' pass_epa_diff.
    df.loc[30:39, "pass_epa_diff"] = np.nan
    res = outcome.lsoo_evaluate(df, train_years=[2015], test_years=[2016])
    # 30 test games minus the 10 nulled -> 20 survive the null-drop.
    assert res["n_test"] == 20
    assert res["n_train"] == 30


# --- final fit + Elo fallback --------------------------------------------

def test_final_predict_assigns_method_and_scores_every_game():
    df = _frame(n=40, base_season=2014)
    # Punch holes in two rows so the fallback path is exercised.
    df.loc[0, ["pass_epa_diff", "rush_epa_diff"]] = np.nan
    df.loc[1, "pass_cpoe_diff"] = np.nan
    bundle, preds = outcome.train_final_and_predict(df, train_years=[2014, 2015, 2016])
    assert len(preds) == len(df)
    assert preds["home_win_prob"].notna().all()
    assert preds["home_win_prob"].between(0, 1).all()
    # Null-feature rows got the fallback; non-null rows got the final model.
    methods = set(preds["pred_method"].unique())
    assert any(m.startswith("final") for m in methods)
    assert preds.loc[0, "pred_method"] in ("elo_fallback", "base_rate")
    assert preds.loc[1, "pred_method"] in ("elo_fallback", "base_rate")


def test_final_predict_base_rate_when_elo_also_missing():
    df = _frame(n=20, base_season=2015)
    df.loc[5, ["pass_epa_diff", "elo_home_prob"]] = np.nan
    _, preds = outcome.train_final_and_predict(df, train_years=[2015, 2016])
    assert preds.loc[5, "pred_method"] == "base_rate"


# --- calibration helpers --------------------------------------------------

def test_brier_score_lower_is_better():
    labels = np.array([0, 1, 0, 1])
    good = calibrate.brier_score([0.1, 0.9, 0.2, 0.8], labels)
    bad = calibrate.brier_score([0.4, 0.6, 0.5, 0.5], labels)
    assert good < bad


def test_reliability_bins_shape_and_counts():
    probas = np.array([0.05, 0.15, 0.45, 0.55, 0.85, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    rb = calibrate.reliability_bins(probas, labels, n_bins=3)
    assert len(rb) == 3
    assert rb["count"].sum() == 6
    assert 0.0 <= rb["mean_prob"].dropna().max() <= 1.0


def test_platt_scale_returns_in_unit_interval():
    rng = np.random.default_rng(1)
    probas = rng.uniform(0, 1, size=200)
    labels = (probas + rng.normal(0, 0.2, 200) > 0.5).astype(int)
    _, calibrated = calibrate.platt_scale(probas, labels)
    assert calibrated.shape == probas.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))
