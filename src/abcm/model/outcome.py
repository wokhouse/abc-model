"""Stage 1 — pre-train a game-outcome (home win probability) model.

Goal: learn "true win probability" from decades of game outcomes using only
pre-game features. The model's predicted ``home_win_prob`` for every game
(including the 2024-2025 labeled games) becomes the ``market_delta`` feature in
Stage 2 (``market_delta = stage1_prob - open_price``).

**The gate.** A reasonable NFL outcome model reaches ~63-68% accuracy and
Brier ~0.23. The hard requirement: at least one fitted model must beat Elo's
``elo_home_prob`` *alone* on the LOSO test fold (Elo baseline ~61% accuracy /
Brier ~0.24 on 2021-2023). If nothing beats Elo, the Stage 2 transfer has no
edge — :func:`build_and_save` reports ``gate_passed: False`` and Stage 2 should
not proceed.

Two models, per the plan:
* Penalized logistic regression — the small-n baseline that generalizes across
  eras more stably.
* XGBoost (depth-2, heavily regularized) with monotonic constraints encoding the
  domain priors: ``elo_diff``, ``pass_epa_diff``, ``rest_diff`` -> increasing.

**Null handling.** Deep-history tracking fields (EPA, CPOE, temp/wind) are
absent for older games, so ~13% of EPA-diff rows and 33% of CPOE rows are null.
We drop null rows when *training* (no fabricated values) but keep an Elo-only
logistic fallback so every game still gets a prediction at inference — deep
history games with missing EPA get a calibrated Elo probability rather than
being silently dropped.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config, io
from . import calibrate

# Ordered feature list used by BOTH models. Column order is load-bearing: the
# XGBoost ``monotone_constraints`` vector is positional, so the order here must
# match the design matrix returned by :func:`_prepare_frame` exactly. We keep
# ``elo_diff`` (pre-game, score-derived but not a leak) and deliberately exclude
# ``elo_home_prob`` to avoid feeding the Elo probability twice (it's the baseline
# we must beat, not a feature).
#
# ``spread_line`` is the nflverse closing consensus spread (positive = home
# favored). The SBR experiment showed the spread is a far better win predictor
# than Elo alone (Brier ~0.211 vs ~0.221 on 2,682 games), so adding it gives
# Stage 1 a much stronger baseline — the model then only needs to find the
# residual edge beyond what the market already prices.
#
# ``elo_diff`` was deliberately REMOVED: once ``spread_line`` is in the model,
# Elo adds *negative* value (spread-alone Brier 0.2151 vs spread+Elo 0.2158 on
# the 2021-2023 LOSO fold). The spread is the market's efficient team-strength
# summary; Elo is redundant noise on top of it. We still compute Elo (it's the
# Stage 1 gate baseline to beat, and the Elo-fallback scores pre-2006 games
# without a spread), but it's no longer a model feature.
FEATURES: list[str] = [
    "spread_line",
    "pass_epa_diff",
    "rush_epa_diff",
    "pass_cpoe_diff",
    "success_rate_diff",
    "yards_per_play_diff",
    "to_lost_diff",
    "rest_diff",
    "div_game",
    "neutral_site",
    "is_outdoor",
    "temp",
    "wind",
]

# Features whose XGBoost split direction is constrained (+1 = increasing).
# ``pass_epa_diff`` realizes the plan's "epa_diff" (no single ``epa_diff`` col).
# ``spread_line`` is +1: a bigger positive line (home favored more) -> higher
# home win prob.
MONOTONIC_FEATURES: dict[str, int] = {
    "spread_line": 1,
    "pass_epa_diff": 1,
    "rest_diff": 1,
}

# The plan's LOSO gate fold and the multi-fold variance estimate.
GATE_TRAIN_YEARS = list(range(1999, 2021))   # train 1999-2020
GATE_TEST_YEARS = [2021, 2022, 2023]         # test 2021-2023
MULTI_FOLDS = [
    (list(range(1999, 2019)), [2019, 2020, 2021]),
    (list(range(1999, 2020)), [2020, 2021, 2022]),
    (list(range(1999, 2021)), [2021, 2022, 2023]),
]
FINAL_TRAIN_YEARS = list(range(1999, 2024))  # final fit uses all 1999-2023


# ---------------------------------------------------------------------------
# Design-matrix preparation
# ---------------------------------------------------------------------------

def _prepare_frame(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X[FEATURES order], y, game_id) with a fixed feature column order.

    Coerces ``home_win`` to int8. Does NOT drop nulls (see :func:`_null_mask`);
    callers decide whether to drop. Reorders columns to ``FEATURES`` so the
    positional monotonic-constraint vector always lines up.
    """
    missing = [c for c in FEATURES if c not in features_df.columns]
    if missing:
        raise KeyError(f"feature frame missing required columns: {missing}")
    if "home_win" not in features_df.columns:
        raise KeyError("feature frame missing the 'home_win' label")
    X = features_df[FEATURES].copy()
    y = features_df["home_win"].astype("int8")
    ids = features_df["game_id"].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), ids


def _null_mask(X: pd.DataFrame) -> pd.Series:
    """Boolean Series: True where a row has any null feature value."""
    return X.isna().any(axis=1)


def monotonic_constraints() -> tuple[int, ...]:
    """Positional monotonic-constraint vector aligned with ``FEATURES``."""
    return tuple(MONOTONIC_FEATURES.get(f, 0) for f in FEATURES)


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------

def make_logreg(**overrides: Any):
    """Penalized logistic-regression baseline (plan hyperparameters).

    sklearn's default penalty is already L2; we set ``C=0.1`` for strong
    regularization and ``class_weight='balanced'`` (the plan's spec).
    """
    from sklearn.linear_model import LogisticRegression

    params: dict[str, Any] = dict(
        C=0.1, class_weight="balanced",
        max_iter=1000, random_state=config.STAGE1_RANDOM_STATE,
    )
    params.update(overrides)
    return LogisticRegression(**params)


def make_xgboost(**overrides: Any):
    """Heavily-regularized XGBoost with monotonic domain priors (plan params)."""
    from xgboost import XGBClassifier

    params: dict[str, Any] = dict(
        max_depth=2, learning_rate=0.03, n_estimators=500,
        subsample=0.6, colsample_bytree=0.5, min_child_weight=10,
        alpha=0.5, reg_lambda=2, objective="binary:logistic",
        eval_metric="logloss", monotone_constraints=monotonic_constraints(),
        random_state=config.STAGE1_RANDOM_STATE, n_jobs=1, verbosity=0,
    )
    params.update(overrides)
    return XGBClassifier(**params)


# ---------------------------------------------------------------------------
# Evaluation: the LOSO gate
# ---------------------------------------------------------------------------

def _score(probas: np.ndarray, y: pd.Series) -> dict[str, float]:
    from sklearn.metrics import accuracy_score

    preds = (np.asarray(probas) > 0.5).astype(int)
    return {
        "brier": calibrate.brier_score(probas, y.to_numpy()),
        "accuracy": float(accuracy_score(y.to_numpy(), preds)),
    }


def _split_nonull(
    features_df: pd.DataFrame, train_years: list[int], test_years: list[int]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the (X_train, y_train, X_test, y_test, meta) with null rows dropped.

    ``meta`` carries the rows' ``elo_home_prob`` (for the Elo baseline) and
    ``game_id`` so every metric is computed on the identical set of games across
    models and baseline.
    """
    X, y, ids = _prepare_frame(features_df)
    season = features_df["season"].reset_index(drop=True)
    elo_p = features_df["elo_home_prob"].reset_index(drop=True)
    valid = ~_null_mask(X)

    tr = season.isin(train_years) & valid
    te = season.isin(test_years) & valid
    return (
        X[tr], y[tr],
        X[te], y[te],
        pd.DataFrame({
            "game_id": ids[te],
            "elo_home_prob": elo_p[te],
        }).reset_index(drop=True),
    )


def lsoo_evaluate(
    features_df: pd.DataFrame,
    *,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> dict[str, Any]:
    """The "must beat Elo" gate evaluation on one LOSO fold.

    Fits LogReg + XGBoost on the train fold, predicts the test fold, and reports
    each model's Brier/accuracy alongside the Elo baseline on the *same* rows.
    Returns a dict with ``gate_passed`` set True iff at least one model beats Elo
    on Brier score (the primary metric; Brier rewards calibration, not just
    ranking).
    """
    train_years = GATE_TRAIN_YEARS if train_years is None else train_years
    test_years = GATE_TEST_YEARS if test_years is None else test_years
    X_tr, y_tr, X_te, y_te, meta = _split_nonull(features_df, train_years, test_years)
    if len(X_tr) == 0 or len(X_te) == 0:
        raise ValueError("empty train or test fold after null-drop")

    elo_scores = _score(meta["elo_home_prob"].to_numpy(), y_te)

    lr = make_logreg().fit(X_tr, y_tr)
    lr_scores = _score(lr.predict_proba(X_te)[:, 1], y_te)

    xg = make_xgboost().fit(X_tr, y_tr)
    xg_scores = _score(xg.predict_proba(X_te)[:, 1], y_te)

    gate_passed = bool(
        lr_scores["brier"] < elo_scores["brier"]
        or xg_scores["brier"] < elo_scores["brier"]
    )
    return {
        "train_years": (min(train_years), max(train_years)),
        "test_years": (min(test_years), max(test_years)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "elo": elo_scores,
        "logreg": lr_scores,
        "xgboost": xg_scores,
        "gate_passed": gate_passed,
    }


def lsoo_multi(features_df: pd.DataFrame, *, folds=None) -> dict[str, Any]:
    """Three rolling-expanding LOSO folds; mean +/- std so we can see if beating
    Elo is stable rather than a single-fold fluke."""
    folds = MULTI_FOLDS if folds is None else folds
    per_fold = [lsoo_evaluate(features_df, train_years=tr, test_years=te) for tr, te in folds]

    def _agg(metric: str, key: str) -> dict[str, float]:
        vals = np.array([f[key][metric] for f in per_fold])
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    return {
        "n_folds": len(folds),
        "elo_brier": _agg("brier", "elo"),
        "logreg_brier": _agg("brier", "logreg"),
        "xgboost_brier": _agg("brier", "xgboost"),
        "elo_accuracy": _agg("accuracy", "elo"),
        "logreg_accuracy": _agg("accuracy", "logreg"),
        "xgboost_accuracy": _agg("accuracy", "xgboost"),
    }


# ---------------------------------------------------------------------------
# Final fit + predict for every game (with Elo fallback)
# ---------------------------------------------------------------------------

def train_final_and_predict(
    features_df: pd.DataFrame,
    *,
    train_years: list[int] | None = None,
    final_model: str = "gate_winner",
    gate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the final model on all train-year non-null rows, score every game.

    ``final_model`` selects the scorer for non-null-feature rows:
    * ``"gate_winner"`` (default) — pick LogReg or XGBoost by lower gate Brier.
      Requires ``gate`` (the result of :func:`lsoo_evaluate`); if absent, falls
      back to ``"xgboost"``.
    * ``"logreg"`` / ``"xgboost"`` — force that model.

    Null-feature rows (deep history with missing EPA) fall back to an Elo-only
    logistic model (``LogisticRegression`` on ``elo_home_prob`` alone, fit on the
    same non-null training rows) so every game still gets a calibrated probability
    rather than being dropped.

    Returns the model bundle (for joblib persistence) and a predictions frame
    ``[game_id, home_win_prob, pred_method]``.
    """
    from sklearn.linear_model import LogisticRegression

    train_years = FINAL_TRAIN_YEARS if train_years is None else train_years
    X, y, ids = _prepare_frame(features_df)
    season = features_df["season"].reset_index(drop=True)
    elo_p = features_df["elo_home_prob"].reset_index(drop=True)
    valid = ~_null_mask(X)

    tr = season.isin(train_years) & valid
    X_tr, y_tr = X[tr], y[tr]

    # Decide which model scores the non-null rows. The research flagged that at
    # small n penalized logistic regression may beat XGBoost; in practice LogReg
    # is the consistent gate winner here, so we honor the gate rather than assume
    # the tree model is better.
    lr = make_logreg().fit(X_tr, y_tr)
    xg = make_xgboost().fit(X_tr, y_tr)
    if final_model == "logreg":
        chosen, chosen_name = lr, "logreg"
    elif final_model == "xgboost":
        chosen, chosen_name = xg, "xgboost"
    else:  # gate_winner
        if gate is not None and "logreg" in gate and "xgboost" in gate:
            chosen_name = "logreg" if gate["logreg"]["brier"] <= gate["xgboost"]["brier"] else "xgboost"
            chosen = lr if chosen_name == "logreg" else xg
        else:
            chosen, chosen_name = xg, "xgboost"

    # Elo-only fallback: calibrate raw elo_home_prob toward home_win.
    elo_cal = LogisticRegression(C=1.0, solver="lbfgs", random_state=config.STAGE1_RANDOM_STATE)
    elo_cal.fit(elo_p[tr].to_numpy().reshape(-1, 1), y_tr.to_numpy())

    # Score every game in chronological frame order.
    probas = np.full(len(X), np.nan)
    method = np.empty(len(X), dtype=object)
    nv = valid.to_numpy()
    if nv.any():
        probas[nv] = chosen.predict_proba(X[nv])[:, 1]
        method[nv] = f"final:{chosen_name}"
    null_idx = np.where(~nv)[0]
    if len(null_idx):
        # Some rows may have null elo_home_prob (pre-2006 history once Elo is
        # extended); those get the raw Elo mean as a last resort.
        ep = elo_p.to_numpy()
        has_elo = ~np.isnan(ep)
        score_idx = null_idx[has_elo[null_idx]]
        if len(score_idx):
            probas[score_idx] = elo_cal.predict_proba(ep[score_idx].reshape(-1, 1))[:, 1]
            method[score_idx] = "elo_fallback"
        miss = null_idx[~has_elo[null_idx]]
        if len(miss):
            probas[miss] = float(y_tr.mean())  # base rate prior
            method[miss] = "base_rate"

    preds = pd.DataFrame({
        "game_id": ids.to_numpy(),
        "home_win_prob": probas,
        "pred_method": method,
    })
    bundle = {
        "final_model": chosen, "final_model_name": chosen_name,
        "logreg": lr, "xgboost": xg, "elo_fallback": elo_cal,
        "features": list(FEATURES),
    }
    return bundle, preds


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_and_save() -> dict[str, Any]:
    """Run the gate, fit the final model, write predictions + model artifact.

    Mirrors the established ``build_and_save`` pattern. Prints a clear PASS/FAIL
    gate verdict and returns a summary dict carrying ``gate_passed`` so Stage 2
    can hard-fail when the outcome model adds no edge over Elo.
    """
    config.ensure_dirs()
    features = io.read_parquet(config.FEATURE_PROCESSED_DIR / "game_features.parquet")
    print(f"[outcome] loaded {len(features)} games from game_features.parquet")

    print("[outcome] running LOSO gate (train 1999-2020, test 2021-2023)...")
    gate = lsoo_evaluate(features)
    print(f"[outcome]   Elo baseline:  acc={gate['elo']['accuracy']:.3f}  brier={gate['elo']['brier']:.3f}")
    print(f"[outcome]   LogReg:        acc={gate['logreg']['accuracy']:.3f}  brier={gate['logreg']['brier']:.3f}")
    print(f"[outcome]   XGBoost:       acc={gate['xgboost']['accuracy']:.3f}  brier={gate['xgboost']['brier']:.3f}")
    verdict = "PASS" if gate["gate_passed"] else "FAIL"
    print(f"[outcome] GATE: {verdict} ({'a model beats Elo on Brier' if gate['gate_passed'] else 'no model beats Elo — Stage 2 has no transfer edge'})")

    print("[outcome] running multi-fold LOSO for variance estimate...")
    multi = lsoo_multi(features)
    print(f"[outcome]   Elo brier {multi['elo_brier']['mean']:.3f}±{multi['elo_brier']['std']:.3f} | "
          f"XGB brier {multi['xgboost_brier']['mean']:.3f}±{multi['xgboost_brier']['std']:.3f}")

    print("[outcome] fitting final model on 1999-2023 and predicting every game...")
    bundle, preds = train_final_and_predict(features, gate=gate)
    final_name = bundle["final_model_name"]
    print(f"[outcome]   gate winner selected as final scorer: {final_name}")

    pred_path = config.FEATURE_PROCESSED_DIR / "game_outcome_probs.parquet"
    io.write_parquet(preds, pred_path)
    model_path = config.MODELS_PROCESSED_DIR / "outcome_model.joblib"
    _save_bundle(bundle, model_path)

    n_final = int(preds["pred_method"].str.startswith("final").sum())
    n_elo = int((preds["pred_method"] == "elo_fallback").sum())
    n_base = int((preds["pred_method"] == "base_rate").sum())

    summary: dict[str, Any] = {
        "gate": gate,
        "gate_passed": gate["gate_passed"],
        "multi_fold": multi,
        "n_predictions": len(preds),
        "n_final_model": n_final,
        "n_elo_fallback": n_elo,
        "n_base_rate": n_base,
        "predictions_path": str(pred_path),
        "model_path": str(model_path),
    }
    print(f"[outcome] wrote {len(preds)} predictions -> {pred_path} "
          f"(final={n_final}, elo_fallback={n_elo}, base_rate={n_base})")
    print(f"[outcome] wrote model bundle -> {model_path}")
    return summary


def _save_bundle(bundle: dict[str, Any], path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
