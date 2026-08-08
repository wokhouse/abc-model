"""Stage 1b — pre-train a game-cover (against-the-spread) probability model.

Stage 2's spread markets are about *covering a margin*, not winning outright.
Stage 1 only produces a home **win** probability, so it cannot supply a spread
``model_prob``. This module fills that gap with a second Stage-1-style model
trained to predict whether the home team will cover the closing spread.

**Cover label.** nflverse ``spread_line`` is the home team's line; a **positive**
line means the **home** team is favored by that many (verified against the
labeled markets — positive lines have ~89% home win rate). Home covers when its
margin of victory reaches the line:

    home_covered = (home_score - away_score) >= spread_line

This reconstructs the home-perspective cover ~95.5% of the time vs the
Polymarket ``resolved_yes_price`` (the residual is closing-line movement), and
the deep history has 6,423 trainable REG games 1999-2023 with ``spread_line`` +
scores at 100% coverage — so, unlike the 41-row spread labeled fold, this
pre-training is well-powered.

The cover line itself is a feature: the model must learn how matchup strength
translates to beating a *specific* number, not just winning. Output is one
``home_cover_prob`` per game (all 7,276); the 2024-2025 rows feed Stage 2's
spread ``model_prob_yes``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config, io
from . import calibrate
from .outcome import (
    FEATURES, MONOTONIC_FEATURES, _null_mask, _save_bundle,
    make_logreg, make_xgboost, monotonic_constraints,
)

# Cover features = the outcome feature set + the spread line itself. The line is
# the number to beat, so it carries the bulk of the cover signal.
COVER_FEATURES: list[str] = list(FEATURES) + ["spread_line"]

# ``spread_line`` is the only new feature; the monotonic priors from the outcome
# model carry over (more elo/epa/rest -> more likely to cover any given line).
# The line itself is *not* monotonic in cover probability in a fixed direction
# (a larger line to beat cuts both ways once you condition on team strength), so
# it stays unconstrained.
COVER_MONOTONIC: dict[str, int] = dict(MONOTONIC_FEATURES)

GATE_TRAIN_YEARS = list(range(1999, 2021))
GATE_TEST_YEARS = [2021, 2022, 2023]
MULTI_FOLDS = [
    (list(range(1999, 2019)), [2019, 2020, 2021]),
    (list(range(1999, 2020)), [2020, 2021, 2022]),
    (list(range(1999, 2021)), [2021, 2022, 2023]),
]
FINAL_TRAIN_YEARS = list(range(1999, 2024))


def _attach_spread_line(features_df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``spread_line`` (and ``result``) onto the game feature frame.

    ``spread_line`` comes from nflverse schedules (home-perspective). Missing
    lines (playoff games, a few gaps) become null rows that drop out of training.
    """
    cols = ["game_id"]
    for c in ("spread_line", "result"):
        if c in schedules.columns:
            cols.append(c)
    out = features_df.merge(schedules[cols], on="game_id", how="left")
    return out


def _prepare_cover_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X[COVER_FEATURES order], home_covered label, game_id).

    Builds the home-perspective cover label from final scores and ``spread_line``.
    nflverse sign convention (verified against the labeled markets): a **positive**
    ``spread_line`` means the **home** team is favored by that many, so home covers
    when its margin of victory reaches the line::

        home_covered = (home_score - away_score) >= spread_line

    This agrees with the Polymarket ``resolved_yes_price`` ~95.5% of the time; the
    residual gap is closing-line movement vs the Polymarket snapshot margin and is
    irreducible noise for the historical training label. Rows without a line get a
    null label (handled by callers, not silently trained as 0).
    """
    missing = [c for c in COVER_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"cover frame missing required columns: {missing}")
    df = df.copy()
    margin = df["home_score"] - df["away_score"]
    # Only assign a label where the line is present; (margin >= NaN) is False, not
    # null, so build the label explicitly to keep nulls as null.
    home_covered = pd.Series([pd.NA] * len(df), dtype="Int64", index=df.index)
    has_line = df["spread_line"].notna()
    home_covered[has_line] = (margin[has_line] >= df["spread_line"][has_line]).astype("int8")
    df["home_covered"] = home_covered
    X = df[COVER_FEATURES].copy()
    y = df["home_covered"].astype("Int64")
    ids = df["game_id"].reset_index(drop=True)
    return X.reset_index(drop=True), y.reset_index(drop=True), ids


def _score(probas: np.ndarray, y: pd.Series) -> dict[str, float]:
    from sklearn.metrics import accuracy_score

    preds = (np.asarray(probas) > 0.5).astype(int)
    return {
        "brier": calibrate.brier_score(probas, y.to_numpy()),
        "accuracy": float(accuracy_score(y.to_numpy(), preds)),
    }


def make_cover_logreg(**overrides: Any):
    """Penalized logistic regression on the cover feature set."""
    lr = make_logreg()
    params = lr.get_params()
    params.update(overrides)
    lr.set_params(**params)
    return lr


def make_cover_xgboost(**overrides: Any):
    """XGBoost on the cover feature set; monotonic priors carry over (line is free)."""
    mc = tuple(COVER_MONOTONIC.get(f, 0) for f in COVER_FEATURES)
    return make_xgboost(monotone_constraints=mc, **overrides)


def _split_nonull(
    df: pd.DataFrame, train_years: list[int], test_years: list[int]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X, y, _ = _prepare_cover_frame(df)
    season = df["season"].reset_index(drop=True)
    valid = (~_null_mask(X)) & y.notna()

    tr = season.isin(train_years) & valid
    te = season.isin(test_years) & valid
    return X[tr], y[tr].astype("int8"), X[te], y[te].astype("int8")


def _baseline_home_cover_rate(y: pd.Series) -> float:
    """Naive baseline: the historical home-cover rate (~0.48). Used as the bar
    the cover model's Brier must beat, mirroring Elo's role in Stage 1."""
    return float(y.mean())


def lsoo_evaluate(
    df: pd.DataFrame,
    *,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> dict[str, Any]:
    """The cover-model gate on one LOSO fold (Brier/accuracy vs the naive base rate)."""
    train_years = GATE_TRAIN_YEARS if train_years is None else train_years
    test_years = GATE_TEST_YEARS if test_years is None else test_years
    X_tr, y_tr, X_te, y_te = _split_nonull(df, train_years, test_years)
    if len(X_tr) == 0 or len(X_te) == 0:
        raise ValueError("empty train or test fold after null-drop")

    base_rate = _baseline_home_cover_rate(y_tr)
    base_brier = float(np.mean((base_rate - y_te.to_numpy()) ** 2))
    base_acc = float(max(base_rate, 1 - base_rate)) if base_rate not in (0.0, 1.0) else float(base_rate)
    base = {"brier": base_brier, "accuracy": base_acc, "base_rate": base_rate}

    lr = make_cover_logreg().fit(X_tr, y_tr)
    lr_scores = _score(lr.predict_proba(X_te)[:, 1], y_te)

    xg = make_cover_xgboost().fit(X_tr, y_tr)
    xg_scores = _score(xg.predict_proba(X_te)[:, 1], y_te)

    gate_passed = bool(
        lr_scores["brier"] < base_brier or xg_scores["brier"] < base_brier
    )
    return {
        "train_years": (min(train_years), max(train_years)),
        "test_years": (min(test_years), max(test_years)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "naive_base": base,
        "logreg": lr_scores,
        "xgboost": xg_scores,
        "gate_passed": gate_passed,
    }


def lsoo_multi(df: pd.DataFrame, *, folds=None) -> dict[str, Any]:
    folds = MULTI_FOLDS if folds is None else folds
    per_fold = [lsoo_evaluate(df, train_years=tr, test_years=te) for tr, te in folds]

    def _agg(metric: str, key: str) -> dict[str, float]:
        vals = np.array([f[key][metric] for f in per_fold])
        return {"mean": float(vals.mean()), "std": float(vals.std())}

    return {
        "n_folds": len(folds),
        "base_brier": _agg("brier", "naive_base"),
        "logreg_brier": _agg("brier", "logreg"),
        "xgboost_brier": _agg("brier", "xgboost"),
        "logreg_accuracy": _agg("accuracy", "logreg"),
        "xgboost_accuracy": _agg("accuracy", "xgboost"),
    }


def train_final_and_predict(
    df: pd.DataFrame,
    *,
    train_years: list[int] | None = None,
    gate: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the gate-winning cover model on 1999-2023, score every game.

    Mirrors :func:`outcome.train_final_and_predict`: null-line or null-feature
    rows fall back to the historical home-cover base rate so every game still
    gets a probability (no fabricated line on playoff/missing-line games).
    """
    train_years = FINAL_TRAIN_YEARS if train_years is None else train_years
    X, y, ids = _prepare_cover_frame(df)
    season = df["season"].reset_index(drop=True)
    valid = (~_null_mask(X)) & y.notna()

    tr = season.isin(train_years) & valid
    X_tr, y_tr = X[tr], y[tr].astype("int8")

    lr = make_cover_logreg().fit(X_tr, y_tr)
    xg = make_cover_xgboost().fit(X_tr, y_tr)
    if gate is not None and "logreg" in gate and "xgboost" in gate:
        chosen_name = "logreg" if gate["logreg"]["brier"] <= gate["xgboost"]["brier"] else "xgboost"
        chosen = lr if chosen_name == "logreg" else xg
    else:
        chosen, chosen_name = xg, "xgboost"

    base_rate = float(y_tr.mean())
    probas = np.full(len(X), base_rate)  # sensible prior for missing-line rows
    method = np.full(len(X), "base_rate", dtype=object)
    nv = valid.to_numpy()
    if nv.any():
        probas[nv] = chosen.predict_proba(X[nv])[:, 1]
        method[nv] = f"final:{chosen_name}"

    preds = pd.DataFrame({
        "game_id": ids.to_numpy(),
        "home_cover_prob": probas,
        "pred_method": method,
    })
    bundle = {
        "final_model": chosen, "final_model_name": chosen_name,
        "logreg": lr, "xgboost": xg, "features": list(COVER_FEATURES),
        "base_rate": base_rate,
    }
    return bundle, preds


def build_and_save() -> dict[str, Any]:
    """Run the cover-model gate, fit the final model, write predictions + artifact."""
    config.ensure_dirs()
    features = io.read_parquet(config.FEATURE_PROCESSED_DIR / "game_features.parquet")
    schedules = io.read_parquet(config.NFLVERSE_HISTORY_DIR / "schedules.parquet")
    df = _attach_spread_line(features, schedules)
    print(f"[cover] loaded {len(df)} games; spread_line non-null: {df['spread_line'].notna().sum()}")

    print("[cover] running LOSO gate (train 1999-2020, test 2021-2023)...")
    gate = lsoo_evaluate(df)
    print(f"[cover]   Naive base:   brier={gate['naive_base']['brier']:.3f}  (base rate {gate['naive_base']['base_rate']:.3f})")
    print(f"[cover]   LogReg:       acc={gate['logreg']['accuracy']:.3f}  brier={gate['logreg']['brier']:.3f}")
    print(f"[cover]   XGBoost:      acc={gate['xgboost']['accuracy']:.3f}  brier={gate['xgboost']['brier']:.3f}")
    verdict = "PASS" if gate["gate_passed"] else "FAIL"
    print(f"[cover] GATE: {verdict}")

    multi = lsoo_multi(df)
    print(f"[cover]   base brier {multi['base_brier']['mean']:.3f}±{multi['base_brier']['std']:.3f} | "
          f"XGB brier {multi['xgboost_brier']['mean']:.3f}±{multi['xgboost_brier']['std']:.3f}")

    print("[cover] fitting final model on 1999-2023 and predicting every game...")
    bundle, preds = train_final_and_predict(df, gate=gate)
    print(f"[cover]   gate winner: {bundle['final_model_name']}")

    pred_path = config.FEATURE_PROCESSED_DIR / "game_cover_probs.parquet"
    io.write_parquet(preds, pred_path)
    model_path = config.MODELS_PROCESSED_DIR / "cover_model.joblib"
    _save_bundle(bundle, model_path)

    n_final = int(preds["pred_method"].str.startswith("final").sum())
    n_base = int((preds["pred_method"] == "base_rate").sum())
    summary: dict[str, Any] = {
        "gate": gate, "gate_passed": gate["gate_passed"], "multi_fold": multi,
        "n_predictions": len(preds), "n_final_model": n_final, "n_base_rate": n_base,
        "predictions_path": str(pred_path), "model_path": str(model_path),
    }
    print(f"[cover] wrote {len(preds)} predictions -> {pred_path} (final={n_final}, base_rate={n_base})")
    print(f"[cover] wrote model bundle -> {model_path}")
    return summary
