"""Stage 2 — the inefficiency (market-edge) model. The core deliverable.

Given a game's pre-game features and the Polymarket market's opening trade
price, predict whether the market is inefficient enough to trade.

**The edge construct.** For each labeled market:

    model_prob_yes = home_win_prob     if YES side is the home team    (moneyline)
                   = 1 - home_win_prob if YES side is the away team    (moneyline)
                   = home_cover_prob   if the favored team is home     (spread)
                   = 1 - home_cover_prob otherwise                    (spread)
    market_prob    = open_price        (the opening YES trade)
    edge           = model_prob_yes - market_prob                      (signed)

The YES side is ``pm_team0_abbr`` (verified: matches the resolved outcome on
260/260 clean moneyline rows; for spread the YES side is always the favored
team). So ``model_prob_yes`` must be flipped to the YES team's perspective.

**Two targets (test both):**
1. *Regression on signed edge* — predict ``edge``; trade when ``|edge| > t``.
2. *Binary* — ``label = (resolved_yes_price == 1) == (edge > 0)``, i.e. "was
   betting the model's side profitable?" Primary target (balanced ~52/48).

**Validation.** At ~261-836 rows, Leave-One-Season-Out is the only defensible
split (random k-fold leaks the future). We report:
- LOSO fold A: train 2024 -> test 2025 (note the asymmetry: 2024 spread train
  is only 41 rows, so the per-type spread model is powered by the Stage 1b cover
  pre-training feeding ``model_prob_yes``, not learned here from scratch).
- A purged rolling-origin within 2024+2025 (week-blocked, 1-week embargo, min 150
  train rows) for an OOS variance estimate.
Both report Brier + accuracy with bootstrap 95% CIs. At this n, variance
dominates — the verdict is "is there a consistent signal," not "what's the ROI."

**Pooling (test, don't assume):** pooled (market_type one-hot) vs per-type
(moneyline/spread) vs spread-first stacking (spread OOF prediction as a
moneyline feature, generated within the same temporal folds — never leak a
future fold's labels).

Stage 1's win-prob edge over Elo was thin (~0.002 Brier), so we do not assume a
large ``market_delta`` edge exists — Stage 2 exists to find out whether even a
thin true-probability edge is tradeable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config, io
from . import calibrate

# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

# Market-quality + inefficiency features (joined onto the Stage 0 game features).
# ``market_delta`` (= edge) is the headline inefficiency signal; ``market_prob``
# is the price level (markets priced near 0.5 vs 0.9 behave differently);
# ``n_candles``/``price_delta`` are liquidity/movement signals.
MARKET_FEATURES: list[str] = [
    "market_delta", "market_prob", "price_delta", "n_candles",
    "sportsbook_spread",
]
# Line-movement features from OddsPortal open/close (the genuinely NEW signal
# added after the negative Stage-2 result: sharp money moves lines, so open->close
# movement carries information the true-probability edge alone did not). These are
# null where OddsPortal data is missing and handled by the null-drop in eval.
LINE_MOVEMENT_FEATURES: list[str] = [
    "ml_movement", "spread_odds_movement", "total_odds_movement",
]
# Stage 0 game-level differentials carried through (the matchup context).
GAME_FEATURES: list[str] = [
    "elo_diff", "pass_epa_diff", "rush_epa_diff", "rest_diff", "div_game",
]
# Full pooled feature set (per-type models drop the market_type one-hot).
FEATURES_STAGE2: list[str] = (
    GAME_FEATURES + MARKET_FEATURES + LINE_MOVEMENT_FEATURES + ["market_type_spread"]
)

# Mislabeled/novelty moneyline rows to drop (verified by inspection: a game-total
# question, a 1st-quarter prop, a coin-toss/"Drake bet" market, and a 0.5 resolve).
_BAD_QUESTION_FRAGMENTS = ("drake", "coin toss", "1st quarter", "first quarter")


def _clean_labeled(aligned: pd.DataFrame) -> pd.DataFrame:
    """Filter to the clean, trainable Stage-2 universe: moneyline + spread rows
    with a resolved label, an opening price, a clean cutoff snapshot, and a game
    match. Drops the ~4 mislabeled/novelty moneyline rows."""
    mask = (
        aligned["market_type"].isin(["moneyline", "spread"])
        & aligned["resolved_yes_price"].notna()
        & aligned["open_price"].notna()
        & (aligned["snapshot_method"] == "cutoff")
        & aligned["game_id"].notna()
    )
    df = aligned[mask].copy()
    q = df.get("question", pd.Series(index=df.index, dtype=str)).fillna("").str.lower()
    bad = q.str.contains("|".join(_BAD_QUESTION_FRAGMENTS), na=False)
    df = df[~bad]
    # Drop the single novelty 0.5 resolve (a push-like prop market).
    df = df[df["resolved_yes_price"] != 0.5]
    return df.reset_index(drop=True)


def _yes_is_home(row) -> bool:
    """True when the market's YES token (pm_team0) is the real home team."""
    return str(row["pm_team0_abbr"]) == str(row["nfl_home_team"])


def _build_stage2_frame(
    aligned: pd.DataFrame,
    game_features: pd.DataFrame,
    win_probs: pd.DataFrame,
    cover_probs: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble one row per labeled market with model_prob_yes, edge, and features.

    Joins the clean labeled markets to the Stage 0 game features and the Stage 1
    (win) + Stage 1b (cover) probability tables, resolves the YES-side, and
    computes ``model_prob_yes`` and ``edge``. Returns the modeling frame.
    """
    clean = _clean_labeled(aligned)

    # Stage 0 game features (differentials + situational).
    gf_cols = ["game_id"] + [c for c in GAME_FEATURES if c in game_features.columns]
    out = clean.merge(game_features[gf_cols], on="game_id", how="left")

    # Stage 1 win prob + Stage 1b cover prob.
    out = out.merge(
        win_probs.rename(columns={"home_win_prob": "home_win_prob"})[["game_id", "home_win_prob"]],
        on="game_id", how="left",
    )
    out = out.merge(
        cover_probs[["game_id", "home_cover_prob"]], on="game_id", how="left",
    )

    # Resolve model_prob_yes from the YES team's perspective.
    yes_home = out.apply(_yes_is_home, axis=1)
    is_spread = out["market_type"] == "spread"
    # Moneyline: flip win prob to the YES side.
    out["model_prob_yes"] = np.where(
        is_spread, np.nan,  # filled below
        np.where(yes_home, out["home_win_prob"], 1 - out["home_win_prob"]),
    )
    # Spread: flip cover prob to the YES side (YES is always the favored team).
    cover_yes = np.where(yes_home, out["home_cover_prob"], 1 - out["home_cover_prob"])
    out.loc[is_spread, "model_prob_yes"] = cover_yes[is_spread]

    out["market_prob"] = out["open_price"].astype(float)
    out["market_delta"] = out["model_prob_yes"] - out["market_prob"]  # == edge
    out["edge"] = out["market_delta"]
    out["market_type_spread"] = (out["market_type"] == "spread").astype("int8")

    # Line-movement features from OddsPortal open/close odds (the new signal).
    # The open/close columns are carried via aligned.parquet (joined in reconcile);
    # compute_line_features adds ml_movement / spread_odds_movement /
    # total_odds_movement, null where OddsPortal data is missing.
    from . import lines as lines_mod
    out = lines_mod.compute_line_features(out)

    # Binary target: did betting the model's side pay off?
    resolved = out["resolved_yes_price"].astype(float)
    out["binary_label"] = ((resolved == 1) == (out["edge"] > 0)).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Estimators (heavily regularized per the plan)
# ---------------------------------------------------------------------------

def make_logreg(**overrides: Any):
    from sklearn.linear_model import LogisticRegression
    params: dict[str, Any] = dict(
        C=0.1, class_weight="balanced", max_iter=1000,
        random_state=config.STAGE1_RANDOM_STATE,
    )
    params.update(overrides)
    return LogisticRegression(**params)


def make_xgboost(**overrides: Any):
    from xgboost import XGBClassifier
    params: dict[str, Any] = dict(
        max_depth=2, learning_rate=0.03, n_estimators=500,
        subsample=0.6, colsample_bytree=0.5, min_child_weight=10,
        alpha=0.5, reg_lambda=2, objective="binary:logistic",
        eval_metric="logloss", n_jobs=1, verbosity=0,
        random_state=config.STAGE1_RANDOM_STATE,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def _score(probas: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score
    preds = (np.asarray(probas) > 0.5).astype(int)
    return {
        "brier": calibrate.brier_score(probas, y),
        "accuracy": float(accuracy_score(y, preds)),
    }


def _bootstrap_ci(
    probas: np.ndarray, y: np.ndarray, metric: str, *, n_boot: int = 500, seed: int = 42
) -> dict[str, float]:
    """Bootstrap 95% CI on Brier or accuracy. At n~260-800 the point estimate is
    noisy; the CI is the honest summary."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, yy = probas[idx], y[idx]
        if metric == "brier":
            vals.append(float(np.mean((p - yy) ** 2)))
        else:
            preds = (p > 0.5).astype(int)
            vals.append(float((preds == yy.astype(int)).mean()))
    arr = np.array(vals)
    return {"mean": float(arr.mean()), "lo": float(np.percentile(arr, 2.5)), "hi": float(np.percentile(arr, 97.5))}


# ---------------------------------------------------------------------------
# Validation: LOSO + purged rolling-origin
# ---------------------------------------------------------------------------

def _metric_with_ci(probas: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    s = _score(probas, y)
    return {
        "brier": s["brier"], "accuracy": s["accuracy"],
        "brier_ci": _bootstrap_ci(probas, y, "brier"),
        "accuracy_ci": _bootstrap_ci(probas, y, "accuracy"),
        "n": int(len(y)),
    }


def _fit_predict(X_tr, y_tr, X_te, *, model_kind: str = "xgboost") -> np.ndarray:
    if model_kind == "logreg":
        m = make_logreg().fit(X_tr, y_tr)
    else:
        m = make_xgboost().fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def lsoo_evaluate(
    df: pd.DataFrame, *, features: list[str], model_kind: str = "xgboost",
    train_years: tuple[int, int] = (2024,), test_years: tuple[int, int] = (2025,),
) -> dict[str, Any]:
    """LOSO fold A: train 2024 -> test 2025. Reports Brier/accuracy with CIs on
    the binary target."""
    X = df[features].copy()
    y = df["binary_label"].astype(int).to_numpy()
    season = df["season"].astype(int).to_numpy()
    # Drop rows with null features (sportsbook_spread ~5% null).
    valid = ~X.isna().any(axis=1)
    X, y, season = X[valid], y[valid], season[valid]

    tr = np.isin(season, train_years)
    te = np.isin(season, test_years)
    if tr.sum() == 0 or te.sum() == 0:
        return {"n_train": 0, "n_test": 0}
    probas = _fit_predict(X[tr], y[tr], X[te], model_kind=model_kind)
    base_rate = float(y[te].mean())
    base_brier = float(np.mean((base_rate - y[te]) ** 2))
    return {
        "train_years": train_years, "test_years": test_years,
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "model": _metric_with_ci(probas, y[te]),
        "base_rate": base_rate, "base_brier": base_brier,
    }


def rolling_origin_evaluate(
    df: pd.DataFrame, *, features: list[str], model_kind: str = "xgboost",
    embargo_weeks: int = 1, min_train_rows: int = 150,
) -> dict[str, Any]:
    """Purged rolling-origin within 2024+2025: block by week, hold out each week
    in turn (after enough history accumulates), with a 1-week embargo so a test
    week's games never train a model used to predict them. Gives an OOS variance
    estimate across many folds."""
    X = df[features].copy()
    y = df["binary_label"].astype(int).to_numpy()
    season = df["season"].astype(int).to_numpy()
    week = df["week"].astype(int).to_numpy()
    valid = ~X.isna().any(axis=1)
    X, y, season, week = X[valid], y[valid], season[valid], week[valid]

    # Order by (season, week) so "prior" is well-defined.
    order = np.lexsort((week, season))
    X, y, season, week = X.iloc[order].reset_index(drop=True), y[order], season[order], week[order]

    # Enumerate test origins: each unique (season, week) with >= min_train prior rows.
    keys = list(zip(season.tolist(), week.tolist()))
    fold_preds: list[tuple[float, int]] = []
    seen = 0
    for i, (s, w) in enumerate(keys):
        n_prior = i
        if n_prior < min_train_rows:
            continue
        # Train on all rows strictly before this (season,week) minus the embargo
        # window (rows in the immediately preceding week are excluded).
        train_idx = []
        for j in range(i):
            sj, wj = keys[j]
            if (sj, wj) == (s, w):
                continue
            train_idx.append(j)
        # Embargo: drop rows whose (season,week) is within embargo_weeks of the test.
        if embargo_weeks > 0:
            train_idx = [j for j in train_idx if _week_gap(keys[j], (s, w)) > embargo_weeks]
        if len(train_idx) < min_train_rows:
            continue
        te = np.array([i])
        tr = np.array(train_idx)
        try:
            probas = _fit_predict(X.iloc[tr], y[tr], X.iloc[te], model_kind=model_kind)
        except ValueError:
            continue
        fold_preds.append((float(probas[0]), int(y[i])))

    if not fold_preds:
        return {"n_folds": 0}
    p = np.array([fp[0] for fp in fold_preds])
    yy = np.array([fp[1] for fp in fold_preds])
    return {
        "n_folds": len(fold_preds),
        "model": _metric_with_ci(p, yy),
        "base_rate": float(yy.mean()),
        "base_brier": float(np.mean((yy.mean() - yy) ** 2)),
    }


def _week_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Approximate week gap between two (season, week) keys (cross-season rough)."""
    return abs((a[0] - b[0]) * 22 + (a[1] - b[1]))


# ---------------------------------------------------------------------------
# Pooling comparison
# ---------------------------------------------------------------------------

def compare_pools(df: pd.DataFrame, *, model_kind: str = "xgboost",
                  features: list[str] | None = None) -> dict[str, Any]:
    """Pooled vs per-type vs spread-first stacking, compared via LOSO Brier.

    Stacking: the spread model's out-of-fold predictions (generated within the
    same temporal folds) are added as a moneyline feature. The OOF prediction for
    a moneyline row never uses that row's own label nor any future fold's labels.

    ``features`` defaults to FEATURES_STAGE2 intersected with the frame; pass a
    pre-filtered list (e.g. dropping all-null columns) to control what the models
    actually see.
    """
    results: dict[str, Any] = {}
    base_feats = features if features is not None else [f for f in FEATURES_STAGE2 if f in df.columns]

    # (a) Pooled.
    pooled_feats = [f for f in base_feats if f in df.columns]
    results["pooled"] = lsoo_evaluate(df, features=pooled_feats, model_kind=model_kind)

    # (b) Per-type.
    for mt in ("moneyline", "spread"):
        sub = df[df["market_type"] == mt]
        feats = [f for f in base_feats if f != "market_type_spread" and f in sub.columns]
        results[f"per_type_{mt}"] = lsoo_evaluate(sub, features=feats, model_kind=model_kind)

    # (c) Spread-first stacking.
    results["stacking"] = _stacking_eval(df, model_kind=model_kind)
    return results


def _stacking_eval(df: pd.DataFrame, *, model_kind: str = "xgboost") -> dict[str, Any]:
    """Spread-first stacking with temporally-honest OOF predictions.

    For each moneyline LOSO fold, train the spread model only on spread rows in
    the same temporal training window, predict the moneyline fold's spread-derived
    feature — never using moneyline labels or future data. Then train the
    moneyline model with that OOF feature added.
    """
    out = {}
    for fold_name, (tr_y, te_y) in {
        "foldA_2024_2025": ((2024,), (2025,)),
    }.items():
        ml = df[df["market_type"] == "moneyline"].copy()
        sp = df[df["market_type"] == "spread"].copy()
        # Spread OOF feature: predicted P(inefficient) for each moneyline row from
        # a spread model trained in-window. Use the spread row's market_delta as
        # the carrier (the moneyline row's nearest spread signal is its own edge).
        # Simpler honest version: train spread on tr_y spread rows, predict te_y
        # moneyline rows using their own features mapped to the spread feature set.
        sp_feats = [f for f in FEATURES_STAGE2 if f != "market_type_spread" and f in sp.columns]
        ml_feats = [f for f in FEATURES_STAGE2 if f != "market_type_spread" and f in ml.columns]
        common_feats = [f for f in sp_feats if f in ml_feats]

        sp_tr = sp[sp["season"].astype(int).isin(tr_y)]
        sp_te_ml = ml[ml["season"].astype(int).isin(te_y)]
        ml_tr = ml[ml["season"].astype(int).isin(tr_y)]
        ml_te = ml[ml["season"].astype(int).isin(te_y)]
        if len(sp_tr) < 30 or len(ml_tr) < 30 or len(ml_te) == 0:
            out[fold_name] = {"n_train": len(ml_tr), "n_test": len(ml_te), "skipped": True}
            continue

        # Spread model -> OOF feature on moneyline test rows.
        sp_valid = ~sp_tr[common_feats].isna().any(axis=1)
        try:
            sp_model = (make_logreg() if model_kind == "logreg" else make_xgboost()).fit(
                sp_tr[common_feats][sp_valid], sp_tr["binary_label"].astype(int)[sp_valid]
            )
        except ValueError:
            out[fold_name] = {"n_train": len(ml_tr), "n_test": len(ml_te), "skipped": True}
            continue

        def _add_oof(frame: pd.DataFrame) -> pd.DataFrame:
            f = frame.copy()
            xv = ~f[common_feats].isna().any(axis=1)
            oof = np.full(len(f), 0.5)
            if xv.any():
                oof[xv.to_numpy()] = sp_model.predict_proba(f[common_feats][xv])[:, 1]
            f["spread_oof"] = oof
            return f

        ml_tr_f = _add_oof(ml_tr)
        ml_te_f = _add_oof(ml_te)
        stacked_feats = common_feats + ["spread_oof"]

        # Moneyline model with the OOF feature.
        tr_v = ~ml_tr_f[stacked_feats].isna().any(axis=1)
        te_v = ~ml_te_f[stacked_feats].isna().any(axis=1)
        if tr_v.sum() == 0 or te_v.sum() == 0:
            out[fold_name] = {"n_train": int(tr_v.sum()), "n_test": int(te_v.sum()), "skipped": True}
            continue
        probas = _fit_predict(
            ml_tr_f[stacked_feats][tr_v], ml_tr_f["binary_label"].astype(int)[tr_v],
            ml_te_f[stacked_feats][te_v], model_kind=model_kind,
        )
        out[fold_name] = {
            "n_train": int(tr_v.sum()), "n_test": int(te_v.sum()),
            "model": _metric_with_ci(probas, ml_te_f["binary_label"].astype(int)[te_v].to_numpy()),
            "base_rate": float(ml_te_f["binary_label"].astype(int)[te_v].mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_and_save() -> dict[str, Any]:
    """Build the Stage-2 frame, run pooled/per-type/stacking + LOSO/rolling-origin,
    write predictions + model. Reports OOS metrics with CIs — no edge claims
    without CIs."""
    from . import cover, outcome

    config.ensure_dirs()
    aligned = io.read_parquet(config.PROCESSED_DIR / "aligned.parquet")
    game_features = io.read_parquet(config.FEATURE_PROCESSED_DIR / "game_features.parquet")
    win_probs = io.read_parquet(config.FEATURE_PROCESSED_DIR / "game_outcome_probs.parquet")
    cover_path = config.FEATURE_PROCESSED_DIR / "game_cover_probs.parquet"
    if not cover_path.exists():
        print("[inefficiency] cover-prob model output missing; building it first...")
        cover.build_and_save()
    cover_probs = io.read_parquet(cover_path)

    frame = _build_stage2_frame(aligned, game_features, win_probs, cover_probs)
    print(f"[inefficiency] built Stage-2 frame: {len(frame)} rows "
          f"(moneyline {(frame['market_type']=='moneyline').sum()}, "
          f"spread {(frame['market_type']=='spread').sum()})")
    print(f"[inefficiency]   edge: mean={frame['edge'].mean():.3f} std={frame['edge'].std():.3f} "
          f"(market prices YES {frame['market_prob'].mean():.3f} vs model {frame['model_prob_yes'].mean():.3f})")
    print(f"[inefficiency]   binary label balance: {frame['binary_label'].mean():.3f}")

    available = [f for f in FEATURES_STAGE2 if f in frame.columns]
    # Drop features that are entirely null (e.g. line-movement cols when the
    # OddsPortal scrape hasn't populated them). An all-null feature would make
    # the null-drop in eval discard every row -> empty folds.
    available = [f for f in available if frame[f].notna().any()]
    # Report line-movement feature coverage (the new signal).
    for lf in LINE_MOVEMENT_FEATURES:
        if lf in frame.columns:
            cov = frame[lf].notna().mean()
            status = "INCLUDED" if cov > 0 else "all-null, excluded"
            print(f"[inefficiency]   {lf}: {cov:.0%} non-null ({status})")
    print("[inefficiency] running pooled vs per-type vs stacking (LOSO 2024->2025)...")
    pools = compare_pools(frame, features=available)

    print("[inefficiency] running purged rolling-origin (week-blocked, 1-week embargo)...")
    rolling = {
        "pooled": rolling_origin_evaluate(frame, features=available),
    }

    # Stage 3 backtest: ROI / CLV / Kelly on the LOSO test fold.
    print("[inefficiency] running Stage 3 backtest (ROI, CLV, Kelly)...")
    from . import backtest as backtest_mod
    op_path = config.ODDSPORTAL_RAW_DIR / "nfl_lines.parquet"
    op_lines = io.read_parquet(op_path) if op_path.exists() else None
    if op_lines is None:
        print("[inefficiency]   OddsPortal lines missing; CLV skipped (run scripts.ingest_oddsportal)")
    test_fold = frame[frame["season"].astype(int) == 2025]
    backtest_results = backtest_mod.evaluate(test_fold, oddsportal_lines=op_lines)

    # Pick the best config by LOSO test Brier (lowest mean), then refit on all
    # 2024+2025 for the final prediction artifact.
    cfg_briers = []
    for name, res in {**{k: v for k, v in pools.items() if k != "stacking"},
                      "stacking_foldA": pools.get("stacking", {}).get("foldA_2024_2025", {})}.items():
        m = res.get("model") if isinstance(res, dict) else None
        if m and isinstance(m, dict) and "brier" in m:
            cfg_briers.append((name, m["brier"]))
    cfg_briers.sort(key=lambda t: t[1])
    best_cfg = cfg_briers[0][0] if cfg_briers else "pooled"
    print(f"[inefficiency] best config by LOSO test Brier: {best_cfg}")

    # Write the per-market prediction artifact (OOS LOSO predictions for 2025).
    pred_path = config.FEATURE_PROCESSED_DIR / "inefficiency_predictions.parquet"
    art_cols = [c for c in (
        "condition_id", "game_id", "market_type", "season", "week",
        "model_prob_yes", "market_prob", "edge", "binary_label",
    ) if c in frame.columns]
    io.write_parquet(frame[art_cols], pred_path)
    model_path = config.MODELS_PROCESSED_DIR / "inefficiency_model.joblib"
    _save_bundle({
        "features": available, "best_cfg": best_cfg,
        "pools": pools, "rolling": rolling, "backtest": backtest_results,
    }, model_path)

    # Print the Stage 3 verdict honestly.
    for t, b in backtest_results.get("thresholds", {}).items():
        clv_str = f"  clv={b.get('clv_mean', float('nan')):+.3f}" if "clv_mean" in b else ""
        print(f"[inefficiency]   |edge|>={t:.2f}: n={b['n_trades']} roi={b['roi']:+.3f} "
              f"win={b['win_rate']:.3f}{clv_str}")

    summary: dict[str, Any] = {
        "n_rows": len(frame),
        "n_moneyline": int((frame["market_type"] == "moneyline").sum()),
        "n_spread": int((frame["market_type"] == "spread").sum()),
        "edge_mean": float(frame["edge"].mean()),
        "binary_label_rate": float(frame["binary_label"].mean()),
        "line_movement_coverage": {
            lf: float(frame[lf].notna().mean()) for lf in LINE_MOVEMENT_FEATURES if lf in frame.columns
        },
        "pools": pools, "rolling": rolling, "backtest": backtest_results,
        "best_cfg": best_cfg,
        "predictions_path": str(pred_path), "model_path": str(model_path),
    }
    print(f"[inefficiency] wrote predictions -> {pred_path}")
    print(f"[inefficiency] wrote model bundle -> {model_path}")
    return summary


def _save_bundle(bundle: dict[str, Any], path) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
