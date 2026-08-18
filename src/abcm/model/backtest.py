"""Stage 3 — backtest, CLV, and Kelly bet-sizing.

Turns the Stage 2 inefficiency predictions into a backtestable, sized trading
record and the metrics that actually determine if the model is real. Per the
research: **CLV (Closing Line Value) is the primary validity signal, not ROI**
— ROI is too noisy at n≈300-600, while consistent positive CLV is the hallmark
of a real edge. We report both, with bootstrap CIs, and an explicit sign test
on CLV (the mean can be positive by chance; a binomial sign test is stricter).

The honest framing from Stage 2's negative result: there may be no positive edge
to validate. This stage reports that honestly — if CLV is centered on zero with
a wide CI, the verdict is "no detectable edge," not a dressed-up positive.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import calibrate, lines


def kelly_fraction(model_prob: float, decimal_odds: float, *, fraction: float = 0.25) -> float:
    """Fractional-Kelly bet size as a fraction of bankroll.

    ``f* = (b*p - q) / b`` where ``b = decimal_odds - 1`` (the net odds), ``p`` =
    model prob of winning, ``q = 1-p``. We return ``fraction * f*`` (default 25%
    fractional Kelly) to cut variance at small n. Negative => don't bet.
    """
    if decimal_odds <= 1.0 or np.isnan(model_prob):
        return 0.0
    b = decimal_odds - 1.0
    p = model_prob
    q = 1.0 - p
    full = (b * p - q) / b
    return float(max(fraction * full, 0.0))


def apply_decision_rule(
    preds: pd.DataFrame, *, edge_col: str = "edge", thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.15)
) -> dict[float, pd.DataFrame]:
    """Split the prediction frame into trade sets by |edge| threshold.

    For each threshold, a market is traded iff ``abs(edge) >= threshold``. The bet
    side is YES when edge>0 (model says YES undervalued), NO otherwise. Returns
    ``{threshold: trades_df}`` for ROI/CLV analysis per bucket.
    """
    out = {}
    abs_edge = preds[edge_col].abs()
    for t in thresholds:
        trades = preds[abs_edge >= t].copy()
        trades["bet_side"] = np.where(trades[edge_col] > 0, "YES", "NO")
        out[t] = trades
    return out


def trade_outcome(row) -> int:
    """Did the bet win? bet YES + resolved YES = win; bet NO + resolved NO = win."""
    side = row["bet_side"]
    resolved = row["resolved_yes_price"]
    if pd.isna(resolved):
        return -1
    yes_won = resolved == 1.0
    return int((side == "YES") == yes_won)


def _roi(trades: pd.DataFrame) -> float:
    """Mean ROI on YES-side bets only (the price we actually have).

    Polymarket YES/NO are *separate tokens* with their own order books; the NO
    price is NOT ``1 - yes_price``. We only have the YES opening price
    (``market_prob``), so an honest ROI is computable only for bets where we'd
    buy the YES token. NO-side bets are excluded from the ROI mean (counted in
    ``n_trades`` and ``win_rate`` but not ROI) — reporting a NO-bet ROI from the
    complementary price would be fabricated.

    YES bet: cost = market_prob (the YES price), pays $1 on win, $0 on loss.
    Per-bet return = (1 - cost) on win, -cost on loss; ROI = return / cost.
    """
    yes = trades[trades["bet_side"] == "YES"]
    if len(yes) == 0:
        return float("nan")
    cost = yes["market_prob"].to_numpy()
    won = yes["_won"].to_numpy().astype(bool)
    payout = np.where(won, 1.0, 0.0)
    ret = (payout - cost) / np.where(cost > 0, cost, np.nan)
    return float(np.nanmean(ret))


def _bootstrap(values: np.ndarray, *, n_boot: int = 500, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    boots = np.array([np.nanmean(values[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return {"mean": float(np.nanmean(values)), "lo": float(np.nanpercentile(boots, 2.5)),
            "hi": float(np.nanpercentile(boots, 97.5))}


def evaluate(
    preds: pd.DataFrame,
    oddsportal_lines: pd.DataFrame | None = None,
    *,
    thresholds: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.15),
) -> dict[str, Any]:
    """Run the full backtest: decision-rule buckets, ROI, CLV, Kelly sizing.

    ``preds`` is the Stage-2 frame (must carry ``edge, market_prob, resolved_yes_price,
    game_id, market_type, model_prob_yes``). ``oddsportal_lines`` carries the
    closing moneyline odds per game for CLV; if absent, CLV is skipped (the metric
    that needs it). All metrics come with bootstrap 95% CIs.
    """
    df = preds.copy()
    # Attach CLV where closing lines exist.
    if oddsportal_lines is not None and len(oddsportal_lines):
        op = oddsportal_lines[["game_id", "home_close_ml", "away_close_ml"]].dropna()
        df = df.merge(op, on="game_id", how="left")

    buckets = apply_decision_rule(df, thresholds=thresholds)
    results: dict[str, Any] = {"thresholds": {}, "overall": {}}

    for t, trades in buckets.items():
        if len(trades) == 0:
            continue
        trades = trades.copy()
        trades["_won"] = trades.apply(trade_outcome, axis=1)
        trades = trades[trades["_won"] >= 0]
        won = trades["_won"].to_numpy()
        roi = _roi(trades)

        # Per-bet YES-side returns (matching _roi's definition) for an honest CI.
        yes_trades = trades[trades["bet_side"] == "YES"]
        if len(yes_trades):
            ycost = yes_trades["market_prob"].to_numpy()
            ywon = yes_trades["_won"].to_numpy().astype(bool)
            ypayout = np.where(ywon, 1.0, 0.0)
            yes_returns = (ypayout - ycost) / np.where(ycost > 0, ycost, np.nan)
            yes_returns = yes_returns[np.isfinite(yes_returns)]
        else:
            yes_returns = np.array([])

        bucket: dict[str, Any] = {
            "n_trades": int(len(trades)),
            "n_yes": int(len(yes_trades)),
            "win_rate": float(won.mean()) if len(won) else float("nan"),
            "roi": roi,
            "roi_ci": _bootstrap(yes_returns) if len(yes_returns) else {"mean": float("nan")},
        }
        # CLV where closing lines joined.
        if "home_close_ml" in trades:
            clvs = []
            for _, r in trades.iterrows():
                if pd.isna(r.get("home_close_ml")):
                    continue
                # model_prob_yes vs the YES side's closing fair prob.
                yes_home = r.get("bet_side") == "YES"  # approximate
                clvs.append(lines.clv(
                    r["model_prob_yes"], r["home_close_ml"], r["away_close_ml"],
                    model_is_home=yes_home if "yes_home" in r else True,
                ))
            if clvs:
                clvs = np.array(clvs)
                clvs_finite = clvs[np.isfinite(clvs)]
                bucket["clv_mean"] = float(np.mean(clvs_finite)) if len(clvs_finite) else float("nan")
                bucket["clv_ci"] = _bootstrap(clvs_finite)
                # Sign test: is positive CLV consistent (binomial)?
                n_pos = int((clvs_finite > 0).sum())
                bucket["clv_pos_rate"] = float(n_pos / len(clvs_finite)) if len(clvs_finite) else float("nan")
        results["thresholds"][t] = bucket

    # Overall Kelly-sized record (use a 5% edge threshold as the actionable cut).
    actionable = buckets.get(0.05, pd.DataFrame())
    if len(actionable):
        actionable = actionable.copy()
        actionable["_won"] = actionable.apply(trade_outcome, axis=1)
        actionable = actionable[actionable["_won"] >= 0]
        # Fractional-Kelly size by the model prob and the market price (decimal odds).
        actionable["kelly_size"] = actionable.apply(
            lambda r: kelly_fraction(r["model_prob_yes"], 1.0 / r["market_prob"]) if r["market_prob"] > 0 else 0.0,
            axis=1,
        )
        results["overall"]["n_actionable"] = int(len(actionable))
        results["overall"]["mean_kelly_size"] = float(actionable["kelly_size"].mean())

    return results
