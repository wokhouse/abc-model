"""Snapshot a per-market pre-game price from the hourly candle files.

For each market we want one "pre-game" price: the last hourly close at or before
``PRICE_SNAPSHOT_HOURS_BEFORE_GAME`` hours before kickoff. That price is the
market's implied probability for the Yes token — the modeling target and the
baseline for closing-line-value comparisons.

Candles are sourced from the trade-history aggregation in ``polymarket.prices``
(one ``candles.parquet`` per ``condition_id``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .. import config
from ..polymarket import prices as pm_prices


def snapshot_price(
    condition_id: str | None,
    game_time: pd.Timestamp | None,
    *,
    hours_before: float = config.PRICE_SNAPSHOT_HOURS_BEFORE_GAME,
) -> dict[str, Any]:
    """Return the pre-game snapshot (Yes-token close) for one market.

    Result keys: ``has_history``, ``snapshot_price``, ``snapshot_time``,
    ``method`` ("cutoff" | "last_fallback" | "no_game_time" | "no_history" |
    "no_condition_id").
    """
    result: dict[str, Any] = {
        "has_history": False,
        "snapshot_price": None,
        "snapshot_time": None,
        "method": "no_history",
    }
    if not condition_id:
        result["method"] = "no_condition_id"
        return result

    candles = pm_prices.candles_for(condition_id)
    if candles.empty:
        return result
    result["has_history"] = True

    # Candles are tz-aware UTC; normalize to tz-naive for comparison.
    ts = pd.to_datetime(candles["ts"], utc=True, errors="coerce").dt.tz_localize(None)
    candles = candles.assign(ts_naive=ts).dropna(subset=["ts_naive"]).sort_values("ts_naive")

    if game_time is None or pd.isna(game_time):
        # No game time: fall back to the last available close.
        last = candles.iloc[-1]
        result["snapshot_price"] = float(last["close"]) if pd.notna(last["close"]) else None
        result["snapshot_time"] = last["ts_naive"]
        result["method"] = "no_game_time"
        return result

    cutoff = game_time - pd.Timedelta(hours=hours_before)
    at_or_before = candles[candles["ts_naive"] <= cutoff]
    if not at_or_before.empty:
        last = at_or_before.iloc[-1]
        result["snapshot_price"] = float(last["close"]) if pd.notna(last["close"]) else None
        result["snapshot_time"] = last["ts_naive"]
        result["method"] = "cutoff"
    else:
        # Price history starts after our cutoff — take the earliest candle.
        first = candles.iloc[0]
        result["snapshot_price"] = float(first["close"]) if pd.notna(first["close"]) else None
        result["snapshot_time"] = first["ts_naive"]
        result["method"] = "last_fallback"
    return result


def add_price_snapshots(
    markets: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Attach a Yes-token snapshot price + method to each market row.

    ``game_time`` is derived from the matched game's gameday where possible.
    """
    # Map game_id -> gameday for the cutoff.
    gameday_by_game: dict[str, pd.Timestamp] = {}
    if not schedules.empty:
        gd = schedules.copy()
        gd["gameday_dt"] = pd.to_datetime(gd["gameday"], errors="coerce")
        for _, r in gd.iterrows():
            if pd.notna(r.get("gameday_dt")):
                gameday_by_game[str(r.get("game_id"))] = r["gameday_dt"]

    snap_prices: list[Any] = []
    snap_times: list[Any] = []
    methods: list[str] = []
    has_history: list[bool] = []

    for _, row in markets.iterrows():
        game_id = row.get("game_id")
        game_time = gameday_by_game.get(str(game_id)) if game_id is not None else None
        # Fall back to the market's own end timestamp if no game matched.
        if game_time is None:
            mt = pd.to_datetime(row.get("market_end"), utc=True, errors="coerce")
            game_time = mt.tz_localize(None) if pd.notna(mt) else None

        snap = snapshot_price(row.get("condition_id"), game_time)
        snap_prices.append(snap["snapshot_price"])
        snap_times.append(snap["snapshot_time"])
        methods.append(snap["method"])
        has_history.append(snap["has_history"])

    out = markets.copy()
    out["snapshot_price"] = snap_prices
    out["snapshot_time"] = snap_times
    out["snapshot_method"] = methods
    out["has_price_history"] = has_history
    return out
