"""Snapshot a per-market pre-game price from the hourly candle files.

For each market we want one "pre-game" price: the last hourly close at or before
``PRICE_SNAPSHOT_HOURS_BEFORE_GAME`` hours before kickoff. That price is the
market's implied probability for the Yes token — the modeling target and the
baseline for closing-line-value comparisons.

We also materialize the market's *opening* price — the first actual Yes-token
trade (tick-level, read from the trade history) — and the open→snapshot delta, a
market feature: a large swing between the market open (where it's least
efficient) and the pre-game close signals information flow or sharp money into
one side. Falls back to the first hourly candle's open when the token id or
trade history is unavailable.

Kickoff time is reconstructed from nflverse ``gameday`` (a date) + ``gametime``
(an ``HH:MM`` Eastern string), converted to UTC. Using the date alone (as an
earlier version did) placed the cutoff ~17-23h before actual kickoff because the
date parses to midnight; the true kickoff hour matters for a T-1h cutoff.

Candles are sourced from the trade-history aggregation in ``polymarket.prices``
(one ``candles.parquet`` per ``condition_id``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .. import config
from ..polymarket import prices as pm_prices

# nflverse ``gametime`` is US Eastern (e.g. "20:20"). Kickoff in UTC needs this.
_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


def snapshot_price(
    condition_id: str | None,
    game_time: pd.Timestamp | None,
    *,
    yes_token_id: str | None = None,
    hours_before: float = config.PRICE_SNAPSHOT_HOURS_BEFORE_GAME,
) -> dict[str, Any]:
    """Return the pre-game snapshot (Yes-token close) for one market.

    Result keys: ``has_history``, ``snapshot_price``, ``snapshot_time``,
    ``method`` ("cutoff" | "last_fallback" | "no_game_time" | "no_history" |
    "no_condition_id"), plus:

    * ``open_price`` / ``open_time`` — the market's true opening price. When
      ``yes_token_id`` is given, this is the *first actual trade* (tick-level)
      from the trade history, which is the moment the market is least efficient.
      Falls back to the first hourly candle's open when trades are unavailable.
    * ``price_delta`` — snapshot close minus open price.
    * ``n_candles`` — hourly-candle count (coverage diagnostic).
    """
    result: dict[str, Any] = {
        "has_history": False,
        "snapshot_price": None,
        "snapshot_time": None,
        "method": "no_history",
        "open_price": None,
        "open_time": None,
        "price_delta": None,
        "n_candles": 0,
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
    result["n_candles"] = int(len(candles))

    # --- Opening price: first actual Yes-token trade (tick-level) -------------
    # The market open is where it's least efficient; the first trade is a more
    # precise signal than the first hourly candle's open. We read the raw trade
    # history and take the earliest Yes-token trade, falling back to the candle
    # open if trades are missing or the token id is unknown.
    open_price: float | None = None
    open_time: pd.Timestamp | None = None
    first_trade = _first_yes_trade(condition_id, yes_token_id)
    if first_trade is not None:
        open_price, open_time = first_trade
    if open_price is None:
        first = candles.iloc[0]
        open_price = float(first["open"]) if pd.notna(first["open"]) else None
        open_time = first["ts_naive"]
    result["open_price"] = open_price
    result["open_time"] = open_time

    if game_time is None or pd.isna(game_time):
        # No game time: fall back to the last available close.
        last = candles.iloc[-1]
        close = float(last["close"]) if pd.notna(last["close"]) else None
        result["snapshot_price"] = close
        result["snapshot_time"] = last["ts_naive"]
        result["method"] = "no_game_time"
        result["price_delta"] = _delta(close, open_price)
        return result

    cutoff = game_time - pd.Timedelta(hours=hours_before)
    at_or_before = candles[candles["ts_naive"] <= cutoff]
    if not at_or_before.empty:
        last = at_or_before.iloc[-1]
        close = float(last["close"]) if pd.notna(last["close"]) else None
        result["snapshot_price"] = close
        result["snapshot_time"] = last["ts_naive"]
        result["method"] = "cutoff"
    else:
        # Price history starts after our cutoff — take the earliest candle.
        close = open_price
        result["snapshot_price"] = close
        result["snapshot_time"] = open_time
        result["method"] = "last_fallback"
    result["price_delta"] = _delta(close, open_price)
    return result


def _first_yes_trade(
    condition_id: str | None, yes_token_id: str | None
) -> tuple[float, pd.Timestamp] | None:
    """Return (price, time) of the earliest Yes-token trade, or None.

    Reads the raw trade history for the market and filters to the Yes token.
    The timestamp is normalized to tz-naive UTC to match the candle timeline.
    """
    if not condition_id or not yes_token_id:
        return None
    trades = pm_prices.trades_for(condition_id)
    if trades.empty:
        return None
    yes = trades[trades["asset"].astype(str) == str(yes_token_id)]
    yes = yes.dropna(subset=["timestamp", "price"]).sort_values("timestamp")
    if yes.empty:
        return None
    first = yes.iloc[0]
    ts = pd.to_datetime(first["timestamp"], unit="s", utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return float(first["price"]), ts.tz_localize(None)


def _delta(close: float | None, open_price: float | None) -> float | None:
    """Snapshot close minus opening price; None if either is missing.

    A single-candle market yields delta 0 (open == close); downstream code can
    treat ``n_candles < 2`` as low-confidence if desired.
    """
    if close is None or open_price is None:
        return None
    return close - open_price


def kickoff_utc(gameday: Any, gametime: Any) -> pd.Timestamp | None:
    """Reconstruct a game's kickoff as a tz-naive UTC timestamp.

    ``gameday`` is a date string; ``gametime`` is an ``HH:MM`` US Eastern string
    (nflverse convention). Returns None if either is missing or unparseable.
    The result is tz-naive UTC to match the candle timeline.
    """
    if gameday is None or gametime is None:
        return None
    gt = str(gametime).strip()
    if ":" not in gt:
        return None
    try:
        parts = gt.split(":")
        local = datetime(int(str(gameday)[:4]), int(str(gameday)[5:7]),
                         int(str(gameday)[8:10]), int(parts[0]), int(parts[1]), tzinfo=_ET)
    except (ValueError, IndexError):
        return None
    return pd.Timestamp(local.astimezone(_UTC)).tz_localize(None)


def add_price_snapshots(
    markets: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """Attach a Yes-token snapshot price + method to each market row.

    ``game_time`` is the game's true kickoff (gameday + gametime, converted from
    Eastern to UTC). Falls back to the market's own end timestamp when no game
    matched or kickoff can't be parsed.
    """
    # Map game_id -> true kickoff UTC for the T-1h cutoff.
    kickoff_by_game: dict[str, pd.Timestamp] = {}
    if not schedules.empty:
        for _, r in schedules.iterrows():
            ko = kickoff_utc(r.get("gameday"), r.get("gametime"))
            if ko is not None and pd.notna(ko):
                kickoff_by_game[str(r.get("game_id"))] = ko

    snap_prices: list[Any] = []
    snap_times: list[Any] = []
    methods: list[str] = []
    has_history: list[bool] = []
    open_prices: list[Any] = []
    open_times: list[Any] = []
    deltas: list[Any] = []
    n_candles: list[int] = []

    for _, row in markets.iterrows():
        game_id = row.get("game_id")
        game_time = kickoff_by_game.get(str(game_id)) if game_id is not None else None
        # Fall back to the market's own end timestamp if no game matched or
        # kickoff unparseable.
        if game_time is None:
            mt = pd.to_datetime(row.get("market_end"), utc=True, errors="coerce")
            game_time = mt.tz_localize(None) if pd.notna(mt) else None

        snap = snapshot_price(
            row.get("condition_id"),
            game_time,
            yes_token_id=row.get("yes_token_id"),
        )
        snap_prices.append(snap["snapshot_price"])
        snap_times.append(snap["snapshot_time"])
        methods.append(snap["method"])
        has_history.append(snap["has_history"])
        open_prices.append(snap["open_price"])
        open_times.append(snap["open_time"])
        deltas.append(snap["price_delta"])
        n_candles.append(snap["n_candles"])

    out = markets.copy()
    out["snapshot_price"] = snap_prices
    out["snapshot_time"] = snap_times
    out["snapshot_method"] = methods
    out["has_price_history"] = has_history
    out["open_price"] = open_prices
    out["open_time"] = open_times
    out["price_delta"] = deltas
    out["n_candles"] = n_candles
    return out
