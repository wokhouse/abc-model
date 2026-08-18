"""Fetch trade history for resolved Polymarket markets and build price series.

Polymarket's CLOB ``/prices-history`` endpoint returns **empty** for resolved
(closed) markets — it only serves live markets. To reconstruct pre-game prices we
use the undocumented-but-official ``data-api.polymarket.com/trades`` endpoint,
which returns the full trade history for a market (keyed by ``conditionId``),
including after it resolves.

Each trade carries a ``price`` (0–1) and an ``asset`` (token id), so we filter to
the market's Yes token to build a Yes-price series, then aggregate to hourly
candles. We persist both the raw trades (provenance) and the candles (analysis).

Note: ``data-api`` returns trades for BOTH outcomes under one ``conditionId``;
``asset`` filtering is client-side only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .. import config, io
from .client import PolymarketClient


# --- File layout -----------------------------------------------------------
# data/raw/polymarket/prices/<condition_id>/trades.parquet   (raw, all outcomes)
# data/raw/polymarket/prices/<condition_id>/candles.parquet  (hourly, Yes token)


def market_price_dir(condition_id: str) -> Path:
    return config.POLYMARKET_PRICES_DIR / condition_id


def trades_path(condition_id: str) -> Path:
    return market_price_dir(condition_id) / "trades.parquet"


def candles_path(condition_id: str) -> Path:
    return market_price_dir(condition_id) / "candles.parquet"


# --- Fetching --------------------------------------------------------------

def fetch_trades(
    client: PolymarketClient,
    condition_id: str,
    *,
    taker_only: bool = True,
) -> pd.DataFrame:
    """Page through every trade for one market (conditionId).

    Returns a DataFrame with columns: ``timestamp`` (epoch s), ``price``,
    ``asset`` (token id), ``side``, ``size``, ``outcome``, ``outcomeIndex``.
    Empty DataFrame if the market has no trades.
    """
    url = f"{config.DATA_API_BASE_URL}/trades"
    rows: list[dict[str, Any]] = []
    offset = 0
    page = config.TRADES_PAGE_SIZE

    while True:
        batch = client.get_json(
            url,
            params={
                "market": condition_id,
                "limit": page,
                "offset": offset,
                "takerOnly": "true" if taker_only else "false",
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "price", "asset", "side", "size", "outcome", "outcomeIndex"]
        )

    df = pd.DataFrame(rows)
    # Keep the columns downstream needs; tolerate missing fields.
    keep = ["timestamp", "price", "asset", "side", "size", "outcome", "outcomeIndex"]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[keep].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["outcomeIndex"] = pd.to_numeric(df["outcomeIndex"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"]).sort_values("timestamp").reset_index(drop=True)
    return df


def aggregate_candles(trades: pd.DataFrame, yes_token_id: str | None) -> pd.DataFrame:
    """Build hourly OHLC candles for the Yes token from raw trades.

    Returns columns: ``ts`` (hour, timestamp), ``open``, ``high``, ``low``,
    ``close``, ``n_trades``, ``volume``. Candles only exist for hours with
    trades; callers forward-fill as needed.
    """
    empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "n_trades", "volume"])
    if trades.empty or not yes_token_id:
        return empty

    yes = trades[trades["asset"].astype(str) == str(yes_token_id)].copy()
    if yes.empty:
        return empty

    yes["ts"] = pd.to_datetime(yes["timestamp"], unit="s", utc=True).dt.floor("h")
    grouped = yes.groupby("ts")
    candles = pd.DataFrame({
        "open": grouped["price"].first(),
        "high": grouped["price"].max(),
        "low": grouped["price"].min(),
        "close": grouped["price"].last(),
        "n_trades": grouped["price"].size(),
        "volume": grouped["size"].sum(),
    }).reset_index()
    return candles


# --- Persistence -----------------------------------------------------------

def fetch_and_save(
    client: PolymarketClient,
    condition_id: str,
    yes_token_id: str | None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Fetch trades + candles for one market and persist them.

    Skips if already on disk (unless ``force=True``). Empty histories are still
    written so we don't re-query. Returns a small status dict.
    """
    t_path = trades_path(condition_id)
    c_path = candles_path(condition_id)
    if t_path.exists() and not force:
        return {"status": "skipped", "condition_id": condition_id}

    trades = fetch_trades(client, condition_id)
    candles = aggregate_candles(trades, yes_token_id)
    io.write_parquet(trades, t_path)
    io.write_parquet(candles, c_path)
    return {
        "status": "empty" if trades.empty else "fetched",
        "condition_id": condition_id,
        "n_trades": len(trades),
        "n_candles": len(candles),
    }


def ingest_markets(
    client: PolymarketClient,
    markets: pd.DataFrame,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Fetch trade history for every market in the frame.

    Iterates one row per market (de-duplicated by conditionId). Returns counts.
    """
    counts = {"fetched": 0, "skipped": 0, "empty": 0, "errors": 0}
    seen: set[str] = set()

    work = markets.dropna(subset=["condition_id"]).drop_duplicates("condition_id")
    for _, row in tqdm(work.iterrows(), total=len(work), desc="prices", unit="mkt"):
        cid = str(row["condition_id"])
        if cid in seen:
            continue
        seen.add(cid)
        if not cid or cid == "nan":
            continue
        try:
            result = fetch_and_save(client, cid, row.get("yes_token_id"), force=force)
        except Exception as exc:  # keep going; surface count at the end
            counts["errors"] += 1
            tqdm.write(f"[prices] error on {cid}: {exc}")
            continue
        if result["status"] == "skipped":
            counts["skipped"] += 1
        elif result["status"] == "empty":
            counts["empty"] += 1
        else:
            counts["fetched"] += 1
    return counts


# --- Snapshot helper (used by reconcile) -----------------------------------

def candles_for(condition_id: str) -> pd.DataFrame:
    """Load the hourly Yes-token candles for a market (empty if absent)."""
    path = candles_path(condition_id)
    if not path.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "n_trades", "volume"])
    return io.read_parquet(path)


def trades_for(condition_id: str) -> pd.DataFrame:
    """Load the raw trade history for a market (all outcomes; empty if absent).

    Unlike ``candles_for`` (which is Yes-token only, hourly), this returns every
    trade for both outcomes at tick resolution. Filter on ``asset`` to isolate
    the Yes token. Used to recover the true tick-level opening price.
    """
    path = trades_path(condition_id)
    if not path.exists():
        return pd.DataFrame(
            columns=["timestamp", "price", "asset", "side", "size", "outcome", "outcomeIndex"]
        )
    return io.read_parquet(path)
