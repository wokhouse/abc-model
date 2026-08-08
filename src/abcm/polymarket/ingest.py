"""Orchestrate the full Polymarket ingestion pass.

Pipeline:
    resolve NFL tag  →  page all closed events  →  flatten to market rows
    →  persist events.parquet + markets.parquet  →  fetch per-token price history

Run via ``scripts.ingest_polymarket``. Each stage prints a small summary so the
operator can eyeball coverage (e.g. how many markets are moneylines).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .. import config, io
from . import events, prices, tags
from .client import PolymarketClient


def ingest_polymarket(
    force_prices: bool = False,
    market_types: list[str] | None = None,
) -> dict[str, Any]:
    """Run the full Polymarket ingestion. Returns a summary dict.

    ``market_types`` scopes the (slow) trade-history fetch to those types only.
    Defaults to moneyline — the modeling target. Event/market tables are always
    written in full regardless, so other types can be backfilled later without
    re-paging events.
    """
    if market_types is None:
        market_types = ["moneyline"]
    config.ensure_dirs()
    summary: dict[str, Any] = {"market_types_fetched": market_types}

    with PolymarketClient() as client:
        # 1. Resolve the NFL tag and cache it.
        tag_payload = tags.resolve_and_cache(client)
        nfl_tag = tag_payload["nfl_tag"]
        tag_id = str(nfl_tag["id"])
        summary["nfl_tag"] = nfl_tag
        print(f"[tags] NFL tag id = {tag_id} ({nfl_tag.get('label')})")

        # 2. Page all closed NFL events.
        all_events = events.fetch_all_closed_events(client, tag_id)
        summary["n_events"] = len(all_events)
        print(f"[events] {len(all_events)} closed NFL events")

        # 3. Flatten to event + market frames and persist.
        events_df = events.events_to_frame(all_events)
        markets_df = events.flatten_markets(all_events)
        io.write_parquet(events_df, config.POLYMARKET_RAW_DIR / "events.parquet")
        io.write_parquet(markets_df, config.POLYMARKET_RAW_DIR / "markets.parquet")
        print(f"[markets] {len(markets_df)} markets persisted")

        if not markets_df.empty:
            type_counts = markets_df["market_type"].value_counts().to_dict()
            summary["market_type_counts"] = type_counts
            print(f"[markets] by type: {type_counts}")

        # 4. Fetch trade history for the requested market types (keyed by
        # conditionId). The data-api serves resolved markets, unlike the CLOB
        # /prices-history. Scoping to moneyline avoids paging thousands of trades
        # on long-running futures markets that aren't modeling targets.
        price_subset = markets_df[markets_df["market_type"].isin(market_types)]
        n_markets_for_prices = len(price_subset.dropna(subset=["condition_id"]))
        summary["n_markets_for_prices"] = n_markets_for_prices
        print(f"[prices] fetching trade history for {n_markets_for_prices} "
              f"{market_types} markets...")
        price_counts = prices.ingest_markets(client, price_subset, force=force_prices)
        summary["price_counts"] = price_counts
        print(f"[prices] {price_counts}")

    return summary
