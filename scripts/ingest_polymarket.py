"""CLI entry point for Polymarket ingestion.

Usage:
    uv run python -m scripts.ingest_polymarket            # default years from config
    uv run python -m scripts.ingest_polymarket --force-prices   # re-fetch all price history
"""
from __future__ import annotations

import argparse

from abcm.polymarket import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest resolved NFL Polymarket markets.")
    parser.add_argument(
        "--force-prices",
        action="store_true",
        help="Re-fetch price history even if per-token parquet files exist.",
    )
    parser.add_argument(
        "--market-types",
        type=str,
        nargs="*",
        default=["moneyline"],
        help="Market types to fetch trade history for (default: moneyline). "
        "Event/market tables are always written in full.",
    )
    args = parser.parse_args()

    summary = ingest.ingest_polymarket(
        force_prices=args.force_prices,
        market_types=args.market_types,
    )
    n_events = summary.get("n_events", 0)
    type_counts = summary.get("market_type_counts", {})
    price_counts = summary.get("price_counts", {})
    print("\n=== Polymarket ingestion summary ===")
    print(f"events: {n_events}")
    print(f"market types: {type_counts}")
    print(f"price files: {price_counts}")


if __name__ == "__main__":
    main()
