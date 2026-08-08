"""CLI entry point for nflverse ingestion.

Usage:
    uv run python -m scripts.ingest_nfl                  # default years from config
    uv run python -m scripts.ingest_nfl --years 2024 2025
"""
from __future__ import annotations

import argparse

from abcm import config
from abcm.nfl import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest nflverse data via nflreadpy.")
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=config.YEARS,
        help="Seasons to pull (default: from config).",
    )
    args = parser.parse_args()

    summary = ingest.ingest_nfl(years=args.years)
    print("\n=== nflverse ingestion summary ===")
    for key, val in summary.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
