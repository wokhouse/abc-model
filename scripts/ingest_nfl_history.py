"""CLI entry point for deep nflverse history ingestion (transfer pre-training).

Usage:
    uv run python -m scripts.ingest_nfl_history
    uv run python -m scripts.ingest_nfl_history --years 1999 2000 2010
"""
from __future__ import annotations

import argparse

from abcm import config
from abcm.nfl import ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download deep nflverse history (1999+) for two-stage transfer "
            "pre-training. Writes to raw/nflverse_history/, separate from the "
            "fast target-season pull."
        )
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=config.NFLVERSE_HISTORY_YEARS,
        help="Seasons to pull (default: 1999-current from config).",
    )
    args = parser.parse_args()

    summary = ingest.ingest_nfl_history(years=args.years)
    print("\n=== nflverse history ingestion summary ===")
    for key, val in summary.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
