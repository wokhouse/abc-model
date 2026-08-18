"""CLI entry point for Kaggle betting-line ingestion.

Usage:
    uv run python -m scripts.ingest_kaggle                  # default years from config
    uv run python -m scripts.ingest_kaggle --years 2024 2025
"""
from __future__ import annotations

import argparse

from abcm import config
from abcm.kaggle import ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download + normalize Kaggle NFL betting-line data."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=config.YEARS,
        help="Seasons to keep (default: from config).",
    )
    args = parser.parse_args()

    summary = ingest.ingest_kaggle(years=args.years)
    print("\n=== Kaggle ingestion summary ===")
    for key, val in summary.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
