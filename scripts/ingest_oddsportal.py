"""CLI entry point for OddsPortal NFL line ingestion.

Usage:
    uv run python -m scripts.ingest_oddsportal
"""
from __future__ import annotations

from abcm.oddsportal import ingest


def main() -> None:
    summary = ingest.ingest_oddsportal()
    print("\n=== OddsPortal ingest summary ===")
    for key, val in summary.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
