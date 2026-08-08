"""CLI entry point for Elo rating computation.

Usage:
    uv run python -m scripts.build_elo
"""
from __future__ import annotations

from abcm.features import elo


def main() -> None:
    summary = elo.build_and_save()
    print("\n=== Elo build summary ===")
    for key, val in summary.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
