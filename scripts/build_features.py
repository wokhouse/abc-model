"""CLI entry point for Stage 0 feature engineering.

Usage:
    uv run python -m scripts.build_features
"""
from __future__ import annotations

from abcm.features import efficiency


def main() -> None:
    summary = efficiency.build_and_save()
    print("\n=== Feature build summary ===")
    for key, val in summary.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
