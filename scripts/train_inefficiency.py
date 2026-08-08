"""CLI entry point for Stage 2 inefficiency-model training.

Usage:
    uv run python -m scripts.train_inefficiency
"""
from __future__ import annotations

from abcm.model import inefficiency


def main() -> None:
    summary = inefficiency.build_and_save()
    print("\n=== Inefficiency model summary ===")
    for key, val in summary.items():
        if key in ("pools", "rolling"):
            continue  # already printed verbosely during the run
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
