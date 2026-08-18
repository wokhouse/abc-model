"""CLI entry point for Stage 1 outcome-model pre-training.

Usage:
    uv run python -m scripts.train_outcome
"""
from __future__ import annotations

from abcm.model import outcome


def main() -> None:
    summary = outcome.build_and_save()
    print("\n=== Outcome model summary ===")
    for key, val in summary.items():
        if key in ("gate", "multi_fold"):
            continue  # already printed verbosely during the run
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
