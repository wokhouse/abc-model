"""CLI entry point for the Stage 1b cover-probability model.

Usage:
    uv run python -m scripts.train_cover
"""
from __future__ import annotations

from abcm.model import cover


def main() -> None:
    summary = cover.build_and_save()
    print("\n=== Cover model summary ===")
    for key, val in summary.items():
        if key in ("gate", "multi_fold"):
            continue
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
