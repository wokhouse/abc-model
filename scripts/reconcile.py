"""CLI entry point for reconciliation.

Usage:
    uv run python -m scripts.reconcile
"""
from __future__ import annotations

from abcm import reconcile


def main() -> None:
    summary = reconcile.reconcile_all()
    print("\n=== Reconciliation summary ===")
    for key, val in summary.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
