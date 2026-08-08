"""Orchestrate market→game reconciliation and write the aligned table.

Reads ``markets.parquet`` + nflverse ``schedules.parquet``, matches moneyline
markets to games, attaches pre-game price snapshots, and writes
``processed/aligned.parquet`` — the single joinable table downstream phases use.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .. import config, io
from . import match as match_mod
from . import snapshot as snap_mod


def reconcile_all() -> dict[str, Any]:
    """Run matching + snapshotting. Returns a summary dict."""
    config.ensure_dirs()
    summary: dict[str, Any] = {}

    markets = io.read_parquet(config.POLYMARKET_RAW_DIR / "markets.parquet")
    schedules = io.read_parquet(config.NFLVERSE_RAW_DIR / "schedules.parquet")
    print(
        f"[reconcile] {len(markets)} markets x {len(schedules)} scheduled games"
    )

    matched = match_mod.reconcile(markets, schedules)
    status_counts = matched["match_status"].value_counts().to_dict()
    summary["match_status_counts"] = status_counts
    n_ml_matched = status_counts.get("matched", 0)
    n_ml = (matched["market_type"] == "moneyline").sum()
    print(f"[reconcile] match status: {status_counts}")
    print(f"[reconcile] moneyline matched: {n_ml_matched}/{n_ml}")

    enriched = snap_mod.add_price_snapshots(matched, schedules)
    method_counts = enriched["snapshot_method"].value_counts().to_dict()
    summary["snapshot_method_counts"] = method_counts
    print(f"[reconcile] snapshot methods: {method_counts}")

    out_path = config.PROCESSED_DIR / "aligned.parquet"
    io.write_parquet(enriched, out_path)
    summary["n_rows"] = len(enriched)
    summary["path"] = str(out_path)
    print(f"[reconcile] wrote {len(enriched)} rows -> {out_path}")
    return summary
