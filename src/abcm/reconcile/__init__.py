"""Orchestrate market→game reconciliation and write the aligned table.

Reads ``markets.parquet`` + nflverse ``schedules.parquet``, re-classifies market
types (the stored labels may predate a classifier fix), matches moneyline and
spread markets to games, attaches pre-game price snapshots (incl. the open→close
delta), and joins derived features — Elo ratings, travel/timezone, sportsbook
betting lines — where they exist. Writes ``processed/aligned.parquet``: the
single joinable table downstream phases use.

Feature tables are optional: if ``elo_ratings.parquet`` / ``betting_lines.parquet``
don't exist yet the join is skipped with a warning rather than failing, so the
pipeline can run in stages.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .. import config, io
from ..polymarket import events as events_mod
from . import match as match_mod
from . import snapshot as snap_mod


def _reclassify_market_types(markets: pd.DataFrame) -> pd.DataFrame:
    """Re-run the classifier over stored markets (cheap; labels may be stale)."""
    import json

    def _outcomes(raw):
        if raw is None:
            return []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []
        return [str(x) for x in raw] if isinstance(raw, list) else []

    out = markets.copy()
    out["market_type"] = out.apply(
        lambda r: events_mod.classify_market(
            r.get("question"), event_title=r.get("event_title"),
            outcomes=_outcomes(r.get("outcomes")),
        ),
        axis=1,
    )
    return out


def _try_join(left: pd.DataFrame, path, on: list[str], label: str) -> pd.DataFrame:
    """Left-join a feature table if it exists; otherwise pass through unchanged."""
    if path is None or not path.exists():
        print(f"[reconcile] {label} not found ({path}); skipping {label} join")
        return left
    right = io.read_parquet(path)
    merged = left.merge(right, on=on, how="left", suffixes=("", f"_{label}"))
    print(f"[reconcile] joined {label} ({len(right)} rows) on {on}")
    return merged


def reconcile_all() -> dict[str, Any]:
    """Run matching + snapshotting + feature joins. Returns a summary dict."""
    config.ensure_dirs()
    summary: dict[str, Any] = {}

    markets = io.read_parquet(config.POLYMARKET_RAW_DIR / "markets.parquet")
    markets = _reclassify_market_types(markets)
    schedules = io.read_parquet(config.NFLVERSE_RAW_DIR / "schedules.parquet")
    print(
        f"[reconcile] {len(markets)} markets x {len(schedules)} scheduled games"
    )
    summary["market_type_counts"] = markets["market_type"].value_counts().to_dict()

    matched = match_mod.reconcile(markets, schedules)
    status_counts = matched["match_status"].value_counts().to_dict()
    summary["match_status_counts"] = status_counts
    n_matched = status_counts.get("matched", 0)
    n_ml = (matched["market_type"] == "moneyline").sum()
    n_sp = (matched["market_type"] == "spread").sum()
    n_ml_matched = ((matched["market_type"] == "moneyline") & (matched["match_status"] == "matched")).sum()
    n_sp_matched = ((matched["market_type"] == "spread") & (matched["match_status"] == "matched")).sum()
    print(f"[reconcile] match status: {status_counts}")
    print(f"[reconcile] moneyline matched: {n_ml_matched}/{n_ml}")
    print(f"[reconcile] spread matched: {n_sp_matched}/{n_sp}")

    enriched = snap_mod.add_price_snapshots(matched, schedules)
    method_counts = enriched["snapshot_method"].value_counts().to_dict()
    summary["snapshot_method_counts"] = method_counts
    print(f"[reconcile] snapshot methods: {method_counts}")

    # --- Feature joins (on game_id where matched) --------------------------
    # Elo ratings: one row per game, pre-game ratings + home win prob.
    elo_path = config.ELO_PROCESSED_DIR / "elo_ratings.parquet"
    elo_cols = ["game_id", "elo_home_pre", "elo_away_pre", "elo_home_prob"]
    if elo_path.exists():
        elo = io.read_parquet(elo_path)[elo_cols]
        enriched = enriched.merge(elo, on="game_id", how="left")
        print(f"[reconcile] joined Elo ({len(elo)} games)")
        n_elo = enriched["elo_home_pre"].notna().sum()
        summary["elo_coverage"] = int(n_elo)
    else:
        print(f"[reconcile] Elo ratings not found ({elo_path}); run scripts.build_elo")

    # Travel features: attach to schedules first, then carry via game_id.
    from ..features import travel as travel_mod

    sched_travel = travel_mod.game_travel_features(schedules)
    travel_cols = ["game_id", "travel_distance_km", "timezone_shift_h", "neutral_site"]
    enriched = enriched.merge(
        sched_travel[travel_cols], on="game_id", how="left"
    )
    n_travel = enriched["travel_distance_km"].notna().sum()
    summary["travel_coverage"] = int(n_travel)
    print(f"[reconcile] joined travel features ({n_travel} rows with distance)")

    # Sportsbook betting lines (Kaggle): join on (season, week, home_team).
    bl_path = config.KAGGLE_RAW_DIR / "betting_lines.parquet"
    if bl_path.exists():
        bl = io.read_parquet(bl_path)
        # Bring season/week/home_team onto the enriched frame via the matched game.
        sched_keys = schedules[["game_id", "season", "week", "home_team"]].dropna(
            subset=["game_id"]
        ).rename(columns={"home_team": "nfl_home_team"})
        enriched = enriched.merge(sched_keys, on=["game_id", "nfl_home_team"], how="left")
        bl_keys = bl.rename(columns={
            "schedule_season": "season",
            "schedule_week": "week",
            "home_team_abbr": "nfl_home_team",
        })[["season", "week", "nfl_home_team", "team_favorite_abbr", "spread_favorite", "over_under_line"]].rename(columns={
            "team_favorite_abbr": "sportsbook_favorite",
            "spread_favorite": "sportsbook_spread",
            "over_under_line": "sportsbook_ou",
        })
        # Only regular-season games have a matching betting line (week is int).
        enriched = enriched.merge(bl_keys, on=["season", "week", "nfl_home_team"], how="left")
        n_bl = enriched["sportsbook_spread"].notna().sum()
        summary["betting_line_coverage"] = int(n_bl)
        print(f"[reconcile] joined betting lines ({n_bl} rows with a spread)")
    else:
        print(f"[reconcile] betting lines not found ({bl_path}); run scripts.ingest_kaggle")

    # OddsPortal opening + closing lines (richer than the single Kaggle line):
    # per-game open/close moneyline + spread/total odds, keyed on game_id. Enables
    # the Stage 2 line-movement features and Stage 3 CLV analysis. Optional: if the
    # scrape hasn't been run the join is skipped (the thin Kaggle line still above).
    op_path = config.ODDSPORTAL_RAW_DIR / "nfl_lines.parquet"
    if op_path.exists():
        op = io.read_parquet(op_path)
        # Keep the open/close columns; drop game_id collision (already on enriched).
        op_cols = [c for c in op.columns if c != "season" and c != "week"]
        op_keys = op[op_cols].drop_duplicates(subset=["game_id"])
        enriched = enriched.merge(op_keys, on="game_id", how="left", suffixes=("", "_op"))
        n_op = enriched["home_close_ml"].notna().sum() if "home_close_ml" in enriched else 0
        summary["oddsportal_coverage"] = int(n_op)
        print(f"[reconcile] joined OddsPortal open/close lines ({n_op} rows, {len(op)} games available)")
    else:
        print(f"[reconcile] OddsPortal lines not found ({op_path}); run scripts.ingest_oddsportal")

    out_path = config.PROCESSED_DIR / "aligned.parquet"
    io.write_parquet(enriched, out_path)
    summary["n_rows"] = len(enriched)
    summary["path"] = str(out_path)
    print(f"[reconcile] wrote {len(enriched)} rows -> {out_path}")
    return summary
