"""Pull raw nflverse data for the target seasons and persist to parquet.

Play-by-play is the largest table (hundreds of thousands of rows per season);
the others are small. All are cached by nflreadpy, so re-runs are cheap.
"""
from __future__ import annotations

from typing import Any

from .. import config, io
from . import loader


def ingest_nfl(years: list[int] | None = None) -> dict[str, Any]:
    """Download pbp, schedules, rosters, and team stats; write to raw/nflverse/.

    Returns a summary dict with row counts per table.
    """
    years = years or config.YEARS
    config.ensure_dirs()
    summary: dict[str, Any] = {"years": years}

    print(f"[nfl] loading schedules {years}...")
    schedules = loader.load_schedules(years)
    io.write_parquet(schedules, config.NFLVERSE_RAW_DIR / "schedules.parquet")
    summary["schedules_rows"] = len(schedules)
    print(f"[nfl] schedules: {len(schedules)} games")

    print(f"[nfl] loading team stats {years}...")
    team_stats = loader.load_team_stats(years)
    io.write_parquet(team_stats, config.NFLVERSE_RAW_DIR / "team_stats.parquet")
    summary["team_stats_rows"] = len(team_stats)
    print(f"[nfl] team_stats: {len(team_stats)} rows")

    print(f"[nfl] loading rosters {years}...")
    rosters = loader.load_rosters(years)
    io.write_parquet(rosters, config.NFLVERSE_RAW_DIR / "rosters.parquet")
    summary["rosters_rows"] = len(rosters)
    print(f"[nfl] rosters: {len(rosters)} rows")

    print(f"[nfl] loading play-by-play {years} (this is the big one)...")
    pbp = loader.load_pbp(years)
    io.write_parquet(pbp, config.NFLVERSE_RAW_DIR / "pbp.parquet")
    summary["pbp_rows"] = len(pbp)
    print(f"[nfl] pbp: {len(pbp)} plays")

    return summary


def ingest_nfl_history(years: list[int] | None = None) -> dict[str, Any]:
    """Download the deep nflverse history for two-stage transfer pre-training.

    Pulls schedules, team stats, rosters, and pbp for a long year range
    (``config.NFLVERSE_HISTORY_YEARS``, default 1999-current) and writes them to
    ``raw/nflverse_history/`` — separate from the fast Polymarket-aligned
    ``raw/nflverse/`` set so the modeling target data is undisturbed. This is the
    unlabeled/parallel history the ML research recommended: decades of
    game-outcome signal (EPA, CPOE, etc.) to pre-train an outcome model whose
    predictions then become the ``market_delta`` inefficiency feature.

    Re-runs re-download; nflreadpy caches under the hood so subsequent runs are
    cheap. Returns a summary dict with row counts per table.
    """
    years = years or config.NFLVERSE_HISTORY_YEARS
    config.ensure_dirs()
    summary: dict[str, Any] = {"years": years, "n_years": len(years)}

    print(f"[nfl-history] loading schedules {min(years)}-{max(years)} ({len(years)} seasons)...")
    schedules = loader.load_schedules(years)
    io.write_parquet(schedules, config.NFLVERSE_HISTORY_DIR / "schedules.parquet")
    summary["schedules_rows"] = len(schedules)
    print(f"[nfl-history] schedules: {len(schedules)} games")

    print(f"[nfl-history] loading team stats {min(years)}-{max(years)}...")
    team_stats = loader.load_team_stats(years)
    io.write_parquet(team_stats, config.NFLVERSE_HISTORY_DIR / "team_stats.parquet")
    summary["team_stats_rows"] = len(team_stats)
    print(f"[nfl-history] team_stats: {len(team_stats)} rows")

    print(f"[nfl-history] loading rosters {min(years)}-{max(years)}...")
    rosters = loader.load_rosters(years)
    io.write_parquet(rosters, config.NFLVERSE_HISTORY_DIR / "rosters.parquet")
    summary["rosters_rows"] = len(rosters)
    print(f"[nfl-history] rosters: {len(rosters)} rows")

    print(f"[nfl-history] loading play-by-play {min(years)}-{max(years)} (this is the big one)...")
    pbp = loader.load_pbp(years)
    io.write_parquet(pbp, config.NFLVERSE_HISTORY_DIR / "pbp.parquet")
    summary["pbp_rows"] = len(pbp)
    print(f"[nfl-history] pbp: {len(pbp)} plays")

    return summary
