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
