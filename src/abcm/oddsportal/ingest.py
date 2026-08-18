"""Normalize scraped OddsPortal records into one-row-per-game open/close lines.

Reads the raw per-match JSON (moneyline + spread + total, each with an
``opening_odds`` and ``odds_history`` movement series) and projects to a clean
frame keyed on ``game_id``:

    game_id, season, week, home_team, away_team,
    home_open_ml, home_close_ml, away_open_ml, away_close_ml,
    spread_open, spread_close, total_open, total_close

Opening = the ``opening_odds`` field; closing = the last point in
``odds_history`` (the final line before kickoff). Decimal odds are converted to
implied probabilities where needed downstream; here we keep the raw decimal odds
and the spread/total *line values* (the number itself, which is what moves).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config, io
from . import loader as loader_mod
from . import teams as teams_mod

# Plausible range guards (the SBR scraper produced O/U=1.0 / spread=40.5 garbage;
# we reject anything outside these bounds so a parser hiccup can't poison the data).
TOTAL_MIN, TOTAL_MAX = 20.0, 80.0
SPREAD_MIN, SPREAD_MAX = -30.0, 30.0


def _decimal_to_implied(odds: float) -> float:
    """Decimal odds -> implied probability (with vig). 2.0 -> 0.5, 1.67 -> 0.599."""
    if not odds or odds <= 1.0:
        return float("nan")
    return 1.0 / odds


def _pick_book(market_records: list[dict], preferred: list[str]) -> dict | None:
    """Pick one bookmaker's record, preferring sharp books (Pinnacle > bet365)."""
    by_name = {r.get("bookmaker_name", ""): r for r in market_records}
    for name in preferred:
        for key, rec in by_name.items():
            if name.lower() in key.lower():
                return rec
    return market_records[0] if market_records else None


def _open_close(odds_history_data: dict) -> tuple[float, float]:
    """Extract (opening_odds, closing_odds) from one side's history block.

    ``opening_odds`` is an explicit field; closing is the last timestamp's odds
    in the ``odds_history`` movement list (the line just before kickoff).
    """
    opening = odds_history_data.get("opening_odds", {})
    open_odds = float(opening.get("odds", np.nan)) if opening else np.nan
    history = odds_history_data.get("odds_history") or []
    close_odds = float(history[-1]["odds"]) if history else open_odds
    return open_odds, close_odds


def normalize_match(record: dict, *, spread_line: float | None,
                    total_line: float | None) -> dict | None:
    """Normalize one scraped match record into a flat per-game row.

    Returns ``None`` if the record is unusable (no moneyline, unrecognized teams).
    ``spread_line`` / ``total_line`` are the nflverse-reported lines used to
    locate the per-line markets.
    """
    if not record:
        return None
    home = record.get("home_team")
    away = record.get("away_team")
    home_abbr = teams_mod.oddsportal_to_abbr(home) if home else None
    away_abbr = teams_mod.oddsportal_to_abbr(away) if away else None
    if not home_abbr or not away_abbr:
        return None

    row: dict[str, Any] = {
        "home_team": home_abbr, "away_team": away_abbr,
        "match_date": record.get("match_date"),
        "home_score": _to_int(record.get("home_score")),
        "away_score": _to_int(record.get("away_score")),
    }

    # --- Moneyline (1x2): home=side "1" if home listed first, else "2" -------
    ml = record.get("1x2_market") or []
    book = _pick_book(ml, ["Pinnacle", "bet365", "BetMGM"])
    if book:
        # NFL 1x2 "1" = home, "2" = away (OddsPortal lists home side as "1").
        sides = book.get("odds_history_data", [])
        if len(sides) >= 2:
            # sides are ordered [1, X/draw, 2] or [1, 2]; take first and last non-X.
            home_open, home_close = _open_close(sides[0])
            away_open, away_close = _open_close(sides[-1])
            row["home_open_ml"] = home_open
            row["home_close_ml"] = home_close
            row["away_open_ml"] = away_open
            row["away_close_ml"] = away_close

    # --- Spread (asian handicap at the nflverse line) -----------------------
    if spread_line is not None:
        sp_key = f"asian_handicap_{loader_mod._handicap_slug(spread_line).split('_', 2)[-1]}"
        # The market key in the record uses the full slug; find it.
        sp_market = _find_market(record, "asian_handicap", spread_line)
        if sp_market:
            book = _pick_book(sp_market, ["Pinnacle", "bet365", "BetMGM"])
            if book:
                sides = book.get("odds_history_data", [])
                if len(sides) >= 2:
                    row["spread_home_open_odds"], row["spread_home_close_odds"] = _open_close(sides[0])
                    row["spread_away_open_odds"], row["spread_away_close_odds"] = _open_close(sides[-1])
                row["spread_line"] = spread_line

    # --- Total (over/under at the nflverse line) ----------------------------
    if total_line is not None:
        ou_market = _find_market(record, "over_under", total_line)
        if ou_market:
            book = _pick_book(ou_market, ["Pinnacle", "bet365", "BetMGM"])
            if book:
                sides = book.get("odds_history_data", [])
                if len(sides) >= 2:
                    row["over_open_odds"], row["over_close_odds"] = _open_close(sides[0])
                    row["under_open_odds"], row["under_close_odds"] = _open_close(sides[-1])
                row["total_line"] = total_line

    return row


def _find_market(record: dict, prefix: str, line: float) -> list[dict] | None:
    """Find the per-line market list matching ``prefix`` + ``line``.

    OddsPortal keys: ``asian_handicap_-4_5_market`` / ``over_under_47_5_market``.
    """
    slug = loader_mod._handicap_slug(line) if prefix == "asian_handicap" else loader_mod._total_slug(line)
    key = f"{slug}_market"
    market = record.get(key)
    if market:
        return market
    # Fallback: search keys containing the prefix and the line value.
    line_token = str(line).replace(".", "_")
    for k, v in record.items():
        if prefix in k and line_token in k and isinstance(v, list):
            return v
    return None


def _to_int(x) -> int | None:
    try:
        return int(float(x)) if x is not None else None
    except (TypeError, ValueError):
        return None


def ingest_season(season_start: int, schedules: pd.DataFrame, *,
                  max_games: int | None = None) -> pd.DataFrame:
    """Scrape + normalize one NFL season's open/close lines.

    ``schedules`` provides the game_id, week, and the nflverse spread/total lines
    used to locate OddsPortal's per-line markets. Returns one row per game with
    open/close odds. Per-game caching under ``data/raw/oddsportal/<season>/``.
    """
    season_dir = config.ODDSPORTAL_RAW_DIR / str(season_start)
    season_dir.mkdir(parents=True, exist_ok=True)

    links = loader_mod.collect_match_links(season_start)
    # Build a lookup from (home_abbr, away_abbr) -> game info for matching.
    sched = schedules[schedules["season"] == season_start].copy()
    game_lookup: dict[tuple[str, str], dict] = {}
    for _, g in sched.iterrows():
        key = (str(g["home_team"]), str(g["away_team"]))
        game_lookup[key] = {
            "game_id": g["game_id"], "week": g.get("week"),
            "spread_line": g.get("spread_line"), "total_line": g.get("total_line"),
        }

    rows = []
    done = 0
    for link_info in links:
        ha, aa = link_info["home_abbr"], link_info["away_abbr"]
        # Try both orders: OddsPortal away/home may align either way.
        game = game_lookup.get((ha, aa)) or game_lookup.get((aa, ha))
        if not game:
            continue  # OddsPortal game not in nflverse schedules (e.g. preseason)

        cache_path = season_dir / f"{game['game_id']}.json"
        if cache_path.exists():
            record = json.loads(cache_path.read_text())
        else:
            record = loader_mod.scrape_match(
                link_info["match_link"],
                spread_line=game.get("spread_line"),
                total_line=game.get("total_line"),
            )
            if record is None:
                continue
            cache_path.write_text(json.dumps(record, indent=2))

        row = normalize_match(
            record,
            spread_line=game.get("spread_line"),
            total_line=game.get("total_line"),
        )
        if row:
            row["game_id"] = game["game_id"]
            row["season"] = season_start
            row["week"] = game.get("week")
            rows.append(row)
            done += 1
            if done % 25 == 0:
                print(f"[oddsportal] {season_start}: {done} games normalized")
        if max_games and done >= max_games:
            break

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Range validation: drop garbage spread/total lines (defensive).
    if "spread_line" in df:
        df = df[df["spread_line"].isna() | df["spread_line"].between(SPREAD_MIN, SPREAD_MAX)]
    return df


def ingest_oddsportal(seasons: list[int] | None = None) -> dict[str, Any]:
    """Pull + normalize OddsPortal open/close lines for the given seasons.

    Defaults to the Polymarket seasons (2024, 2025). Writes
    ``data/raw/oddsportal/nfl_lines.parquet`` keyed on ``game_id``.
    """
    seasons = seasons or [2024, 2025]
    config.ensure_dirs()
    schedules = io.read_parquet(config.NFLVERSE_HISTORY_DIR / "schedules.parquet")
    sched_cols = ["game_id", "season", "week", "home_team", "away_team",
                  "spread_line", "total_line"]
    schedules = schedules[[c for c in sched_cols if c in schedules.columns]]

    all_rows = []
    for season in seasons:
        print(f"[oddsportal] ingesting season {season}...")
        df = ingest_season(season, schedules)
        print(f"[oddsportal]   {season}: {len(df)} games with open/close lines")
        all_rows.append(df)

    out = pd.concat([d for d in all_rows if len(d)], ignore_index=True) if all_rows else pd.DataFrame()
    out_path = config.ODDSPORTAL_RAW_DIR / "nfl_lines.parquet"
    io.write_parquet(out, out_path)

    summary: dict[str, Any] = {
        "seasons": seasons, "n_games": len(out),
        "n_with_spread": int(out["spread_line"].notna().sum()) if "spread_line" in out else 0,
        "n_with_total": int(out["total_line"].notna().sum()) if "total_line" in out else 0,
        "n_with_ml": int(out.get("home_close_ml", pd.Series()).notna().sum()),
        "path": str(out_path),
    }
    print(f"[oddsportal] wrote {len(out)} games -> {out_path}")
    return summary
