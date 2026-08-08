"""Match Polymarket moneyline markets to nflverse games and resolve winners.

A moneyline market's two outcomes are team nicknames (e.g. ``["Texans","Bears"]``).
We map each to an nflverse abbr, then locate the NFL game where those two teams
met within a date window of the market's end date. Polymarket's ``outcomePrices``
tells us which outcome won; we cross-check that against the game result to flag
mismatches for the EDA.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .. import config, io
from . import teams


def _parse_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [str(x) for x in raw] if isinstance(raw, list) else []


def _safe_date(val: Any) -> pd.Timestamp | None:
    """Parse Polymarket's mixed date formats ('2024-08-01T12:00:00Z',
    '2025-01-27 06:44:01+00') to a tz-naive Timestamp (or None)."""
    if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        ts = pd.to_datetime(val, utc=True, errors="coerce")
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT:
        return None
    return ts.tz_localize(None)


def prepare_schedules(schedules: pd.DataFrame) -> pd.DataFrame:
    """Project nflverse schedules to the columns the matcher needs."""
    df = schedules.copy()
    df["gameday_dt"] = pd.to_datetime(df["gameday"], errors="coerce")
    # Index by team-pair + date for O(1) lookup. A game's two teams are unordered,
    # so we store both (away,home) and (home,away) keys.
    df = df.dropna(subset=["gameday_dt"])
    return df


def _game_key(a: str, b: str, day: pd.Timestamp) -> tuple[str, str, pd.Timestamp]:
    """Order-independent team pair key for a given day."""
    pair = tuple(sorted((a, b)))
    return (pair[0], pair[1], day.normalize())  # type: ignore[index]


def build_game_index(schedules: pd.DataFrame) -> dict[tuple[str, str, pd.Timestamp], pd.Series]:
    """Map (team_a, team_b, day) -> schedule row, for a ±N day lookup window.

    We index at day granularity and search the window around the market's end
    date at match time, since kickoff date is the stable signal.
    """
    index: dict[tuple[str, str, pd.Timestamp], pd.Series] = {}
    for _, row in schedules.iterrows():
        away = row.get("away_team")
        home = row.get("home_team")
        day = row.get("gameday_dt")
        if pd.isna(day) or not away or not home:
            continue
        index[_game_key(away, home, day)] = row
    return index


# How many days on either side of the market end date to look for the game.
_MATCH_WINDOW_DAYS = 3


def match_one(
    row: pd.Series,
    game_index: dict[tuple[str, str, pd.Timestamp], pd.Series],
) -> dict[str, Any]:
    """Match a single moneyline market row to a game; return enrichment fields."""
    outcomes = _parse_json_list(row.get("outcomes"))
    abbrs = [teams.nickname_to_abbr(o) for o in outcomes]
    matched_game_id = None
    home_team = away_team = None
    match_status = "no_team_outcomes"

    if len(abbrs) == 2 and all(a is not None for a in abbrs):
        market_day = _safe_date(row.get("market_end"))
        if market_day is None:
            market_day = _safe_date(row.get("event_end"))

        found = None
        if market_day is not None:
            for delta in range(-_MATCH_WINDOW_DAYS, _MATCH_WINDOW_DAYS + 1):
                key = _game_key(abbrs[0], abbrs[1], market_day + pd.Timedelta(days=delta))
                if key in game_index:
                    found = game_index[key]
                    break

        if found is not None:
            matched_game_id = found.get("game_id")
            home_team = found.get("home_team")
            away_team = found.get("away_team")
            match_status = "matched"
        else:
            match_status = "teams_found_no_game"

    return {
        "pm_team0_abbr": abbrs[0] if len(abbrs) >= 1 else None,
        "pm_team1_abbr": abbrs[1] if len(abbrs) >= 2 else None,
        "game_id": matched_game_id,
        "nfl_home_team": home_team,
        "nfl_away_team": away_team,
        "match_status": match_status,
    }


def reconcile(markets: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Join moneyline markets to nflverse games. Returns one row per market.

    Non-moneyline markets are kept but marked ``match_status='not_moneyline'`` so
    the EDA can report the full population; matching only runs on moneylines.
    """
    sched = prepare_schedules(schedules)
    game_index = build_game_index(sched)

    out = markets.copy()
    # ``outcomes`` in the flat frame is the raw value from Gamma; normalize to list.
    out["outcomes_list"] = out["outcomes"].apply(_parse_json_list)

    enrichments = out.apply(lambda r: match_one(r, game_index), axis=1, result_type="expand")
    enrichments.columns = [
        "pm_team0_abbr",
        "pm_team1_abbr",
        "game_id",
        "nfl_home_team",
        "nfl_away_team",
        "match_status",
    ]
    # Rows that aren't moneyline (per our classifier) get a distinct status.
    is_moneyline = out["market_type"] == "moneyline"
    enrichments.loc[~is_moneyline, "match_status"] = "not_moneyline"

    return pd.concat([out, enrichments], axis=1)
