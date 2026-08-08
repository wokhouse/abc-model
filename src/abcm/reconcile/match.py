"""Match Polymarket game markets to nflverse games and resolve outcomes.

Two market types are matched to games:

* **moneyline** — two team-nickname outcomes (e.g. ``["Texans","Bears"]``). We map
  each to an nflverse abbr, then locate the game where those teams met within a
  date window of the market's end date.
* **spread** — phrased as ``"Spread: Steelers (-5.5)"`` or
  ``"Will the Chiefs win by 4 or more points?"`` with ``["Yes","No"]`` outcomes.
  The favored team and margin come from the question; *both* teams come from the
  event title (``"Steelers vs. Panthers"``), since the Yes/No outcomes carry no
  team identity.

Polymarket's ``outcomePrices`` (captured as ``resolved_yes_price``) tells us
which outcome won; we cross-check that against the game result to flag
mismatches for the EDA.
"""
from __future__ import annotations

import json
import re
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

# --- Spread parsing --------------------------------------------------------
#
# Spread markets carry no team identity in their Yes/No outcomes; we recover the
# favored team + margin from the question and *both* teams from the event title.

# "Spread: Steelers (-5.5)" / "Spread: Jets (1.5)". A negative margin means the
# team is favored by |margin|; a positive margin means they are getting points.
_SPREAD_QUESTION_RE = re.compile(
    r"^\s*Spread:\s*(.+?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*$", re.IGNORECASE
)
# "Will the Chiefs win by 4 or more points?" — favored team wins by >= margin.
_MARGIN_OF_VICTORY_RE = re.compile(
    r"^\s*(?:Will\s+)?(?:the\s+)?(.+?)\s+(?:win|beat|defeat)\b.*?\bby\s+(\d+(?:\.\d+)?)\s+or\s+more\s+points",
    re.IGNORECASE,
)
# "X vs. Y" / "X vs Y" matchup in the event title (may have a leading scope like
# "NFL Kickoff: " or "NFL Week 1: " that we strip).
_VS_RE = re.compile(r"\bv\.?s\.?\b", re.IGNORECASE)


def _teams_from_event_title(event_title: str | None) -> list[str]:
    """Extract two team abbrs from a matchup title like 'Steelers vs. Panthers'.

    Returns a list of 0, 1, or 2 abbrs (unmatched tokens drop out). Polymarket
    titles sometimes carry a leading scope ('NFL Kickoff:', 'NFL Week 1:') that
    we split off before parsing the team pair.
    """
    if not event_title:
        return []
    parts = _VS_RE.split(event_title)
    if len(parts) != 2:
        return []
    # Each side may carry trailing scope noise (e.g. 'NFL Kickoff: Chiefs' after
    # a split on ' vs '); take the last whitespace token of each side as the team.
    abbrs: list[str] = []
    for side in parts:
        side = side.strip()
        # Drop a trailing scope segment ('NFL Kickoff:', 'NFL:') if present.
        if ":" in side:
            side = side.rsplit(":", 1)[1].strip()
        # The team is the last 1-2 tokens; try the full tail first, then last word.
        tokens = side.split()
        for n in (len(tokens), len(tokens) - 1):
            if 0 < n <= len(tokens):
                candidate = " ".join(tokens[-n:]) if n > 1 else tokens[-1]
                abbr = teams.nickname_to_abbr(candidate)
                if abbr is not None:
                    abbrs.append(abbr)
                    break
    return abbrs


def parse_spread(question: str | None, event_title: str | None) -> dict[str, Any]:
    """Extract (favored_abbr, opponent_abbr, margin) from a spread market.

    ``margin`` is the favored team's point spread (always >= 0). For
    ``"Spread: Jets (1.5)"`` the named team is the *underdog* (getting 1.5), so
    we flip: the opponent is the favorite. Returns empty values if unparseable.
    """
    result: dict[str, Any] = {
        "spread_favored_abbr": None,
        "spread_opponent_abbr": None,
        "spread_margin": None,
    }
    if not question:
        return result

    favored_raw: str | None = None
    margin: float | None = None
    named_is_favorite = True

    m = _SPREAD_QUESTION_RE.match(question)
    if m:
        favored_raw = m.group(1)
        signed = float(m.group(2))
        # "Spread: Steelers (-5.5)" -> favored by 5.5. "Spread: Jets (1.5)" ->
        # Jets are getting 1.5, i.e. the underdog; opponent is favored by 1.5.
        if signed < 0:
            margin = abs(signed)
            named_is_favorite = True
        else:
            margin = signed
            named_is_favorite = False
    else:
        m = _MARGIN_OF_VICTORY_RE.match(question)
        if m:
            favored_raw = m.group(1)
            margin = float(m.group(2))
            named_is_favorite = True

    if favored_raw is None or margin is None:
        return result

    named_abbr = teams.nickname_to_abbr(favored_raw)

    # Recover the opponent from the event title's "X vs. Y" matchup.
    title_abbrs = _teams_from_event_title(event_title)
    opponent_abbr = None
    if named_abbr is not None and len(title_abbrs) == 2:
        opponent_abbr = next((a for a in title_abbrs if a != named_abbr), None)

    if named_abbr is None:
        # Favored team name didn't resolve; keep opponent pair if both resolved.
        if len(title_abbrs) == 2:
            # Without a named favorite we can't orient the spread; bail.
            return result
        return result

    if named_is_favorite:
        result["spread_favored_abbr"] = named_abbr
        result["spread_opponent_abbr"] = opponent_abbr
    else:
        # Named team is the underdog; opponent (from the title) is the favorite.
        result["spread_favored_abbr"] = opponent_abbr
        result["spread_opponent_abbr"] = named_abbr
    result["spread_margin"] = margin
    return result


def spread_covered(
    favored_abbr: str | None,
    margin: float | None,
    home_team: str | None,
    away_team: str | None,
    home_score: float | None,
    away_score: float | None,
) -> bool | None:
    """Did the favored team cover the spread?

    Returns True/False if determinable, None if inputs are missing or the
    favored team isn't one of the game's two teams (a data-quality red flag).
    """
    if favored_abbr is None or margin is None:
        return None
    if home_score is None or away_score is None:
        return None
    if favored_abbr not in {home_team, away_team}:
        return None
    if favored_abbr == home_team:
        fav_score, und_score = home_score, away_score
    else:
        fav_score, und_score = away_score, home_score
    # Favorite covers if they win by more than the margin; a push (exact margin)
    # is treated as not-covered for binary labeling.
    return (fav_score - und_score) > margin


def _find_game(
    abbrs: list[str],
    market_day: pd.Timestamp | None,
    game_index: dict[tuple[str, str, pd.Timestamp], pd.Series],
) -> pd.Series | None:
    """Locate the scheduled game where two teams met near ``market_day``."""
    if market_day is None or len(abbrs) != 2 or not all(abbrs):
        return None
    for delta in range(-_MATCH_WINDOW_DAYS, _MATCH_WINDOW_DAYS + 1):
        key = _game_key(abbrs[0], abbrs[1], market_day + pd.Timedelta(days=delta))
        if key in game_index:
            return game_index[key]
    return None


def match_one(
    row: pd.Series,
    game_index: dict[tuple[str, str, pd.Timestamp], pd.Series],
) -> dict[str, Any]:
    """Match a single moneyline/spread market row to a game.

    Returns enrichment fields including team abbrs, the matched game_id, a
    match_status, and (for spreads) the parsed favored team / margin / cover.
    """
    market_type = row.get("market_type")
    market_day = _safe_date(row.get("market_end"))
    if market_day is None:
        market_day = _safe_date(row.get("event_end"))

    # Default enrichment fields (kept consistent across types).
    result: dict[str, Any] = {
        "pm_team0_abbr": None,
        "pm_team1_abbr": None,
        "game_id": None,
        "nfl_home_team": None,
        "nfl_away_team": None,
        "match_status": "no_team_outcomes",
        "spread_favored_abbr": None,
        "spread_opponent_abbr": None,
        "spread_margin": None,
    }

    if market_type == "spread":
        return _match_spread(row, market_day, game_index, result)
    if market_type == "moneyline":
        return _match_moneyline(row, market_day, game_index, result)
    result["match_status"] = "not_moneyline"
    return result


def _match_moneyline(
    row: pd.Series,
    market_day: pd.Timestamp | None,
    game_index: dict[tuple[str, str, pd.Timestamp], pd.Series],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Match a moneyline market via its two team-nickname outcomes."""
    outcomes = _parse_json_list(row.get("outcomes"))
    abbrs = [teams.nickname_to_abbr(o) for o in outcomes]
    if len(abbrs) >= 1:
        result["pm_team0_abbr"] = abbrs[0]
    if len(abbrs) >= 2:
        result["pm_team1_abbr"] = abbrs[1]

    if len(abbrs) != 2 or not all(abbrs):
        return result  # leave match_status='no_team_outcomes'

    found = _find_game(abbrs, market_day, game_index)
    if found is None:
        result["match_status"] = "teams_found_no_game"
        return result
    result["match_status"] = "matched"
    result["game_id"] = found.get("game_id")
    result["nfl_home_team"] = found.get("home_team")
    result["nfl_away_team"] = found.get("away_team")
    return result


def _match_spread(
    row: pd.Series,
    market_day: pd.Timestamp | None,
    game_index: dict[tuple[str, str, pd.Timestamp], pd.Series],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Match a spread market: parse favored team + margin, then locate the game."""
    parsed = parse_spread(row.get("question"), row.get("event_title"))
    result["spread_favored_abbr"] = parsed["spread_favored_abbr"]
    result["spread_opponent_abbr"] = parsed["spread_opponent_abbr"]
    result["spread_margin"] = parsed["spread_margin"]
    # Mirror the two teams into pm_team abbrs for downstream uniformity.
    result["pm_team0_abbr"] = parsed["spread_favored_abbr"]
    result["pm_team1_abbr"] = parsed["spread_opponent_abbr"]

    fav = parsed["spread_favored_abbr"]
    opp = parsed["spread_opponent_abbr"]
    if fav is None or opp is None:
        result["match_status"] = "no_team_outcomes"
        return result

    found = _find_game([fav, opp], market_day, game_index)
    if found is None:
        result["match_status"] = "teams_found_no_game"
        return result
    result["match_status"] = "matched"
    result["game_id"] = found.get("game_id")
    result["nfl_home_team"] = found.get("home_team")
    result["nfl_away_team"] = found.get("away_team")
    return result


def reconcile(markets: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Join moneyline and spread markets to nflverse games. One row per market.

    Other market types are kept but marked ``match_status='not_moneyline'`` so
    the EDA can report the full population; matching only runs on moneylines
    and spreads.
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
        "spread_favored_abbr",
        "spread_opponent_abbr",
        "spread_margin",
    ]
    return pd.concat([out, enrichments], axis=1)
