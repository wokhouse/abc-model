"""Load + validate the sportsbookreviewsonline.com NFL archive (2011-2021).

The pre-scraped JSON (from the flancast90/sportsbookreview-scraper repo) carries
opening and closing spreads + totals for 2,956 games. Its parser has two known
bugs we defend against:

1. **Spread/total swap** — when the scraper mis-guesses which HTML row holds the
   spread vs the total, it emits impossible values (O/U=1.0, spread=40.5). We
   drop any row whose spread is outside ±30 or whose total is outside [20, 80].
2. **Team-name noise** — nicknames with typos/dupes/relocations
   (``Washingtom``, ``Fortyniners``, ``BuffaloBills`` vs ``Bills``, ``Oakland``,
   ``St.Louis``). A full nickname->abbr map normalizes them.

Output: one row per clean game, keyed on ``game_id`` (joined to nflverse
schedules by date + teams), with open/close spreads, totals, and the cover label.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .. import config

# Plausible NFL ranges (anything outside is parser garbage, dropped).
SPREAD_MIN, SPREAD_MAX = -30.0, 30.0
TOTAL_MIN, TOTAL_MAX = 20.0, 80.0

# SBR nickname -> nflverse abbr. Covers all 44 surface forms in the archive,
# including typos (Washingtom), dupes (Bills/BuffaloBills), and relocations
# (Oakland->LV, SanDiego->LAC, St.Louis->LA). "0" and unknowns -> None.
SBR_NICKNAME_TO_ABBR: dict[str, str] = {
    "bears": "CHI", "bengals": "CIN", "bills": "BUF", "buffalobills": "BUF",
    "broncos": "DEN", "browns": "CLE", "buccaneers": "TB", "tampa": "TB",
    "cardinals": "ARI", "chargers": "LAC", "sandiego": "LAC",
    "chiefs": "KC", "kcchiefs": "KC", "kansas": "KC",
    "colts": "IND", "commanders": "WAS", "washingtom": "WAS",  # typo in source
    "cowboys": "DAL", "dolphins": "MIA", "eagles": "PHI", "falcons": "ATL",
    "fortyniners": "SF", "giants": "NYG", "jaguars": "JAX", "jets": "NYJ",
    "newyork": "NYJ",  # ambiguous; Jets appear as "NewYork" in the archive
    "lions": "DET", "packers": "GB", "panthers": "CAR", "patriots": "NE",
    "raiders": "LV", "oakland": "LV", "lvraiders": "LV",
    "rams": "LA", "losangeles": "LA", "st.louis": "LA",
    "ravens": "BAL", "saints": "NO", "seahawks": "SEA",
    "steelers": "PIT", "texans": "HOU", "titans": "TEN", "vikings": "MIN",
}


def _parse_date(date_float: float) -> str | None:
    """``20110908.0`` -> ``"2011-09-08"`` (ISO). None if unparseable."""
    try:
        s = str(int(float(date_float)))
        return datetime.strptime(s, "%Y%m%d").date().isoformat()
    except (TypeError, ValueError):
        return None


def _to_abbr(name) -> str | None:
    if not isinstance(name, str):
        return None
    return SBR_NICKNAME_TO_ABBR.get(name.strip().lower())


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw SBR JSON as-is (no filtering). For inspection."""
    path = path or (config.SBR_HISTORY_RAW_DIR / "nfl_archive_10Y.json")
    df = pd.read_json(path)
    return df


def load_lines(*, validate: bool = True) -> pd.DataFrame:
    """Load, normalize, and validate the SBR archive to a clean per-game frame.

    Returns one row per clean game with: ``season, gameday, home_abbr,
    away_abbr, home_score, away_score, home_open_spread, home_close_spread,
    open_over_under, close_over_under, spread_movement, total_movement,
    home_covered_close``. Rows with garbage spreads/totals or unrecognized
    teams are dropped when ``validate=True``.
    """
    df = load_raw()
    df["gameday"] = df["date"].apply(_parse_date)
    df["home_abbr"] = df["home_team"].apply(_to_abbr)
    df["away_abbr"] = df["away_team"].apply(_to_abbr)
    for c in ("home_final", "away_final"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["home_score"] = df["home_final"]
    df["away_score"] = df["away_final"]

    # Line movement: the favorite's number getting more negative (a bigger
    # spread to cover) = the market backed the favorite; the total moving up =
    # over money. Sign convention: SBR home_open_spread < 0 means home favored.
    df["spread_movement"] = df["home_close_spread"] - df["home_open_spread"]
    df["total_movement"] = df["close_over_under"] - df["open_over_under"]

    # Cover label against the CLOSING line: home covered iff its margin beat the
    # closing spread. (home_close_spread < 0 => home favored by |spread|.)
    # home covers when home_margin > -home_close_spread, i.e. margin + spread > 0.
    df["margin"] = df["home_score"] - df["away_score"]
    df["home_covered_close"] = ((df["margin"] + df["home_close_spread"]) > 0).astype("Int64")
    # And against the OPENING line (did the side that beat the open also beat close?).
    df["home_covered_open"] = ((df["margin"] + df["home_open_spread"]) > 0).astype("Int64")

    if validate:
        valid_spread = df["home_close_spread"].between(SPREAD_MIN, SPREAD_MAX)
        valid_total = df["open_over_under"].between(TOTAL_MIN, TOTAL_MAX)
        valid_teams = df["home_abbr"].notna() & df["away_abbr"].notna()
        valid_scores = df["home_score"].notna() & df["away_score"].notna()
        df = df[valid_spread & valid_total & valid_teams & valid_scores].copy()

    keep = [
        "season", "gameday", "home_abbr", "away_abbr",
        "home_score", "away_score", "margin",
        "home_open_spread", "home_close_spread", "spread_movement",
        "open_over_under", "close_over_under", "total_movement",
        "home_covered_open", "home_covered_close",
        "home_close_ml", "away_close_ml",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def join_game_ids(lines: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach nflverse ``game_id`` by matching on date + teams.

    nflverse game_id format is ``{season}_{week}_{AWAY}_{HOME}``. SBR has no
    week, so we match on gameday + (home_abbr, away_abbr) to pull the game_id.
    Games that don't match (data errors, edge cases) are dropped with a count.
    """
    sched = schedules[["game_id", "season", "gameday", "home_team", "away_team"]].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"]).dt.strftime("%Y-%m-%d")
    out = lines.merge(
        sched.rename(columns={"home_team": "home_abbr", "away_team": "away_abbr"}),
        on=["season", "gameday", "home_abbr", "away_abbr"], how="left",
    )
    n_unmatched = out["game_id"].isna().sum()
    if n_unmatched:
        print(f"[sbr] {n_unmatched}/{len(out)} games did not match a nflverse game_id")
    return out.dropna(subset=["game_id"]).reset_index(drop=True)
