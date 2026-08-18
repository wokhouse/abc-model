"""Normalize Polymarket team-name labels to nflverse ``team_abbr`` codes.

Polymarket moneyline markets expose outcomes as team nicknames (e.g. ``"Texans"``,
``"49ers"``), while nflverse schedules use 2-3 letter abbreviations (``HOU``,
``SF``). This table bridges the two.

The Polymarket labels also carry noise we strip before lookup:
  * trailing whitespace (``"Bears "``)
  * spread/total suffixes (``"Chiefs -1.5"``, ``"Eagles +1.5"``)
  * "Over"/"Under" (totals markets — not team labels)
"""
from __future__ import annotations

import re

# Polymarket nickname -> nflverse team_abbr. Covers all 32 active clubs plus
# the nicknames Polymarket has used historically.
TEAM_NICKNAME_TO_ABBR: dict[str, str] = {
    "cardinals": "ARI",
    "falcons": "ATL",
    "ravens": "BAL",
    "bills": "BUF",
    "panthers": "CAR",
    "bears": "CHI",
    "bengals": "CIN",
    "browns": "CLE",
    "cowboys": "DAL",
    "broncos": "DEN",
    "lions": "DET",
    "packers": "GB",
    "texans": "HOU",
    "colts": "IND",
    "jaguars": "JAX",
    "chiefs": "KC",
    "raiders": "LV",
    "rams": "LA",
    "charger": "LAC",       # singular forms seen occasionally
    "chargers": "LAC",
    "dolphins": "MIA",
    "vikings": "MIN",
    "patriots": "NE",
    "saints": "NO",
    "giants": "NYG",
    "jets": "NYJ",
    "eagles": "PHI",
    "steelers": "PIT",
    "49ers": "SF",
    "seahawks": "SEA",
    "buccaneers": "TB",
    "titans": "TEN",
    "commanders": "WAS",
    # Historical Washington names (relocated/rebranded) — normalize to WAS.
    "washington": "WAS",
    "football team": "WAS",
    "redskins": "WAS",
    # Historical Raiders/Oilers aliases.
    "oilers": "TEN",
}

# Outcomes that are never a team (totals markets, generic labels).
_NON_TEAM_OUTCOMES = {"over", "under", "yes", "no", "field"}


def _clean_outcome(label: str) -> str:
    """Strip trailing whitespace and a trailing spread/total like '-1.5' or '+3'."""
    if not isinstance(label, str):
        return ""
    # Drop anything after a +/- number (e.g. "Chiefs -1.5" -> "Chiefs").
    label = re.split(r"\s+[-+]\d", label)[0]
    return label.strip().lower()


def nickname_to_abbr(label: str | None) -> str | None:
    """Map a Polymarket outcome label to an nflverse team_abbr.

    Returns ``None`` for non-team outcomes (Over/Under/Yes/No) or unknown names.

    Polymarket occasionally uses the abbreviation itself (``"LAC"``, ``"BUF"``)
    rather than the nickname, so an exact uppercase abbreviation is passed
    through after the nickname lookup fails.
    """
    if not label:
        return None
    cleaned = _clean_outcome(label)
    if cleaned in _NON_TEAM_OUTCOMES:
        return None
    abbr = TEAM_NICKNAME_TO_ABBR.get(cleaned)
    if abbr is not None:
        return abbr
    # Abbreviation passthrough: Polymarket sometimes uses "LAC"/"BUF"/"KC".
    upper = label.strip().upper()
    if upper in _ALL_ABBRS:
        return upper
    return None


def all_abbrs() -> set[str]:
    """Return the full set of known nflverse abbreviations."""
    return set(TEAM_NICKNAME_TO_ABBR.values())


_ALL_ABBRS = all_abbrs()
