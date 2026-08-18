"""OddsPortal team-name -> nflverse abbreviation normalization.

OddsPortal uses full franchise names ("Kansas City Chiefs"); nflverse uses
3-letter abbrs ("KC"). The map mirrors :data:`abcm.kaggle.loader.TEAM_NAME_TO_ABBR`
since both sources use full names, but OddsPortal occasionally uses a slightly
different surface form (e.g. "Los Angeles Rams" vs the Rams' history). Unknown
names return ``None`` so callers can flag a join failure rather than silently
mismatch.
"""
from __future__ import annotations

# OddsPortal "home_team"/"away_team" values are full franchise names, lowercased
# for matching. Covers current franchises; historical relocations collapse to the
# current abbr (OddsPortal itself maps old franchises to current nicknames).
ODDSPORTAL_NAME_TO_ABBR: dict[str, str] = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}


def oddsportal_to_abbr(name: str) -> str | None:
    """Normalize an OddsPortal team name to a nflverse abbr.

    Returns ``None`` for an unrecognized name (caller decides how to handle the
    miss — typically a flagged join failure, never a silent default).
    """
    if not isinstance(name, str):
        return None
    return ODDSPORTAL_NAME_TO_ABBR.get(name.strip().lower())
