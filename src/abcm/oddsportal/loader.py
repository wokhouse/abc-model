"""Wrap OddsHarvester to scrape NFL opening+closing odds with caching/resume.

OddsHarvester (``oddsharvester``, Playwright-based) fetches OddsPortal match
pages and extracts per-bookmaker odds with an ``opening_odds`` timestamp + the
full ``odds_history`` movement series. We wrap it to:

* Collect match links for an NFL season once (cached).
* Scrape each match's moneyline + the spread/total lines that nflverse reports
  for that game (the spread/total markets are per-line on OddsPortal —
  ``asian_handicap_-4_5``, ``over_under_47_5`` — so we must know the line first;
  we take it from nflverse ``spread_line`` / ``total_line``).
* Cache each match's raw JSON to ``data/raw/oddsportal/<season>/<game>.json`` so
  a Cloudflare timeout mid-run doesn't lose progress — re-running resumes.

The cache key is the OddsPortal match link, which encodes the two teams. We
mirror links to game_ids via the team abbrs + date.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from .. import config
from . import teams as teams_mod

SPORT = "american-football"
LEAGUE = "nfl"


def _season_slug(season_start: int) -> str:
    """OddsPortal season format: '2024-2025' for the season starting 2024."""
    return f"{season_start}-{season_start + 1}"


def collect_match_links(season_start: int, *, max_pages: int | None = None) -> list[dict]:
    """Collect OddsPortal NFL match links for one season.

    Returns a list of ``{match_link, home_team, away_team, home_abbr,
    away_abbr}``. Cached to ``links_<season>.json``; re-running returns the cache
    unless ``force=True``. Team abbrs are resolved here so callers can match to
    game_ids without re-parsing the URL slug.
    """
    season_dir = config.ODDSPORTAL_RAW_DIR / str(season_start)
    season_dir.mkdir(parents=True, exist_ok=True)
    cache = season_dir / f"links_{season_start}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        return rows

    cmd = [
        "oddsharvester", "historic",
        "-s", SPORT, "-l", LEAGUE, "--season", _season_slug(season_start),
        "--links-only", "--headless", "-o", str(cache),
    ]
    if max_pages:
        cmd += ["--max-pages", str(max_pages)]
    _run(cmd)
    raw = json.loads(cache.read_text()) if cache.exists() else []
    rows = []
    for r in raw:
        link = r.get("match_link") if isinstance(r, dict) else r
        home, away = _teams_from_link(link)
        rows.append({
            "match_link": link,
            "home_team": home, "away_team": away,
            "home_abbr": teams_mod.oddsportal_to_abbr(home) if home else None,
            "away_abbr": teams_mod.oddsportal_to_abbr(away) if away else None,
        })
    cache.write_text(json.dumps(rows, indent=2))
    return rows


def _teams_from_link(link: str) -> tuple[str | None, str | None]:
    """Parse the two team slug-names out of an OddsPortal h2h match URL.

    URL shape: ``.../h2h/<away-slug>/<home-slug>/#...`` (away listed first per the
    spike: ``kansas-city-chiefs/philadelphia-eagles`` with Eagles as home/venue).
    Slugs look like ``kansas-city-chiefs-vienoNr7`` (trailing ID hash); we strip
    the hash and title-case the name.
    """
    m = re.search(r"/h2h/([^/]+)/([^/]+)/", link or "")
    if not m:
        return None, None
    away_slug, home_slug = m.group(1), m.group(2)
    return _slug_to_name(away_slug), _slug_to_name(home_slug)


def _slug_to_name(slug: str) -> str:
    """``kansas-city-chiefs-vienoNr7`` -> ``Kansas City Chiefs``.

    Strips the trailing OddsPortal ID hash (8 alphanumerics after the last ``-``).
    """
    parts = slug.split("-")
    # Drop trailing ID hash: a short token of letters+digits at the end.
    if len(parts) > 1 and re.fullmatch(r"[a-zA-Z0-9]{6,}", parts[-1]):
        parts = parts[:-1]
    return " ".join(p.capitalize() for p in parts)


def scrape_match(
    match_link: str,
    *,
    spread_line: float | None = None,
    total_line: float | None = None,
    target_bookmaker: str | None = None,
    request_delay: float = 1.0,
) -> dict | None:
    """Scrape one match's odds (moneyline + requested spread/total lines).

    Returns the raw OddsHarvester record (a dict) or ``None`` on failure. The
    ``spread_line`` / ``total_line`` are the nflverse-reported lines for this
    game — OddsPortal's spread/total markets are per-line, so we request the
    specific ``asian_handicap_<line>`` / ``over_under_<line>`` markets.

    Set ``target_bookmaker`` (e.g. ``"Pinnacle"``) to filter to one book; default
    returns all books and the normalizer picks the sharpest available.
    """
    markets = ["1x2"]
    if spread_line is not None:
        markets.append(_handicap_slug(spread_line))
    if total_line is not None:
        markets.append(_total_slug(total_line))
    cmd = [
        "oddsharvester", "historic",
        "-s", SPORT, "-l", LEAGUE, "--season", "current",
        "-m", ",".join(markets), "--odds-history", "--headless",
        "--match-link", match_link,
        "--request-delay", str(request_delay),
    ]
    if target_bookmaker:
        cmd += ["--target-bookmaker", target_bookmaker]
    tmp = Path(config.ODDSPORTAL_RAW_DIR) / "_tmp_match.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["-o", str(tmp)]
    try:
        _run(cmd)
    except RuntimeError:
        return None
    if not tmp.exists():
        return None
    records = json.loads(tmp.read_text())
    tmp.unlink(missing_ok=True)
    return records[0] if records else None


def _handicap_slug(line: float) -> str:
    """``-4.5`` -> ``asian_handicap_-4_5`` (OddsPortal market slug convention)."""
    s = f"{line:+}".replace("+", "") if line != 0 else "0"
    s = s.replace(".", "_").replace("-", "-")  # keep the sign
    # OddsPortal uses no leading + ; -4.5 -> asian_handicap_-4_5
    if line > 0:
        s = f"+{line}".replace(".", "_")
    elif line < 0:
        s = f"{line}".replace(".", "_")
    else:
        s = "0"
    return f"asian_handicap_{s}"


def _total_slug(line: float) -> str:
    """``47.5`` -> ``over_under_47_5``."""
    return f"over_under_{str(line).replace('.', '_')}"


def _run(cmd: list[str], retries: int = 2) -> None:
    """Run an OddsHarvester CLI command, retrying on transient failure."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, timeout=180,
            )
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))  # backoff before retry
    raise RuntimeError(f"OddsHarvester command failed after {retries+1} attempts: {last_err}")
