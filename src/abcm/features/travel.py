"""Travel distance and timezone shift per game.

For each scheduled game the away team travels to the home team's city. We
compute:

* ``travel_distance_km`` — great-circle distance between the away team's home
  city and the home team's home city (haversine).
* ``timezone_shift_h`` — timezone difference from the away team's home city to
  the game site (the home team's city), positive = away travels east.

Team coordinates/timezones come from a bundled reference table
(``data/team_locations.csv``) rather than an external geocoding service — NFL
home cities are stable, so a hand-verified static table avoids a network
dependency and auth/rate-limit concerns.

Neutral-site games (London, Mexico City) are detected from the schedule's
``location`` field and flagged: travel is measured from the away team's home to
the *home* team's home city as usual (a reasonable proxy when the true venue
isn't in the table), but ``neutral_site`` is set so downstream code can discount
the home-field advantage. A full neutral-site venue table could refine this.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Bundled reference: team_abbr -> (lat, lon, tz_offset_hours).
_TEAM_LOCATIONS_PATH = Path(__file__).resolve().parent / "data" / "team_locations.csv"


def load_team_locations() -> pd.DataFrame:
    """Load the bundled team location reference table."""
    return pd.read_csv(_TEAM_LOCATIONS_PATH)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (haversine)."""
    import math

    r = 6371.0088  # mean Earth radius in km
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(a))


def _location_lookup() -> dict[str, dict[str, float]]:
    """team_abbr -> {lat, lon, tz_offset_hours}."""
    df = load_team_locations()
    return {
        str(r["team_abbr"]): {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "tz_offset_hours": float(r["tz_offset_hours"]),
        }
        for _, r in df.iterrows()
    }


def game_travel_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Attach travel_distance_km, timezone_shift_h, neutral_site to each game.

    Returns the schedules frame with the three new columns. Games with an
    unknown team (e.g. an abbr not in the reference table) get null travel
    features.
    """
    loc = _location_lookup()
    distances: list[float | None] = []
    shifts: list[float | None] = []
    neutral: list[bool] = []

    for _, row in schedules.iterrows():
        away = str(row.get("away_team"))
        home = str(row.get("home_team"))
        is_neutral = str(row.get("location", "")).strip().lower() == "neutral"
        neutral.append(is_neutral)

        away_loc = loc.get(away)
        home_loc = loc.get(home)
        if away_loc is None or home_loc is None:
            distances.append(None)
            shifts.append(None)
            continue
        distances.append(
            _haversine_km(
                away_loc["lat"], away_loc["lon"], home_loc["lat"], home_loc["lon"]
            )
        )
        # Positive shift = away team travels east (later local time at game site).
        shifts.append(home_loc["tz_offset_hours"] - away_loc["tz_offset_hours"])

    out = schedules.copy()
    out["travel_distance_km"] = distances
    out["timezone_shift_h"] = shifts
    out["neutral_site"] = neutral
    return out
