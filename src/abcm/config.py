"""Central configuration for the abc-model data pipeline.

All paths, target years, and API behavior live here. The rest of the codebase
reads these values and never touches ``os.environ`` directly, so behavior can be
reconfigured from one place (or via a ``.env`` file).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # python-dotenv is a soft dependency at import time; env vars still work.
    pass


# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

POLYMARKET_RAW_DIR = RAW_DIR / "polymarket"
POLYMARKET_PRICES_DIR = POLYMARKET_RAW_DIR / "prices"
NFLVERSE_RAW_DIR = RAW_DIR / "nflverse"


def ensure_dirs() -> None:
    """Create the data directory tree if missing."""
    for d in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        POLYMARKET_RAW_DIR,
        POLYMARKET_PRICES_DIR,
        NFLVERSE_RAW_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# --- Target seasons --------------------------------------------------------

def _parse_years(raw: str | None) -> list[int]:
    if not raw:
        return [2023, 2024, 2025]
    return [int(y) for y in raw.split(",") if y.strip()]


YEARS: list[int] = _parse_years(os.environ.get("ABC_YEARS"))


# --- Polymarket API --------------------------------------------------------

GAMMA_BASE_URL = os.environ.get(
    "ABC_POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"
)
CLOB_BASE_URL = os.environ.get(
    "ABC_POLYMARKET_CLOB_URL", "https://clob.polymarket.com"
)
# The data-api serves full trade history for RESOLVED markets (the CLOB
# /prices-history endpoint returns empty once a market closes). Keyed by
# conditionId. Undocumented but official; no auth required.
DATA_API_BASE_URL = os.environ.get(
    "ABC_POLYMARKET_DATA_URL", "https://data-api.polymarket.com"
)

NFL_TAG_SLUG = "nfl"

# Page size for event pagination on the Gamma API.
EVENTS_PAGE_SIZE = 100
# Page size for trade-history pagination on the data-api (max 500).
TRADES_PAGE_SIZE = 500


# --- HTTP behavior ---------------------------------------------------------

HTTP_MAX_RETRIES = int(os.environ.get("ABC_HTTP_MAX_RETRIES", "5"))
HTTP_TIMEOUT = float(os.environ.get("ABC_HTTP_TIMEOUT", "30"))
PRICE_CONCURRENCY = int(os.environ.get("ABC_PRICE_CONCURRENCY", "4"))


# --- Price snapshot --------------------------------------------------------

# When snapshotting a market's price, take the last trade at or before this many
# hours before the scheduled game time. Falls back to the last available price.
PRICE_SNAPSHOT_HOURS_BEFORE_GAME = 1
