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
NFLVERSE_HISTORY_DIR = RAW_DIR / "nflverse_history"
KAGGLE_RAW_DIR = RAW_DIR / "kaggle"
ELO_PROCESSED_DIR = PROCESSED_DIR / "elo"
FEATURE_PROCESSED_DIR = PROCESSED_DIR / "features"
MODELS_PROCESSED_DIR = PROCESSED_DIR / "models"


def ensure_dirs() -> None:
    """Create the data directory tree if missing."""
    for d in (
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        POLYMARKET_RAW_DIR,
        POLYMARKET_PRICES_DIR,
        NFLVERSE_RAW_DIR,
        NFLVERSE_HISTORY_DIR,
        KAGGLE_RAW_DIR,
        ELO_PROCESSED_DIR,
        FEATURE_PROCESSED_DIR,
        MODELS_PROCESSED_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


# --- Target seasons --------------------------------------------------------

def _parse_years(raw: str | None) -> list[int]:
    if not raw:
        return [2023, 2024, 2025]
    return [int(y) for y in raw.split(",") if y.strip()]


YEARS: list[int] = _parse_years(os.environ.get("ABC_YEARS"))


def _parse_years_range(raw: str | None, default_lore: int, default_hi: int) -> list[int]:
    """Parse a year list, defaulting to a contiguous inclusive range."""
    if not raw:
        return list(range(default_lore, default_hi + 1))
    return [int(y) for y in raw.split(",") if y.strip()]


# Deep NFL history for the two-stage transfer-learning recipe: pre-train a
# game-outcome model on decades of nflverse data (pbp/features back to 1999),
# then apply it to the thin Polymarket-labeled 2024+ set. This is a much larger
# pull than YEARS (the fast Polymarket-aligned set) and writes to separate files
# so the modeling target data isn't disturbed.
NFLVERSE_HISTORY_YEARS: list[int] = _parse_years_range(
    os.environ.get("ABC_NFLVERSE_HISTORY_YEARS"), 1999, 2025
)


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


# --- Kaggle (sportsbook betting lines) -------------------------------------

# Credentials are read from the ABC_-prefixed env vars (matching the rest of the
# codebase) and bridged to the kaggle package's expected KAGGLE_* names at use.
KAGGLE_USERNAME = os.environ.get("ABC_KAGGLE_USERNAME", "")
KAGGLE_KEY = os.environ.get("ABC_KAGGLE_KEY", "")
# Toby Crabtree's "NFL Scores and Betting Data" — game results + sportsbook
# lines since 1979. The main file is spreadspoke_scores.csv.
KAGGLE_DATASET = os.environ.get(
    "ABC_KAGGLE_DATASET", "tobycrabtree/nfl-scores-and-betting-data"
)


# --- Elo ratings -----------------------------------------------------------

# We compute FiveThirtyEight-style team Elo ourselves (the canonical 538 dataset
# is frozen at the 2021 season and can't cover our 2024-2026 markets). Elo needs
# years of history to stabilize, so we pull schedules for a long lookback window
# (schedules are tiny; pbp stays at the current YEARS).
#
# The lookback starts at 1999 (the nflverse schedule floor) so that the
# Stage 1 outcome pre-training window (1999-2023) is fully covered by ``elo_diff``.
# Caveat: Elo is cold-started at the season-start prior (1505) and is noisy for
# the first ~3 seasons (1999-2001) until it stabilizes. That is acceptable: those
# early rows are a minority of the training set, and their rolling efficiency
# features are also null for the same reason, so the model already discounts them.
def _parse_years_list(raw: str | None, default: str) -> list[int]:
    if not raw:
        return [int(y) for y in default.split(",")]
    return [int(y) for y in raw.split(",") if y.strip()]


ELO_HISTORY_YEARS: list[int] = _parse_years_list(
    os.environ.get("ABC_ELO_HISTORY_YEARS"), "1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025"
)

# 538 team-Elo constants (https://fivethirtyeight.com/features/how-our-nfl-predictions-work/).
ELO_MEAN = float(os.environ.get("ABC_ELO_MEAN", "1505"))  # season-start prior
ELO_REGRESSION = float(os.environ.get("ABC_ELO_REGRESSION", "0.333"))  # toward mean
ELO_HFA = float(os.environ.get("ABC_ELO_HFA", "55"))  # home-field advantage (Elo pts)
ELO_K_BASE = float(os.environ.get("ABC_ELO_K_BASE", "20"))  # K-factor multiplier


# --- Modeling --------------------------------------------------------------

# Pinned random_state for every estimator and CV split, so a model run is a pure
# function of its inputs (per the cross-cutting reproducibility requirement).
STAGE1_RANDOM_STATE: int = int(os.environ.get("ABC_STAGE1_RANDOM_STATE", "42"))

