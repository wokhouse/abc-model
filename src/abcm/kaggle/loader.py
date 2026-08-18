"""Load and normalize the spreadspoke betting-lines dataset.

The dataset (``spreadspoke_scores.csv``) uses full franchise names
(``"New England Patriots"``) while nflverse uses 2-3 letter abbreviations
(``"NE"``). This module downloads the CSV via the kaggle package and projects it
to a normalized frame keyed on ``(schedule_season, schedule_week, home_team_abbr)``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .. import config

# Full franchise name (as it appears in spreadspoke_scores.csv) -> nflverse abbr.
# Covers historical names too (San Diego -> LAC, Oakland -> LV, etc.).
TEAM_NAME_TO_ABBR: dict[str, str] = {
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
    # Historical names (relocations/rebrands).
    "san diego chargers": "LAC",
    "oakland raiders": "LV",
    "st. louis rams": "LA",
    "washington redskins": "WAS",
    "washington football team": "WAS",
}


def _configure_credentials() -> None:
    """Bridge ABC_KAGGLE_* env vars to the kaggle package's expected names.

    The kaggle package reads ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` from the
    environment (or ``~/.kaggle/kaggle.json``). We set them from our ABC_-prefixed
    config so the rest of the codebase stays consistent. Raises a clear error if
    credentials are absent.
    """
    if config.KAGGLE_USERNAME and config.KAGGLE_KEY:
        os.environ.setdefault("KAGGLE_USERNAME", config.KAGGLE_USERNAME)
        os.environ.setdefault("KAGGLE_KEY", config.KAGGLE_KEY)
        return
    raise RuntimeError(
        "Kaggle credentials not configured. Set ABC_KAGGLE_USERNAME and "
        "ABC_KAGGLE_KEY in your .env (see .env.example)."
    )


def download_dataset(dest_dir: Path | None = None) -> Path:
    """Download the betting-lines dataset CSVs to ``dest_dir`` via the kaggle API.

    Returns the destination directory. Idempotent: skips the download if the
    expected CSV is already present.
    """
    _configure_credentials()
    import kaggle  # imported lazily so the package is optional at import time

    dest_dir = dest_dir or config.KAGGLE_RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = dest_dir / "spreadspoke_scores.csv"
    if expected.exists():
        return dest_dir
    kaggle.api.dataset_download_files(
        config.KAGGLE_DATASET, path=str(dest_dir), unzip=True
    )
    return dest_dir


def load_betting_lines(years: list[int] | None = None) -> pd.DataFrame:
    """Load + normalize spreadspoke_scores.csv to nflverse keys.

    Columns kept: ``schedule_season, schedule_week, schedule_date,
    home_team_abbr, away_team_abbr, team_favorite_abbr, spread_favorite,
    over_under_line, stadium_neutral``. Filtered to ``years`` if given.
    """
    dest_dir = download_dataset()
    csv_path = dest_dir / "spreadspoke_scores.csv"
    df = pd.read_csv(csv_path)

    if years:
        df = df[df["schedule_season"].isin(years)]

    # schedule_week mixes ints ("1") and playoff labels ("Wildcard", "Division").
    # Regular-season weeks join to nflverse on an int week; coerce and drop the
    # non-numeric playoff rows (nflverse uses a separate game_type for those).
    df = df.assign(
        schedule_week=df["schedule_week"].apply(
            lambda w: int(w) if str(w).strip().isdigit() else None
        )
    ).dropna(subset=["schedule_week"])
    df["schedule_week"] = df["schedule_week"].astype(int)

    def to_abbr(name: str) -> str | None:
        if not isinstance(name, str):
            return None
        return TEAM_NAME_TO_ABBR.get(name.strip().lower())

    df = df.assign(
        home_team_abbr=df["team_home"].apply(to_abbr),
        away_team_abbr=df["team_away"].apply(to_abbr),
        team_favorite_abbr=df["team_favorite_id"],
    )
    # Normalize the favorite id (already an abbr like 'NE', but sometimes blank).
    # Keep only rows where both teams resolved.
    df = df.dropna(subset=["home_team_abbr", "away_team_abbr"])

    keep = [
        "schedule_season", "schedule_week", "schedule_date",
        "home_team_abbr", "away_team_abbr",
        "team_favorite_abbr", "spread_favorite", "over_under_line",
        "stadium_neutral",
    ]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)
