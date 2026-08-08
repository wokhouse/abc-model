"""Wrappers around ``nflreadpy`` that configure caching and return pandas.

``nflreadpy`` returns polars DataFrames. We convert to pandas here so the rest
of the pipeline works in a single frame type. Caching is configured from env
(see ``config.py``) so repeated ingestion runs don't re-download.
"""
from __future__ import annotations

import os

import pandas as pd

try:
    from nflreadpy import config as nfl_config
    import nflreadpy as nfl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nflreadpy is required. Install it with: uv add nflreadpy"
    ) from exc


_CACHE_CONFIGURED = False


def _ensure_cache_config() -> None:
    """Apply nflreadpy cache settings from env once per process."""
    global _CACHE_CONFIGURED
    if _CACHE_CONFIGURED:
        return
    kwargs: dict[str, object] = {}
    mode = os.environ.get("NFLREADPY_CACHE")
    if mode:
        kwargs["cache_mode"] = mode
    cache_dir = os.environ.get("NFLREADPY_CACHE_DIR")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    duration = os.environ.get("NFLREADPY_CACHE_DURATION")
    if duration and duration.isdigit():
        kwargs["cache_duration"] = int(duration)
    if kwargs:
        nfl_config.update_config(**kwargs)
    _CACHE_CONFIGURED = True


def _to_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a polars or pandas DataFrame to pandas."""
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return df


def load_pbp(years: list[int]) -> pd.DataFrame:
    _ensure_cache_config()
    return _to_pandas(nfl.load_pbp(years))


def load_schedules(years: list[int]) -> pd.DataFrame:
    _ensure_cache_config()
    return _to_pandas(nfl.load_schedules(years))


def load_rosters(years: list[int]) -> pd.DataFrame:
    _ensure_cache_config()
    return _to_pandas(nfl.load_rosters(years))


def load_team_stats(years: list[int]) -> pd.DataFrame:
    _ensure_cache_config()
    return _to_pandas(nfl.load_team_stats(years, summary_level="week"))
