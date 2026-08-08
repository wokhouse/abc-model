"""Thin read/write helpers for parquet and JSON.

Centralizing these keeps serialization consistent across the pipeline and lets
callers stay focused on data shape rather than file paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to parquet, creating parent dirs. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
