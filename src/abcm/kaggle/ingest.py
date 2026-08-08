"""Ingest Kaggle betting-line data to a normalized parquet."""
from __future__ import annotations

from typing import Any

from .. import config, io
from . import loader


def ingest_kaggle(years: list[int] | None = None) -> dict[str, Any]:
    """Download + normalize the spreadspoke betting lines; write to parquet.

    Returns a summary dict with row count and output path.
    """
    years = years or config.YEARS
    config.ensure_dirs()
    summary: dict[str, Any] = {"years": years}

    print(f"[kaggle] downloading {config.KAGGLE_DATASET}...")
    df = loader.load_betting_lines(years=years)
    out_path = config.KAGGLE_RAW_DIR / "betting_lines.parquet"
    io.write_parquet(df, out_path)
    summary["betting_lines_rows"] = len(df)
    summary["path"] = str(out_path)
    print(f"[kaggle] betting_lines: {len(df)} rows -> {out_path}")
    return summary
