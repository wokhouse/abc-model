# abc-model

NFL Polymarket prediction model. This repository currently implements the
**data gathering phase**: acquiring resolved NFL markets and price history from
Polymarket, NFL play-by-play / schedules / rosters from nflverse, and
reconciling the two into a single joinable table.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # install runtime + dev deps
cp .env.example .env       # optional; defaults work out of the box
```

## Pipeline

Three ingestion steps, run in order:

```bash
# 1. Resolved NFL events + markets + per-token price history
uv run python -m scripts.ingest_polymarket

# 2. nflverse play-by-play, schedules, rosters, team stats
uv run python -m scripts.ingest_nfl

# 3. Match markets to NFL games; emit processed/aligned.parquet
uv run python -m scripts.reconcile
```

Outputs land under `data/` (gitignored):

```
data/raw/polymarket/   events.parquet, markets.parquet, prices/<token_id>.parquet
data/raw/nflverse/     pbp.parquet, schedules.parquet, rosters.parquet, team_stats.parquet
data/processed/        aligned.parquet   ← 1 row per market, joined to its NFL game
```

## Validation

The EDA notebook quantifies data coverage before any modeling:

```bash
uv run jupyter notebook notebooks/01_data_validation.ipynb
```

## Tests

```bash
uv run pytest
```

## Scope notes

This phase does **not** include feature engineering, modeling, or calibration.
Those follow once the validation notebook confirms we have enough usable labeled
data (the main open risk is that single-game NFL moneyline markets on Polymarket
are sparse before the 2024 season).
