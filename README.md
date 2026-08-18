# abc-model

NFL Polymarket prediction model. This repository currently implements the
**data gathering + feature phase**: acquiring resolved NFL markets and price
history from Polymarket, NFL play-by-play / schedules / rosters from nflverse,
sportsbook betting lines from Kaggle, computing derived features (Elo ratings,
travel/timezone), and reconciling everything into a single joinable table.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # install runtime + dev deps
cp .env.example .env       # optional; defaults work out of the box
```

## Pipeline

The pipeline gathers raw data, computes derived features, then reconciles
everything into one joinable table:

```bash
# --- Raw data ---
# 1. Resolved NFL events + markets + per-token price history
uv run python -m scripts.ingest_polymarket

# 2. nflverse play-by-play, schedules, rosters, team stats
uv run python -m scripts.ingest_nfl

# 3. Sportsbook betting lines (Kaggle; needs ABC_KAGGLE_* in .env)
uv run python -m scripts.ingest_kaggle

# --- Derived features ---
# 4. Elo ratings (computed from nflverse scores; 538 dataset is frozen at 2021)
uv run python -m scripts.build_elo

# --- Reconcile ---
# 5. Match moneyline + spread markets to games; join all features
uv run python -m scripts.reconcile
```

Outputs land under `data/` (gitignored):

```
data/raw/polymarket/   events.parquet, markets.parquet, prices/<condition_id>/
data/raw/nflverse/     pbp.parquet, schedules.parquet, rosters.parquet, team_stats.parquet
data/raw/kaggle/       betting_lines.parquet   ← sportsbook spread/total lines
data/processed/elo/    elo_ratings.parquet     ← per-game team-Elo (computed)
data/processed/        aligned.parquet         ← 1 row per market, joined to its NFL game
```

`aligned.parquet` carries, per market: the re-classified market type, the matched
game_id, pre-game price snapshot + open→close delta, parsed spread
(favored/margin), Elo ratings + win prob, travel distance + timezone shift, and
the sportsbook spread/total where available.

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

This phase covers data gathering + feature materialization. It does **not**
include modeling, calibration, or walk-forward validation. Those follow once the
validation notebook confirms we have enough usable labeled data. Note the
canonical FiveThirtyEight Elo dataset is frozen at the 2021 season, so Elo is
computed forward from nflverse scores here.
