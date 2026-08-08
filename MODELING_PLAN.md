# Modeling Plan

Implementation guide for the remaining modeling stages, from the current
data-gathering endpoint to a backtestable Polymarket inefficiency model.

**Modeling goal:** given a game's pre-game features and the Polymarket market's
*opening trade price*, predict whether the market is inefficient — i.e., whether
a model-derived true probability disagrees enough with the opening price to
recommend a trade.

This plan follows the research synthesis (small-n, temporally-imbalanced,
walk-forward-required) and the two-stage transfer recipe it recommended. Each
stage lists inputs, outputs, files, and concrete implementation notes.

---

## Current state (data phase complete)

`data/processed/aligned.parquet` — 1 row per matched market (2,154 total):

| Group | Columns | Coverage |
|---|---|---|
| Identity/join | `game_id, market_id, condition_id, season, week, market_type, nfl_home_team, nfl_away_team` | 100% |
| Label | `resolved_yes_price` (0/1 Yes outcome) | 100% |
| Market prices | `open_price, open_time, snapshot_price, price_delta, n_candles, snapshot_method` | 78% (1,674/2,154) |
| Spread fields | `spread_favored_abbr, spread_opponent_abbr, spread_margin` | 87% (spread markets) |
| Elo | `elo_home_pre, elo_away_pre, elo_home_prob` | 100% |
| Travel | `travel_distance_km, timezone_shift_h, neutral_site` | 100% |
| Sportsbook | `sportsbook_favorite, sportsbook_spread, sportsbook_ou` | 96% (regular season) |

Usable labeled rows (clean cutoff snapshot + label): **moneyline 261, spread
836, game_total 1**. Distribution: 2024 ≈ 210, 2025 ≈ 1,944; **no 2026 data has
landed yet.** (The earlier "moneyline 258 / spread 633 / total 891" figures did
not reconcile against `aligned.parquet` — there is no `total` market_type and
the totals don't sum; counts above are the actual cutoff-clean + labeled rows.)

Deep history at `data/raw/nflverse_history/` (1999–2025): 7,276 games, 1.28M pbp
plays, 14,531 weekly team_stats. **Note:** `team_stats.parquet` carries
EPA/CPOE/turnovers/first-downs back to 1999 but has **no success-rate or
play-count columns** — success rate and yards per play are aggregated from
`pbp.parquet` (`success`, `yards_gained`, `play`, `down`). CPOE and air_yards
are only populated from 2006 (tracking-data origin).

---

## Stage 0 — Feature engineering (rolling efficiency from deep history)

**Goal:** build a per-game feature frame keyed on `game_id`, derived from nflverse
*strictly prior* to each game (no look-ahead). These features feed both the
Stage 1 outcome model (on 1999–2023 history) and the Stage 2 inefficiency model
(on the 261 moneyline / 836 spread labeled rows).

**Files:** `src/abcm/features/efficiency.py`, `scripts/build_features.py`

### Rolling team-efficiency features (the prompt's headline predictors)
From `team_stats.parquet` (weekly, per team) plus `pbp.parquet`, compute
exponentially-weighted or simple rolling means over a team's **prior N games**
(N ∈ {5, 10, 16} or EWMA with α≈0.15). Per team, per game:

- `off_pass_epa_roll`, `def_pass_epa_allowed_roll` — passing EPA generated/allowed
- `off_rush_epa_roll`, `def_rush_epa_allowed_roll`
- `off_success_rate_roll`, `def_success_rate_allowed_roll` — **sourced from
  `pbp.parquet`** (`success` flag over pass/run scrimmage plays); team_stats has
  no success/play-count columns
- `cpoe_roll` — QB completion % over expected (the prompt's QB-efficiency signal)
- `off_yards_per_play_roll`, `def_ypp_allowed_roll` — **sourced from `pbp.parquet`**
  (`yards_gained` per pass/run play)
- `turnover_diff_roll` — takeaways minus giveaways (from `fumbles_lost_total`,
  `def_interceptions`)

> **Implemented as:** `home_*_roll{5,10,16}` / `home_*_ewma` and the `_allowed`
> mirrors plus home-minus-away `*_diff` (window 10) and `*_diff_ewma`. The
> defense-allowed features come from joining each game's opponent's rolling
> offense. `ELO_HISTORY_YEARS` was extended to start at **1999** so `elo_diff`
> covers the full Stage 1 training window (cold-start noise in 1999–2001 is
> acceptable since those rows' rolling features are also null).

**Critical leakage control:** the rolling window for a given `(team, game)` must
include **only that team's games before this game's date**. Sort team_stats by
`season, week`, group by `team`, shift by 1 (exclude the current game), then
rolling-mean. The first few games of each season (and 1999) will be null —
that's correct, not a bug. (Verified: 0 shift-by-1 mismatches across all
14,278 games.)

### Differential features
Once both teams' rolling stats exist for a game, compute the **home-minus-away**
differential:
- `epa_diff = home_off_pass_epa_roll - away_def_pass_epa_allowed_roll` (and the
  symmetric `away_off - home_def`)
- `elo_diff = elo_home_pre - elo_away_pre` (already have this)
- `cpoe_diff`, `success_rate_diff`, `ypp_diff`, `to_diff`

### Situational features (already in schedules — just carry through)
- `rest_diff = home_rest - away_rest`
- `div_game`, `roof` (dome/outdoor → one-hot), `neutral_site`
- `travel_distance_km`, `timezone_shift_h` (away team's burden)
- `temp`, `wind` (outdoor games only; null for dome — impute or flag)

### Output
`data/processed/features/game_features.parquet` — one row per game, keyed on
`game_id`, with all rolling + differential + situational features. Cover the full
1999–2025 range (so Stage 1 can train on history). The 2024–2025 rows join to the
labeled markets via `game_id`. (Built: 7,276 games × 71 columns; `elo_diff` and
all engineered differentials at 100% coverage in 2024–2025; `temp`/`wind` 65%
due to dome games, flagged via `is_outdoor`.)

**Validation hook:** assert that rolling features for game G never include
game G's own stats (unit test: shift-by-1 invariant). Assert no feature has
>50% nulls in the 2024–2025 modeling window.

---

## Stage 1 — Pre-train the game-outcome model on deep history

**Goal:** learn "true win probability" from decades of game outcomes, using only
pre-game features. This model's predictions become the `market_delta` feature in
Stage 2.

**Files:** `src/abcm/model/outcome.py`, `scripts/train_outcome.py`

### Training set
All regular-season games 1999–2023 from `game_features.parquet` (~6,000+ games).
Target: `home_win = home_score > away_score` (binary). Hold out 2024–2025 from
this stage's training (those are the Stage 2 labeled games; using them here to
train the outcome model is fine, but for an honest Stage 1 estimate, train
1999–2023 and evaluate on a 2021–2023 LOSO fold).

### Model
Start with **penalized logistic regression** as the baseline (the research
flagged that at small n, this may beat XGBoost — and it generalizes across eras
more stably). Then fit XGBoost as a comparison:

```
LogisticRegression(C=0.1, penalty='l2', class_weight='balanced')
XGBClassifier(max_depth=2, learning_rate=0.03, n_estimators=500,
              subsample=0.6, colsample_bytree=0.5, min_child_weight=10,
              alpha=0.5, reg_lambda=2, objective='binary:logistic',
              eval_metric='logloss')
```

### Monotonic constraints (the research's recommended domain priors)
On the XGBoost model, enforce:
- `elo_diff` → increasing (higher Elo gap → higher home win prob)
- `epa_diff` → increasing
- `rest_diff` → increasing (more rest = better)

### Output
A trained outcome model + its predicted `home_win_prob` for every game in
`game_features.parquet` (including the 2024–2026 labeled games). Write predictions
to `data/processed/features/game_outcome_probs.parquet` (`game_id, home_win_prob`).

### Honest OOS check
LOSO within the history: train 1999–2020 → test 2021–2023. Report Brier score
and calibration. A reasonable NFL outcome model gets ~63–68% accuracy and Brier
~0.23. If the outcome model is no better than Elo's `elo_home_prob` alone, the
Stage 2 transfer has no edge — stop and reconsider features before proceeding.

---

## Stage 2 — Build the inefficiency (market-edge) model

**Goal:** predict, at the moment of the Polymarket opening trade, whether the
market is inefficient enough to trade. This is the core deliverable.

**Files:** `src/abcm/model/inefficiency.py`, `scripts/train_inefficiency.py`,
`src/abcm/model/backtest.py`

### The target, precisely
The modeling target is **edge**, not raw outcome. For each market:

```
model_prob  = stage1_home_win_prob            (for moneyline)
            | stage1 cover probability         (for spread — derived)
market_prob = open_price                       (the opening trade)
edge        = model_prob - market_prob         (signed)
```

Two formulation options (test both):
1. **Regression on edge** — predict the signed `edge`; trade when `|edge| > threshold`.
2. **Binary classification** — label = "did the market resolve in the direction
   of the edge?" i.e. `label = (resolved_yes_price == 1) == (edge > 0)`. This is
   the "was betting the model's side profitable?" question.

### Feature set (the labeled rows: 261 moneyline / 836 spread)
Join to `game_features.parquet` on `game_id`, then add:
- `market_prob` (open_price), `price_delta`, `n_candles` (liquidity signal)
- `market_delta = stage1_prob - open_price` (**the headline inefficiency signal**)
- `market_type` (moneyline/spread — one-hot, for the pooled model)
- `sportsbook_spread, sportsbook_ou` (closing lines — the research's circularity
  warning applies only if the target is "beat the market using the market's own
  close"; for an outcome/edge target these are legitimate features)

### Pooling decision (test, don't assume)
The research recommends spread-first stacking:
1. Train the **spread model** on 836 rows first (better powered).
2. Generate its out-of-fold predictions within the CV folds.
3. Add the spread OOF prediction as a feature in the **moneyline model** (261 rows).
Also fit a **pooled model** (moneyline + spread, `market_type` as feature) as a
baseline. Compare via LOSO Brier — don't assume pooling wins.

### Validation: Leave-One-Season-Out (LOSO)
This is the only defensible scheme at our scale (research: CPCV infeasible, random
k-fold is leakage). Two folds:
- Fold A: train 2024 (≈210 clean rows) → test 2025 (≈1,944)
- Fold B: train 2024+2025 → test 2026 (the 2026 holdout is **not yet in
  `aligned.parquet`** — it lands as the season plays out; live forward-testing
  the 2026 season is the real out-of-sample gate, per Stage 3).

Supplement with a **purged rolling-origin** within 2024+2025 (block by week, 1-week
embargo, min 150 training rows) to get a variance estimate on the OOS log-loss.
Report both LOSO and rolling-origin with confidence intervals.

**OOF generation for stacking:** the spread model's OOF predictions fed into the
moneyline model MUST be generated within the same temporal folds — never let a
future fold's labels leak into a spread prediction used to train the moneyline fold.

### Regularization (small-n, high-p)
At ~261–836 rows with ~20–30 features, overfitting is the dominant risk:
- `max_depth=2`, `learning_rate=0.01–0.05`, `n_estimators` via early stopping
- `min_child_weight=10`, `subsample=0.6`, `colsample_bytree=0.5`
- `alpha=0.5`, `reg_lambda=2` (L1/L2)
- Early stopping on **CV-averaged** log-loss (not a single tiny validation fold),
  `early_stopping_rounds=30–50` (the single-fold noise floor is high)
- Also fit penalized logistic regression as a baseline; at 261 moneyline rows it
  may win.

### Calibration
**Use Platt scaling (sigmoid), NOT isotonic regression.** This is a correction
to the original prompt: isotonic overfits below ~1,000–2,000 samples (primary
source: Niculescu-Mizil & Caruana 2005; sklearn docs). At our scale the
moneyline calibration set is ~50–100 rows — "no method is reliably trustworthy"
there; Platt is least-bad. Calibrate on a temporally-held-out 20% split, separate
from the hyperparameter-tuning fold. Evaluate with Brier score + reliability
diagrams (5–6 bins, not 10 — fewer bins at low n).

---

## Stage 3 — Backtest & bet-sizing

**Goal:** turn model probabilities into a backtestable, sized trading record
with CLV and Kelly analysis. This is what determines if the model is real.

**Files:** `src/abcm/model/backtest.py`, `notebooks/02_backtest.ipynb`

### Backtest protocol
For each market in the LOSO test folds (only ever predictions the model didn't
train on):
1. Compute `edge = calibrated_model_prob - open_price`.
2. Apply decision rule: trade when `|edge| > threshold` (sweep threshold 0.03–0.15).
3. De-vig: `market_prob` is raw Polymarket; for CLV vs. the close, strip the vig
   from `snapshot_price` before comparing.
4. Record the bet side, size, and outcome (`resolved_yes_price`).

### Metrics to report (honest about variance)
- **ROI** by market type and edge bucket — but ROI is noisy at n≈300–600.
- **Closing Line Value (CLV):** `calibrated_prob - de-vigged snapshot_price`.
  The research says consistent positive CLV is a stronger validity signal than
  ROI. This is the primary metric.
- **Brier score** (proper scoring rule) on the test fold — report with the LOSO
  fold sizes explicit.
- **Reliability diagrams** (5–6 bins).
- **95% CI** on all metrics via bootstrap within the test fold; state the fold
  sizes and that absolute-performance uncertainty is high.

### Bet sizing
- **Kelly fraction:** `f* = (b·p - q) / b` where `b` = decimal odds, `p` = model
  prob, `q = 1-p`. Use **fractional Kelly (25%)** to reduce variance at small n.
- Cap position sizes; the liquidity features (`volume, liquidity, n_candles`)
  flag markets too thin to trade at size — a 5% edge on a 10-share market isn't
  actionable.

### What "done" looks like
The model is defensible if, on the untouched 2026 holdout:
- CLV is consistently positive (sign test, not just mean), AND
- ROI is positive across edge buckets (not just aggregate), AND
- Calibration reliability holds within the wide CIs.

If only the 2025 fold passes and 2026 fails, that's a likely overfit signal —
the research is explicit that 2-fold LOSO can't distinguish skill from luck, so
forward-testing the 2026 season live is the real test.

---

## Cross-cutting concerns

### Data leakage checklist (apply at every stage)
- [ ] Rolling features exclude the current game (shift-by-1 invariant, unit-tested)
- [ ] Stage 1 outcome predictions for the labeled games come from a model trained
      **without** those games' labels (the labeled games can be in Stage 1's
      feature rows, but their outcomes must not train the Stage 1 model if you
      want a clean transfer — or use nested CV)
- [ ] OOF stacking predictions respect temporal folds
- [ ] Calibration split is temporal (last 20% by date), not random
- [ ] No feature uses post-game data (injury backfills, retroactive stat corrections)
- [ ] The 2026 holdout is touched once, at the very end

### Reproducibility
- Pin `random_state` everywhere (XGBoost, sklearn, CV splits)
- Write all predictions/probas to parquet with the fold/seed in the filename
- The `game_features.parquet` build is deterministic given the raw data

### Suggested file layout
```
src/abcm/model/
    __init__.py
    outcome.py          # Stage 1: outcome model train/predict
    inefficiency.py     # Stage 2: edge model train/predict
    backtest.py         # Stage 3: backtest, CLV, Kelly
    calibrate.py        # Platt scaling + reliability diagrams
scripts/
    build_features.py   # Stage 0
    train_outcome.py    # Stage 1
    train_inefficiency.py  # Stage 2
    run_backtest.py     # Stage 3
notebooks/
    02_backtest.ipynb   # Stage 3 analysis
data/processed/features/
    game_features.parquet
    game_outcome_probs.parquet
data/processed/models/
    outcome_model.joblib
    inefficiency_model.joblib
```

---

## Stage dependencies (order)

```
Stage 0 (features) ──► Stage 1 (outcome pre-train) ──► Stage 2 (inefficiency) ──► Stage 3 (backtest)
                                              │
                                              └── market_delta = stage1_prob - open_price feeds Stage 2
```

Stage 0 must complete before anything. Stage 1's model quality gates whether
Stage 2 is worth building (if Stage 1 can't beat Elo alone, there's no transfer
edge to exploit). Stage 2 and Stage 3 iterate together — backtest results drive
feature/threshold tuning, but the 2026 holdout is locked until the final run.

## Out of scope for now
- QB-adjusted Elo (deferred stretch goal; team Elo is in place)
- Game-total market matching (6,400 markets classified but not matched; could
  expand the labeled set ~7× but harder matching problem)
- Live forward-testing infra (2026 holdout simulates this; true live deployment
  is a follow-on)
