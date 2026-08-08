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
`game_features.parquet` (including the 2024–2025 labeled games). Write predictions
to `data/processed/features/game_outcome_probs.parquet`
(`game_id, home_win_prob, pred_method`).

### Honest OOS check
LOSO within the history: train 1999–2020 → test 2021–2023. Report Brier score
and calibration. A reasonable NFL outcome model gets ~63–68% accuracy and Brier
~0.23. If the outcome model is no better than Elo's `elo_home_prob` alone, the
Stage 2 transfer has no edge — stop and reconsider features before proceeding.

> **Implemented result (gate PASSED, thin edge).** The LOSO gate (train 1999–2020
> → test 2021–2023) and a 3-fold rolling-expanding LOSO for variance were built
> in `src/abcm/model/outcome.py` + `calibrate.py`. Numbers:
>
> | Model | Acc | Brier | (multi-fold Brier mean±std) |
> |---|---|---|---|
> | Elo baseline | 0.601 | 0.228 | 0.221 ± 0.005 |
> | **LogReg** | **0.643** | **0.226** | **0.219 ± 0.005** |
> | XGBoost | 0.610 | 0.232 | 0.224 ± 0.006 |
>
> Key findings, stated plainly:
> - **Only LogReg beats Elo on Brier** — in all 3 multi-folds, with *lower*
>   accuracy variance. XGBoost does **not** beat Elo on Brier in any fold, exactly
>   as the research predicted ("at small n, penalized logistic may beat XGBoost").
> - The final scorer is therefore the **gate winner (LogReg)**, not XGBoost by
>   default. `train_final_and_predict(final_model="gate_winner", gate=...)` picks
>   the lower-Brier model automatically.
> - **The edge is small** (~0.002–0.003 Brier). Stage 2 should not assume a large
>   `market_delta` edge exists; it's a real-but-thin signal.
> - **Null handling:** ~46% of games (mostly pre-tracking-era deep history) lack
>   EPA/CPOE features and are scored by an Elo-only logistic fallback rather than
>   being dropped — no fabricated values. In 2024–2025 (the Stage 2 window) 65% of
>   games use the final model, 35% the Elo fallback.
> - `monotone_constraints` on `elo_diff`/`pass_epa_diff`/`rest_diff` are wired up
>   (the plan's `epa_diff` → `pass_epa_diff`, the only EPA-diff column).
>
> Outputs: `data/processed/features/game_outcome_probs.parquet`
> (`game_id, home_win_prob, pred_method`, 7,276 rows, 0 nulls) and
> `data/processed/models/outcome_model.joblib`. The summary dict carries
> `gate_passed: True` so Stage 2 can hard-fail if this is ever re-run and
> regresses. `market_delta = home_win_prob - open_price` is computable on the
> labeled rows (mean −0.016, std 0.315 — sensibly centered).

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

> **Implemented result — HONEST NEGATIVE: no tradeable inefficiency found.**
>
> Stage 1b (cover-prob model, `src/abcm/model/cover.py`) was added to supply a
> spread `model_prob` — Stage 1 only predicts home *win* prob, but spreads are
> about *covering*. Trained on 6,423 REG games 1999-2023 (cover label
> `(home_score-away_score) >= spread_line`; nflverse sign convention is the
> *opposite* of Vegas — **positive** `spread_line` = home favored). The cover
> gate barely passes (LogReg brier 0.249 vs naive base 0.250; ~51% accuracy —
> beating the closing spread is near-coin-flip, as expected).
>
> Stage 2 (`src/abcm/model/inefficiency.py`) built the edge frame on 1,090 clean
> labeled rows (254 moneyline, 836 spread), resolved the YES side
> (`model_prob_yes = home_prob if YES==home else 1-home_prob`), and evaluated
> pooled vs per-type vs spread-first stacking on the binary target
> (`was betting the model's side profitable?`), with bootstrap 95% CIs:
>
> | Config (LOSO 2024→2025) | n_test | Brier [95% CI] | Base Brier | Verdict |
> |---|---|---|---|---|
> | pooled | 849 | 0.276 [0.266, 0.286] | 0.249 | ❌ worse than base |
> | per-type moneyline | 93 | 0.279 [0.242, 0.318] | 0.245 | ❌ worse (CI includes base) |
> | per-type spread | 756 | 0.251 [0.245, 0.257] | 0.249 | ⚠️ indistinguishable from base |
> | stacking | 93 | 0.280 [0.243, 0.319] | — | ❌ no lift over per-type |
> | rolling-origin pooled | 872 | 0.270 [0.261, 0.280] | 0.249 | ❌ worse than base |
>
> **Every configuration's Brier is at or above the base rate.** A naive "predict
> the base-rate probability" outperforms the model. Accuracy is ~0.49–0.54, no
> better than chance. The `edge` signal is biased negative (mean −0.026: the
> market systematically prices YES above the model), and the model cannot turn
> that into a profitable decision rule.
>
> **This is the honest answer to the project's core question, and it is
> consistent with the thin Stage 1 edge** (~0.002 Brier over Elo). A win-prob
> model that only barely beats Elo does not produce a `market_delta` large or
> consistent enough to identify tradeable Polymarket inefficiency at these sample
> sizes. Stage 3 (backtest/CLV) is unlikely to rescue this — CLV beats ROI as a
> validity signal, but there's no positive edge here to validate. The honest
> recommendation: do not deploy; revisit only if (a) Stage 1's edge widens
> (better features / more history), (b) the labeled set grows (the deferred
> 6,400 game-total markets, or 2026 data), or (c) a fundamentally different
> signal (e.g. line movement / QB news) is added.

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
- [x] Rolling features exclude the current game (shift-by-1 invariant, unit-tested
      — 0 mismatches across all 14,278 games in Stage 0)
- [x] Stage 1 outcome predictions for the labeled games come from a model trained
      **without** those games' labels — the final fit uses 1999–2023 only; the
      2024–2025 labeled games are scored, never trained on
- [x] OOF stacking predictions respect temporal folds (spread OOF feature for
      moneyline generated from in-window spread rows only; unit-tested)
- [ ] Calibration split is temporal (last 20% by date), not random
- [ ] No feature uses post-game data (injury backfills, retroactive stat corrections)
- [ ] The 2026 holdout is touched once, at the very end

### Reproducibility
- Pin `random_state` everywhere (XGBoost, sklearn, CV splits) — `config.STAGE1_RANDOM_STATE=42`
- Write all predictions/probas to parquet with the fold/seed in the filename
- The `game_features.parquet` build is deterministic given the raw data

### Suggested file layout
```
src/abcm/model/
    __init__.py
    outcome.py          # Stage 1: home-win-prob model train/predict
    cover.py            # Stage 1b: home-cover-prob model (spread model_prob)
    inefficiency.py     # Stage 2: edge model train/predict + LOSO/rolling-origin
    backtest.py         # Stage 3: backtest, CLV, Kelly (not yet built)
    calibrate.py        # Platt scaling + Brier/reliability helpers
scripts/
    build_features.py   # Stage 0
    build_elo.py        # Elo ratings (Stage 0 dependency)
    train_outcome.py    # Stage 1
    train_cover.py      # Stage 1b
    train_inefficiency.py  # Stage 2
    run_backtest.py     # Stage 3 (not yet built)
notebooks/
    02_backtest.ipynb   # Stage 3 analysis (not yet built)
data/processed/features/
    game_features.parquet
    game_outcome_probs.parquet
    game_cover_probs.parquet
    inefficiency_predictions.parquet
data/processed/models/
    outcome_model.joblib
    cover_model.joblib
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
