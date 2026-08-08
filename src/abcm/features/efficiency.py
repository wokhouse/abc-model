"""Game-level efficiency features from deep nflverse history.

Stage 0 of the modeling plan: build a per-game feature frame keyed on
``game_id``, covering 1999-2025, that feeds both the Stage 1 outcome
pre-training (on history) and the Stage 2 inefficiency model (on the 2024-2025
labeled rows).

Inputs (all from ``data/raw/nflverse_history``):

* ``schedules.parquet`` — game metadata: home/away teams, scores, rest, roof,
  temp, wind, division flag.
* ``team_stats.parquet`` — weekly per-team boxscore stats. A row for
  ``(team=X, opponent=Y)`` holds **X's offense** (its ``passing_epa`` is what X
  generated) **plus X's defense counters** (``def_interceptions``, ``def_sacks``,
  ``fumble_recovery_opp``). Note: team_stats has no success-rate or play-count
  columns — those come from play-by-play.
* ``pbp.parquet`` — play-by-play; the only source for ``success`` (per-play
  success flag) and ``yards_gained`` per play, so success rate and yards per
  play are aggregated here.
* ``elo_ratings.parquet`` — pre-game team Elo (computed by ``features.elo``).

**Leakage control (the core invariant).** Every rolling window for a
``(team, game)`` must include *only that team's games before this game's date*.
We enforce it by sorting each team's games by ``(season, week)`` and applying
``groupby(team).shift(1)`` *before* the rolling mean, so the current game's own
boxscore can never enter its own feature vector. The first game(s) of each
team's history and each season are null — that is correct, not a bug.

Output: ``data/processed/features/game_features.parquet`` — one row per game,
keyed on ``game_id``, with rolling, differential, situational, and Elo columns.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import config, io

# Rolling windows for team-efficiency means. The plan calls for N in {5, 10, 16}
# plus an EWMA (alpha ~0.15). window=10 is treated as the primary horizon for the
# differential features (a balance of recency and stability at ~2.5 games/week).
ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 16)
PRIMARY_WINDOW: int = 10
EWMA_ALPHA: float = 0.15

# Seasons whose rolling features must be well-covered (>50% non-null) or the build
# fails loudly rather than silently producing a degraded model input.
MODELING_SEASONS: tuple[int, ...] = (2024, 2025)


# ---------------------------------------------------------------------------
# Per-(team, game) efficiency from team_stats (offense + defense counters)
# ---------------------------------------------------------------------------

def _per_team_game_efficiency(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Collapse team_stats into one row per (team, game_id) of game efficiency.

    Each output row holds that team's *offensive* production for the game plus a
    few *defensive* counters (takeaways). Total yards combine pass + rush yards;
    turnovers lost = interceptions thrown + fumbles lost; takeaways = INTs caught
    + opponent fumbles recovered.

    Columns kept: ``season, week, game_id, team, opponent_team`` and the derived
    efficiency measures.
    """
    df = team_stats.copy()
    df["total_yards"] = (
        df["passing_yards"].fillna(0) + df["rushing_yards"].fillna(0)
    )
    df["pass_fd"] = df["passing_first_downs"].fillna(0)
    df["rush_fd"] = df["rushing_first_downs"].fillna(0)
    df["first_downs"] = df["pass_fd"] + df["rush_fd"]
    df["to_lost"] = df["passing_interceptions"].fillna(0) + df["fumbles_lost_total"].fillna(0)
    # Takeaways this team's defense generated: interceptions + opponent fumbles
    # recovered (fumble_recovery_opp is the opponent's fumble recovered by us).
    df["takeaways"] = df["def_interceptions"].fillna(0) + df["fumble_recovery_opp"].fillna(0)

    cols = [
        "season", "week", "game_id", "team", "opponent_team",
        "pass_epa", "rush_epa", "pass_cpoe",
        "total_yards", "first_downs",
        "pass_attempts", "carries", "targets",
        "to_lost", "takeaways",
    ]
    out = pd.DataFrame({
        "season": df["season"],
        "week": df["week"],
        "game_id": df["game_id"],
        "team": df["team"],
        "opponent_team": df["opponent_team"],
        "pass_epa": df["passing_epa"],
        "rush_epa": df["rushing_epa"],
        "pass_cpoe": df["passing_cpoe"],
        "total_yards": df["total_yards"],
        "first_downs": df["first_downs"],
        "pass_attempts": df["attempts"],
        "carries": df["carries"],
        "targets": df["targets"],
        "to_lost": df["to_lost"],
        "takeaways": df["takeaways"],
    })
    return out[cols]


# ---------------------------------------------------------------------------
# Per-(team, game) success rate / yards per play from play-by-play
# ---------------------------------------------------------------------------

def _per_team_game_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate success rate and yards per play per (game_id, posteam) from pbp.

    Only genuine offensive scrimmage plays (``play_type`` in {pass, run}) count,
    so kneel-downs, penalties, and specials are excluded from the denominators.
    Returns ``[game_id, team, success_rate, yards_per_play, n_plays]`` keyed on
    ``(game_id, team)``. play-by-play only reaches back to 1999 and its success
    flag is reliable throughout, but the ``cpoe``/air-yards tracking fields begin
    in 2006 — that does not affect success_rate/ypp here.
    """
    pbp = pbp.copy()
    scrim = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    scrim["success"] = scrim["success"].astype("float")
    scrim["yards_gained"] = scrim["yards_gained"].astype("float")
    agg = (
        scrim.groupby(["game_id", "posteam"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            yards_per_play=("yards_gained", "mean"),
            n_plays=("play", "count"),
        )
        .rename(columns={"posteam": "team"})
    )
    return agg[["game_id", "team", "success_rate", "yards_per_play", "n_plays"]]


# ---------------------------------------------------------------------------
# Rolling team-efficiency features (leakage-controlled)
# ---------------------------------------------------------------------------

# The per-game measures we roll forward. ``off_*`` are this team's own offense;
# the defensive "allowed" mirrors are produced by re-keying the same rolling
# table on ``opponent_team`` (see :func:`_defense_allowed`).
_OFFENSE_MEASURES: tuple[str, ...] = (
    "pass_epa", "rush_epa", "pass_cpoe", "success_rate", "yards_per_play",
    "to_lost",
)


def _roll_team(team_game: pd.DataFrame, *, windows=ROLLING_WINDOWS, ewma_alpha=EWMA_ALPHA) -> pd.DataFrame:
    """Add rolling offense features per team, shifted by 1 to drop the current game.

    Sort key is ``(team, season, week)``. For each measure ``m`` and each window
    ``w`` we compute, *within each team's chronological sequence*,
    ``m.shift(1).rolling(w, min_periods=1).mean()`` — the mean of that team's
    prior games only (so the current game's boxscore can never enter its own
    feature). The EWMA variant uses ``shift(1).ewm(alpha).mean()``. Both shift and
    rolling are applied per team group so a game never rolls in another team's rows.
    """
    df = team_game.sort_values(["team", "season", "week"]).reset_index(drop=True)

    measures = [m for m in _OFFENSE_MEASURES if m in df.columns]
    new_cols: dict[str, pd.Series] = {}
    for _, g in df.groupby("team", sort=False):
        for m in measures:
            shifted = g[m].astype(float).shift(1)
            for w in windows:
                new_cols.setdefault(f"{m}_roll{w}", pd.Series(index=df.index, dtype=float))
                new_cols[f"{m}_roll{w}"].loc[g.index] = shifted.rolling(w, min_periods=1).mean().to_numpy()
            new_cols.setdefault(f"{m}_ewma", pd.Series(index=df.index, dtype=float))
            new_cols[f"{m}_ewma"].loc[g.index] = shifted.ewm(alpha=ewma_alpha, adjust=False).mean().to_numpy()

    for name, col in new_cols.items():
        df[name] = col
    return df


def _defense_allowed(team_roll: pd.DataFrame) -> pd.DataFrame:
    """Attach defense-allowed rolling features alongside each team's own offense.

    A team's *defensive pass EPA allowed* in a game is its opponent's *offensive
    pass EPA* that game. So for each ``(team, game)`` row we look up the rolling
    offensive stat of the row whose ``team`` equals this row's ``opponent_team``
    and whose ``game_id`` matches, and carry it as ``*_allowed``. The team's own
    offense rolling columns are kept (not replaced): each row holds both the
    team's offense and the offense its defense faced.
    """
    off_cols = [c for c in team_roll.columns if any(c.startswith(m + "_") for m in _OFFENSE_MEASURES)]
    # The opponent's offense for THIS game lives in the row where team == opponent_team.
    opp = team_roll[["game_id", "team"] + off_cols].rename(
        columns={"team": "opponent_team", **{c: c + "_allowed" for c in off_cols}}
    )
    return team_roll.merge(opp, on=["game_id", "opponent_team"], how="left")


def _rolling_team_game(
    team_stats: pd.DataFrame,
    pbp: pd.DataFrame | None,
    *,
    windows=ROLLING_WINDOWS,
    ewma_alpha=EWMA_ALPHA,
) -> pd.DataFrame:
    """Full per-(team, game) frame with offense + defense-allowed rolling features.

    Joins the team_stats efficiency with pbp success/ypp, rolls everything with
    shift(1), then attaches the opponent's rolling offense as defense-allowed.
    """
    eff = _per_team_game_efficiency(team_stats)
    if pbp is not None and len(pbp):
        pbp_agg = _per_team_game_from_pbp(pbp)
        eff = eff.merge(pbp_agg, on=["game_id", "team"], how="left")
    else:
        # Float NaN (not pd.NA) so the columns stay numeric and roll cleanly.
        eff["success_rate"] = np.nan
        eff["yards_per_play"] = np.nan
        eff["n_plays"] = pd.NA
    rolled = _roll_team(eff, windows=windows, ewma_alpha=ewma_alpha)
    return _defense_allowed(rolled)


# ---------------------------------------------------------------------------
# Game-level assembly: home/away pull + differentials
# ---------------------------------------------------------------------------

def _game_efficiency_features(
    schedules: pd.DataFrame,
    team_roll: pd.DataFrame,
) -> pd.DataFrame:
    """Pull each game's home and away rolling features and form differentials.

    Uses the schedules' authoritative ``home_team``/``away_team`` columns (the
    ``game_id`` string lists away_team first, so we never trust its order).
    Differentials are home-minus-away at the primary window; the EWMA differentials
    are added as a recency-weighted companion.
    """
    off = team_roll.copy()

    def _pull(side: str) -> pd.DataFrame:
        """Rename rolling cols to ``{side}_*`` keyed on (game_id, that team)."""
        roll_cols = [
            c for c in off.columns
            if c.endswith(f"_roll{PRIMARY_WINDOW}") or c.endswith("_ewma")
            or c.endswith(f"_roll{PRIMARY_WINDOW}_allowed")
        ]
        rename = {c: f"{side}_{c}" for c in roll_cols}
        sub = off[["game_id", "team"] + roll_cols].rename(columns={"team": side, **rename})
        return sub

    base = schedules[["game_id", "home_team", "away_team"]].copy()
    feat = (
        base
        .merge(_pull("home"), left_on=["game_id", "home_team"], right_on=["game_id", "home"], how="left")
        .drop(columns=["home"])
        .merge(_pull("away"), left_on=["game_id", "away_team"], right_on=["game_id", "away"], how="left")
        .drop(columns=["away"])
    )

    # Differentials at the primary window and the EWMA companion.
    for m in _OFFENSE_MEASURES:
        h = f"home_{m}_roll{PRIMARY_WINDOW}"
        a = f"away_{m}_roll{PRIMARY_WINDOW}"
        if h in feat.columns and a in feat.columns:
            feat[f"{m}_diff"] = feat[h] - feat[a]
        he = f"home_{m}_ewma"
        ae = f"away_{m}_ewma"
        if he in feat.columns and ae in feat.columns:
            feat[f"{m}_diff_ewma"] = feat[he] - feat[ae]
    return feat


def situational_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """Carry through rest/roof/weather/division/neutral-site context per game.

    No imputation: dome/outdoor games have null ``temp``/``wind`` and we keep them
    null plus an ``is_outdoor`` flag so the model stage can decide how to handle it.
    """
    df = schedules.copy()
    out = pd.DataFrame({"game_id": df["game_id"]})
    out["rest_diff"] = df["home_rest"] - df["away_rest"]
    out["div_game"] = df["div_game"]
    roof = df["roof"].fillna("unknown").str.lower()
    for level in ("dome", "outdoors", "closed", "open"):
        out[f"roof_{level}"] = (roof == level).astype("int8")
    out["neutral_site"] = (
        df["location"].fillna("").str.strip().str.lower() == "neutral"
    ).astype("int8")
    out["is_outdoor"] = (roof == "outdoors").astype("int8")
    out["temp"] = df["temp"]
    out["wind"] = df["wind"]
    return out


def add_elo(game_features: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    """Left-join pre-game Elo ratings and add ``elo_diff``.

    Only the *pre-game* columns are joined (``_post`` ratings are post-game and
    would leak the outcome). Where Elo is missing (pre-2006 in the old build; the
    cold-start rows), ``elo_*`` are null.
    """
    cols = elo[["game_id", "elo_home_pre", "elo_away_pre", "elo_home_prob"]].copy()
    out = game_features.merge(cols, on="game_id", how="left")
    out["elo_diff"] = out["elo_home_pre"] - out["elo_away_pre"]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_features(
    schedules: pd.DataFrame,
    team_stats: pd.DataFrame,
    pbp: pd.DataFrame | None,
    elo: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full game-level feature frame.

    Order: rolling team efficiency (shift(1)-controlled) -> home/away pull +
    differentials -> situational context -> Elo join. Returns one row per game
    keyed on ``game_id``, sorted by ``(season, week)``.
    """
    team_roll = _rolling_team_game(team_stats, pbp)
    eff = _game_efficiency_features(schedules, team_roll)
    sit = situational_features(schedules)
    out = eff.merge(sit, on="game_id", how="left")
    out = add_elo(out, elo)
    # Attach the modeling label context (scores/season/week/date) for downstream
    # convenience without re-joining schedules later. Also carry the closing
    # consensus spread/total lines: the spread is a strong pre-game team-strength
    # summary (the market's efficient estimate) and a high-value Stage 1 feature
    # (see outcome.FEATURES). nflverse sign convention: positive spread_line =
    # home favored by that many.
    ctx_cols = ["game_id", "season", "week", "gameday", "home_team", "away_team",
                "home_score", "away_score"]
    for extra in ("spread_line", "total_line"):
        if extra in schedules.columns:
            ctx_cols.append(extra)
    ctx = schedules[ctx_cols].copy()
    ctx["home_win"] = (ctx["home_score"] > ctx["away_score"]).astype("Int64")
    # gameday may already be present on eff only via merge; keep schedules as source.
    out = out.merge(ctx, on=["game_id", "home_team", "away_team"], how="left", suffixes=("", "_ctx"))
    if "season_ctx" in out.columns:
        out.drop(columns=[c for c in out.columns if c.endswith("_ctx")], inplace=True)
    out = out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    return out


def build_and_save() -> dict[str, Any]:
    """Read raw history, build game features, write to processed/features/.

    Mirrors ``features.elo.build_and_save``: ensures dirs, reads the four inputs,
    writes ``game_features.parquet`` via :func:`io.write_parquet`, prints
    ``[efficiency] ...`` progress, and returns a summary dict. Fails loudly if any
    engineered feature is >50% null in the 2024-2025 modeling window.
    """
    config.ensure_dirs()

    print("[efficiency] loading raw history...")
    schedules = io.read_parquet(config.NFLVERSE_HISTORY_DIR / "schedules.parquet")
    team_stats = io.read_parquet(config.NFLVERSE_HISTORY_DIR / "team_stats.parquet")
    elo = io.read_parquet(config.ELO_PROCESSED_DIR / "elo_ratings.parquet")
    pbp_path = config.NFLVERSE_HISTORY_DIR / "pbp.parquet"
    pbp = io.read_parquet(pbp_path) if pbp_path.exists() else None

    print(
        f"[efficiency] schedules={len(schedules)} team_stats={len(team_stats)} "
        f"elo={len(elo)} pbp={'yes' if pbp is not None and len(pbp) else 'no'}"
    )

    features = build_features(schedules, team_stats, pbp, elo)
    out_path = config.FEATURE_PROCESSED_DIR / "game_features.parquet"
    io.write_parquet(features, out_path)

    # Coverage check: no engineered feature >50% null in the modeling window.
    engineered = [c for c in features.columns if c not in {
        "game_id", "home_team", "away_team", "season", "week", "gameday",
        "home_score", "away_score", "home_win",
    }]
    model_rows = features[features["season"].isin(MODELING_SEASONS)]
    high_null = {
        c: float(model_rows[c].isna().mean())
        for c in engineered
        if model_rows[c].isna().mean() > 0.5
    }
    if high_null:
        raise ValueError(
            "Engineered features with >50% null in the modeling window "
            f"(2024-2025): {high_null}"
        )

    summary: dict[str, Any] = {
        "n_games": len(features),
        "season_range": f"{int(features['season'].min())}-{int(features['season'].max())}",
        "n_columns": features.shape[1],
        "elo_coverage": float(features["elo_diff"].notna().mean()),
        "pbp_success_coverage": float(
            features["success_rate_diff"].notna().mean()
            if "success_rate_diff" in features.columns else float("nan")
        ),
        "path": str(out_path),
    }
    print(f"[efficiency] wrote {len(features)} games x {features.shape[1]} cols -> {out_path}")
    print(f"[efficiency] elo_diff coverage {summary['elo_coverage']:.1%}, "
          f"success_rate_diff coverage {summary['pbp_success_coverage']:.1%}")
    return summary
