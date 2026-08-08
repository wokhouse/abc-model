"""FiveThirtyEight-style NFL team Elo ratings.

We compute team-Elo ourselves from nflverse game scores because the canonical
538 dataset is frozen at the 2021 season and cannot cover our 2024-2026 markets.
The algorithm follows FiveThirtyEight's published methodology:

* Mean 1505; each team regresses 1/3 toward the mean at the end of every season.
* Home-field advantage is worth ~55 Elo points (reduced for neutral games).
* Expected probability:
      ``p_home = 1 / (1 + 10^((elo_away - elo_home - HFA) / 400))``
* K-factor (margin-of-victory multiplier):
      ``K = K_BASE * log10(margin + 1) * (2.2 / ((winner_elo_diff * 0.001) + 0.05))``
  where ``winner_elo_diff`` is the winner's pre-game Elo minus the loser's
  (this penalizes a favorite winning by little and rewards an underdog upset).
* The winner gains ``K * (1 - p_winner)``; the loser drops the same amount.

Output is one row per team per game with pre/post ratings and the home team's
pre-game Elo win probability, joinable to ``aligned.parquet`` on ``game_id``.

Sources:
  https://fivethirtyeight.com/features/introducing-nfl-elo-ratings/
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .. import config, io


def _expected_prob(elo_home: float, elo_away: float, hfa: float) -> float:
    """Home team's pre-game Elo win probability (logistic on the rating gap).

    The exponent is clamped to the float-safe range so an extreme rating gap
    (possible early in the history when some teams have few games) can't overflow.
    """
    exponent = (elo_away - elo_home - hfa) / 400.0
    exponent = max(min(exponent, 300.0), -300.0)
    return 1.0 / (1.0 + 10.0 ** exponent)


def _k_factor(
    margin: float,
    winner_elo_diff: float,
    *,
    k_base: float = config.ELO_K_BASE,
) -> float:
    """538 margin-of-victory K multiplier.

    ``margin`` is the winner's final score margin (>= 1). ``winner_elo_diff`` is
    the winner's pre-game Elo minus the loser's (can be negative for an upset).
    Formula (per FiveThirtyEight):

        MoV = ln(|margin| + 1) * (2.2 / ((winner_elo - loser_elo) * 0.001 + 2.2))

    The ``+ 2.2`` constant (NOT 0.05) keeps the denominator large enough that
    K stays in a sane range; the 0.001 scales the Elo gap so a big favorite
    winning barely yields a small multiplier. The denominator is floored for
    numerical safety.
    """
    mov = max(abs(margin), 1.0)
    denom = max((winner_elo_diff * 0.001) + 2.2, 0.01)
    return k_base * math.log(mov + 1) * (2.2 / denom)


def compute_elo(
    schedules: pd.DataFrame,
    *,
    mean: float = config.ELO_MEAN,
    regression: float = config.ELO_REGRESSION,
    hfa: float = config.ELO_HFA,
    k_base: float = config.ELO_K_BASE,
) -> pd.DataFrame:
    """Iterate games in date order, producing per-team-game Elo ratings.

    ``schedules`` must have: ``game_id, season, gameday, away_team, home_team,
    away_score, home_score, location`` (regular + postseason games; preseason
    should be excluded — its score signal is noisy).

    Returns one row per game with: ``game_id, date, season, home_team,
    away_team, elo_home_pre, elo_away_pre, elo_home_post, elo_away_post,
    elo_home_prob``.
    """
    df = schedules.copy()
    df["date"] = pd.to_datetime(df["gameday"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "game_id"]).reset_index(drop=True)

    # Initialize every team that appears at the mean prior.
    ratings: dict[str, float] = {}
    last_season: int | None = None
    rows: list[dict[str, Any]] = []

    def _rating(team: str) -> float:
        if team not in ratings:
            ratings[team] = mean
        return ratings[team]

    def _is_neutral(loc: Any) -> bool:
        return isinstance(loc, str) and loc.strip().lower() == "neutral"

    for _, g in df.iterrows():
        season = int(g["season"])
        # End-of-season regression: pull every known team toward the mean once
        # when we cross into a new season.
        if last_season is not None and season != last_season:
            for team in ratings:
                ratings[team] += (mean - ratings[team]) * regression
        last_season = season

        home = g["home_team"]
        away = g["away_team"]
        home_score = g["home_score"]
        away_score = g["away_score"]

        if pd.isna(home_score) or pd.isna(away_score):
            continue  # unplayed game or missing score

        elo_home = _rating(home)
        elo_away = _rating(away)
        site_hfa = 0.0 if _is_neutral(g.get("location")) else hfa
        p_home = _expected_prob(elo_home, elo_away, site_hfa)

        # Margin of victory from the winner's perspective.
        if home_score >= away_score:
            winner, loser = home, away
            winner_elo, loser_elo = elo_home, elo_away
            margin = home_score - away_score
            p_winner = p_home
        else:
            winner, loser = away, home
            winner_elo, loser_elo = elo_away, elo_home
            margin = away_score - home_score
            p_winner = 1.0 - p_home

        winner_elo_diff = winner_elo - loser_elo
        k = _k_factor(margin, winner_elo_diff, k_base=k_base)
        shift = k * (1.0 - p_winner)

        ratings[winner] = winner_elo + shift
        ratings[loser] = loser_elo - shift

        rows.append({
            "game_id": g["game_id"],
            "date": g["date"],
            "season": season,
            "home_team": home,
            "away_team": away,
            "elo_home_pre": elo_home,
            "elo_away_pre": elo_away,
            "elo_home_post": ratings[home],
            "elo_away_post": ratings[away],
            "elo_home_prob": p_home,
        })

    return pd.DataFrame(rows)


def build_and_save(years: list[int] | None = None) -> dict[str, Any]:
    """Pull long-history schedules, compute Elo, write to processed/elo/.

    Elo needs many seasons of history to stabilize; ``config.ELO_HISTORY_YEARS``
    controls the lookback. Only ratings for ``years`` (the modeling seasons) are
    required downstream, but the full history is computed to anchor them.
    """
    import nflreadpy as nfl  # lazy import: nflreadpy is optional at module load

    history = years or config.ELO_HISTORY_YEARS
    config.ensure_dirs()

    # Cache the long-history schedules so re-runs don't re-download.
    hist_path = config.ELO_PROCESSED_DIR / "schedules_history.parquet"
    if hist_path.exists():
        all_sched = io.read_parquet(hist_path)
    else:
        print(f"[elo] loading schedules {min(history)}-{max(history)}...")
        sched = nfl.load_schedules(list(history))
        all_sched = sched.to_pandas() if hasattr(sched, "to_pandas") else sched
        # Exclude preseason (noisy score signal); keep REG + postseason.
        all_sched = all_sched[all_sched["game_type"] != "PRE"]
        io.write_parquet(all_sched, hist_path)
        print(f"[elo] cached {len(all_sched)} historical games -> {hist_path}")

    print(f"[elo] computing Elo over {len(all_sched)} games...")
    ratings = compute_elo(all_sched)
    out_path = config.ELO_PROCESSED_DIR / "elo_ratings.parquet"
    io.write_parquet(ratings, out_path)

    # Sanity: did the pre-game Elo favorite actually win? (outcome prediction)
    target = ratings[ratings["season"].isin(config.YEARS)]
    summary: dict[str, Any] = {
        "history_games": len(all_sched),
        "ratings_rows": len(ratings),
        "path": str(out_path),
    }
    if len(target):
        # Join the score back to determine the actual winner.
        scored = all_sched[["game_id", "home_score", "away_score"]].dropna()
        tgt = target.merge(scored, on="game_id", how="inner")
        if len(tgt):
            home_won = tgt["home_score"] >= tgt["away_score"]
            predicted_home = tgt["elo_home_prob"] > 0.5
            correct = (home_won == predicted_home).sum()
            summary["target_seasons"] = config.YEARS
            summary["target_games"] = len(tgt)
            summary["outcome_accuracy"] = float(correct / len(tgt))
            print(f"[elo] outcome accuracy on {config.YEARS}: {summary['outcome_accuracy']:.1%}")
    print(f"[elo] wrote {len(ratings)} rows -> {out_path}")
    return summary
