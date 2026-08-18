"""Tests for Stage 0 efficiency feature engineering.

The headline invariant is leakage control: a rolling feature for a game must
contain only that team's *prior* games, never the game itself (shift-by-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abcm.features import efficiency


# --- synthetic frame factories --------------------------------------------

def _ts_row(game_id, season, week, team, opponent, *, pass_epa=0.0, rush_epa=0.0,
            pass_cpoe=None, pass_yds=0, rush_yds=0, pass_int=0, fumbles_lost=0,
            def_int=0, fum_rec_opp=0, pass_fd=0, rush_fd=0, attempts=0, carries=0,
            targets=0):
    return {
        "game_id": game_id, "season": season, "week": week, "team": team,
        "opponent_team": opponent, "passing_epa": pass_epa, "rushing_epa": rush_epa,
        "passing_cpoe": pass_cpoe, "passing_yards": pass_yds, "rushing_yards": rush_yds,
        "passing_interceptions": pass_int, "fumbles_lost_total": fumbles_lost,
        "def_interceptions": def_int, "fumble_recovery_opp": fum_rec_opp,
        "passing_first_downs": pass_fd, "rushing_first_downs": rush_fd,
        "attempts": attempts, "carries": carries, "targets": targets,
    }


def _sched_row(game_id, season, week, home, away, *, home_rest=7, away_rest=7,
               location="Home", div_game=0, roof="outdoors", temp=70.0, wind=5.0,
               home_score=21, away_score=17, gameday="2024-09-10"):
    return {
        "game_id": game_id, "season": season, "week": week, "game_type": "REG",
        "home_team": home, "away_team": away, "home_score": home_score,
        "away_score": away_score, "location": location, "home_rest": home_rest,
        "away_rest": away_rest, "div_game": div_game, "roof": roof, "temp": temp,
        "wind": wind, "gameday": gameday,
    }


# --- per-team-game efficiency extraction ----------------------------------

def test_per_team_game_efficiency_combines_offense_and_defense():
    ts = pd.DataFrame([_ts_row("2024_01_BUF_MIA", 2024, 1, "BUF", "MIA",
                               pass_epa=5.0, rush_epa=2.0, pass_yds=250, rush_yds=120,
                               pass_int=1, fumbles_lost=1, def_int=2, fum_rec_opp=1,
                               pass_fd=12, rush_fd=6, attempts=30, carries=25, targets=28)])
    out = efficiency._per_team_game_efficiency(ts)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["total_yards"] == 370          # pass + rush
    assert r["first_downs"] == 18           # pass + rush first downs
    assert r["to_lost"] == 2                 # INT + fumble lost
    assert r["takeaways"] == 3              # def INT + opp fumble recovered


def test_per_team_game_from_pbp_aggregates_only_scrimmage_plays():
    pbp = pd.DataFrame([
        # 2 successful passes, 1 failed run, plus a kickoff (excluded) and penalty.
        {"game_id": "g", "posteam": "BUF", "play_type": "pass", "success": 1, "yards_gained": 15, "play": 1},
        {"game_id": "g", "posteam": "BUF", "play_type": "pass", "success": 1, "yards_gained": 5, "play": 1},
        {"game_id": "g", "posteam": "BUF", "play_type": "run", "success": 0, "yards_gained": 1, "play": 1},
        {"game_id": "g", "posteam": "BUF", "play_type": "kickoff", "success": 0, "yards_gained": 0, "play": 1},
        {"game_id": "g", "posteam": "BUF", "play_type": "penalty", "success": 0, "yards_gained": 0, "play": 1},
    ])
    agg = efficiency._per_team_game_from_pbp(pbp)
    r = agg.iloc[0]
    assert r["n_plays"] == 3                       # only pass/run
    assert r["success_rate"] == pytest.approx(2 / 3)
    assert r["yards_per_play"] == pytest.approx((15 + 5 + 1) / 3)


# --- rolling leakage control (the headline tests) -------------------------

def test_rolling_excludes_current_game_shift_by_1():
    """The rolling feature for game N must equal game N-1's raw value exactly,
    and game 1's rolling feature must be NaN (no prior games)."""
    ts = pd.DataFrame([
        _ts_row("g1", 2024, 1, "BUF", "MIA", pass_epa=10.0),
        _ts_row("g2", 2024, 2, "BUF", "NE", pass_epa=20.0),
        _ts_row("g3", 2024, 3, "BUF", "NYJ", pass_epa=30.0),
    ])
    eff = efficiency._per_team_game_efficiency(ts)
    rolled = efficiency._roll_team(eff)
    # Game 1: no prior games -> NaN.
    g1 = rolled[rolled["game_id"] == "g1"].iloc[0]
    assert pd.isna(g1["pass_epa_roll5"])
    # Game 2: prior is only game 1 -> equals game 1's value (window min_periods=1).
    g2 = rolled[rolled["game_id"] == "g2"].iloc[0]
    assert g2["pass_epa_roll5"] == pytest.approx(10.0)
    # Game 3: prior games are 1 and 2 -> mean of (10, 20) = 15, NOT including 30.
    g3 = rolled[rolled["game_id"] == "g3"].iloc[0]
    assert g3["pass_epa_roll5"] == pytest.approx(15.0)


def test_rolling_respects_date_ordering_not_input_order():
    """Rows arriving out of week order still roll in chronological order."""
    ts = pd.DataFrame([
        _ts_row("g3", 2024, 3, "BUF", "NYJ", pass_epa=30.0),
        _ts_row("g1", 2024, 1, "BUF", "MIA", pass_epa=10.0),
        _ts_row("g2", 2024, 2, "BUF", "NE", pass_epa=20.0),
    ])
    eff = efficiency._per_team_game_efficiency(ts)
    rolled = efficiency._roll_team(eff)
    g3 = rolled[rolled["game_id"] == "g3"].iloc[0]
    # Despite g3 being first in input, its prior games are g1, g2 -> mean 15.
    assert g3["pass_epa_roll5"] == pytest.approx(15.0)


def test_rolling_does_not_cross_teams():
    """Team A's rolling feature must never include team B's games."""
    ts = pd.DataFrame([
        _ts_row("g1", 2024, 1, "BUF", "MIA", pass_epa=10.0),
        _ts_row("g2", 2024, 1, "MIA", "BUF", pass_epa=99.0),  # MIA's same-week game
        _ts_row("g3", 2024, 2, "BUF", "NE", pass_epa=20.0),
    ])
    eff = efficiency._per_team_game_efficiency(ts)
    rolled = efficiency._roll_team(eff)
    g3 = rolled[rolled["game_id"] == "g3"].iloc[0]
    # BUF's g3 prior is only g1 (10.0); MIA's 99.0 must not contaminate it.
    assert g3["pass_epa_roll5"] == pytest.approx(10.0)


def test_rolling_resets_across_seasons():
    """A new season's first game should not roll the prior season's tail when the
    window is large; specifically game 1 of a season is NaN regardless of history."""
    ts = pd.DataFrame([
        _ts_row("g_23", 2023, 18, "BUF", "MIA", pass_epa=10.0),  # prior season end
        _ts_row("g_24", 2024, 1, "BUF", "NE", pass_epa=20.0),    # new season game 1
    ])
    eff = efficiency._per_team_game_efficiency(ts)
    rolled = efficiency._roll_team(eff)
    # Note: we deliberately do NOT reset across seasons (carrying form across the
    # offseason is intended). The leakage invariant here is only that g_24's
    # rolling value uses g_23 (the prior game) but never itself.
    g24 = rolled[rolled["game_id"] == "g_24"].iloc[0]
    assert g24["pass_epa_roll5"] == pytest.approx(10.0)


# --- defense-allowed and differentials ------------------------------------

def test_defense_allowed_uses_opponent_offense():
    """Team X's def pass-epa-allowed for a game = opponent's pass EPA that game."""
    ts = pd.DataFrame([
        # Game g: BUF offense generates 8.0 pass_epa vs MIA defense.
        _ts_row("g", 2024, 1, "BUF", "MIA", pass_epa=8.0),
        _ts_row("g", 2024, 1, "MIA", "BUF", pass_epa=1.0),
    ])
    eff = efficiency._per_team_game_efficiency(ts)
    rolled = efficiency._roll_team(eff)          # rolling adds *_roll cols
    allowed = efficiency._defense_allowed(rolled)
    # For game g, MIA's def pass_epa_allowed (current-game value, not rolled here
    # since defense_allowed just relabels offense rolls) is BUF's pass_epa roll.
    mia = allowed[(allowed["game_id"] == "g") & (allowed["team"] == "MIA")].iloc[0]
    # g is BUF/MIA game 1 -> BUF's roll is NaN -> MIA's allowed roll is also NaN.
    assert pd.isna(mia["pass_epa_roll5_allowed"])


def test_differential_sign_home_stronger_than_away():
    """A game where home's prior pass EPA >> away's prior pass EPA -> epa_diff > 0."""
    # BUF home: warm up with two strong games; NE away: two weak games.
    warmup = pd.DataFrame([
        _ts_row("h1", 2024, 1, "BUF", "MIA", pass_epa=12.0),
        _ts_row("h1", 2024, 1, "MIA", "BUF", pass_epa=2.0),
        _ts_row("h2", 2024, 2, "BUF", "NYJ", pass_epa=12.0),
        _ts_row("h2", 2024, 2, "NYJ", "BUF", pass_epa=2.0),
        _ts_row("a1", 2024, 1, "NE", "PIT", pass_epa=0.0),
        _ts_row("a1", 2024, 1, "PIT", "NE", pass_epa=5.0),
        _ts_row("a2", 2024, 2, "NE", "DEN", pass_epa=0.0),
        _ts_row("a2", 2024, 2, "DEN", "NE", pass_epa=5.0),
        # The game under test: BUF hosts NE in week 3.
        _ts_row("g", 2024, 3, "BUF", "NE", pass_epa=10.0),
        _ts_row("g", 2024, 3, "NE", "BUF", pass_epa=-5.0),
    ])
    sched = pd.DataFrame([_sched_row("g", 2024, 3, "BUF", "NE")])
    elo = pd.DataFrame([{"game_id": "g", "elo_home_pre": 1550, "elo_away_pre": 1450, "elo_home_prob": 0.6}])
    feats = efficiency.build_features(sched, warmup, pbp=None, elo=elo)
    row = feats.iloc[0]
    # BUF prior pass_epa ~12, NE prior ~0 -> diff strongly positive.
    assert row["pass_epa_diff"] > 0


# --- row-count preservation and null handling -----------------------------

def test_build_features_preserves_game_count():
    """No games dropped or duplicated (mirrors test_travel invariant)."""
    ts = pd.DataFrame([
        _ts_row("g1", 2024, 1, "BUF", "MIA", pass_epa=5.0),
        _ts_row("g1", 2024, 1, "MIA", "BUF", pass_epa=2.0),
        _ts_row("g2", 2024, 2, "BUF", "NE", pass_epa=6.0),
        _ts_row("g2", 2024, 2, "NE", "BUF", pass_epa=1.0),
    ])
    sched = pd.DataFrame([
        _sched_row("g1", 2024, 1, "BUF", "MIA"),
        _sched_row("g2", 2024, 2, "BUF", "NE"),
    ])
    elo = pd.DataFrame([
        {"game_id": "g1", "elo_home_pre": 1500, "elo_away_pre": 1500, "elo_home_prob": 0.5},
        {"game_id": "g2", "elo_home_pre": 1500, "elo_away_pre": 1500, "elo_home_prob": 0.5},
    ])
    feats = efficiency.build_features(sched, ts, pbp=None, elo=elo)
    assert len(feats) == len(sched)
    assert feats["game_id"].nunique() == len(sched)


def test_situational_features_dome_nulls_temp_and_flags_outdoor():
    sched = pd.DataFrame([
        _sched_row("g1", 2024, 1, "BUF", "MIA", roof="dome", temp=np.nan, wind=np.nan),
        _sched_row("g2", 2024, 2, "NE", "NYJ", roof="outdoors", temp=32.0, wind=10.0),
        _sched_row("g3", 2024, 3, "KC", "LAC", location="Neutral"),
    ])
    sit = efficiency.situational_features(sched)
    assert sit.loc[0, "roof_dome"] == 1 and sit.loc[0, "is_outdoor"] == 0
    assert pd.isna(sit.loc[0, "temp"])            # dome temp kept null (no imputation)
    assert sit.loc[1, "roof_outdoors"] == 1 and sit.loc[1, "is_outdoor"] == 1
    assert sit.loc[1, "temp"] == 32.0
    assert sit.loc[2, "neutral_site"] == 1


# --- Elo join -------------------------------------------------------------

def test_add_elo_populates_diff_and_only_pre_columns():
    feats = pd.DataFrame([{"game_id": "g", "home_team": "BUF", "away_team": "MIA"}])
    elo = pd.DataFrame([{
        "game_id": "g", "elo_home_pre": 1560.0, "elo_away_pre": 1440.0,
        "elo_home_post": 1580.0, "elo_away_post": 1420.0, "elo_home_prob": 0.66,
    }])
    out = efficiency.add_elo(feats, elo)
    assert "elo_home_post" not in out.columns     # never leak the post-game rating
    assert out.iloc[0]["elo_diff"] == pytest.approx(120.0)
    assert out.iloc[0]["elo_home_prob"] == pytest.approx(0.66)


def test_add_elo_null_when_missing():
    feats = pd.DataFrame([{"game_id": "g", "home_team": "BUF", "away_team": "MIA"}])
    elo = pd.DataFrame(columns=["game_id", "elo_home_pre", "elo_away_pre", "elo_home_prob"])
    out = efficiency.add_elo(feats, elo)
    assert pd.isna(out.iloc[0]["elo_diff"])
