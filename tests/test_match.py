"""Tests for market -> game matching."""
from __future__ import annotations

import json

import pandas as pd

from abcm.reconcile import match as match_mod


def _make_market_row(
    *,
    market_id="m1",
    outcomes=("Texans", "Bears"),
    market_end="2024-09-15T18:00:00Z",
    market_type="moneyline",
):
    return {
        "market_id": market_id,
        "question": "NFL: Texans vs. Bears",
        "outcomes": json.dumps(list(outcomes)),
        "market_end": market_end,
        "event_end": market_end,
        "market_type": market_type,
        "yes_token_id": "token-yes",
        "no_token_id": "token-no",
    }


def _make_schedule(*, game_id, season, week, gameday, away, home):
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "gameday": gameday,
        "away_team": away,
        "home_team": home,
    }


def test_moneyline_matches_within_window():
    markets = pd.DataFrame([_make_market_row(market_end="2024-09-15T18:00:00Z")])
    # Game is two days after the market end date — inside the ±3 day window.
    schedules = pd.DataFrame(
        [_make_schedule(game_id="2024_02_CHI_HOU", season=2024, week=2,
                        gameday="2024-09-17", away="HOU", home="CHI")]
    )
    out = match_mod.reconcile(markets, schedules)
    assert len(out) == 1
    assert out.iloc[0]["match_status"] == "matched"
    assert out.iloc[0]["game_id"] == "2024_02_CHI_HOU"
    assert out.iloc[0]["pm_team0_abbr"] == "HOU"
    assert out.iloc[0]["pm_team1_abbr"] == "CHI"


def test_teams_unordered_still_match():
    """Outcome order need not match nflverse away/home order."""
    markets = pd.DataFrame([_make_market_row(outcomes=("Bears", "Texans"))])
    schedules = pd.DataFrame(
        [_make_schedule(game_id="2024_02_CHI_HOU", season=2024, week=2,
                        gameday="2024-09-15", away="HOU", home="CHI")]
    )
    out = match_mod.reconcile(markets, schedules)
    assert out.iloc[0]["match_status"] == "matched"


def test_no_matching_game_flagged():
    """Both teams resolve but no game in the window -> teams_found_no_game."""
    markets = pd.DataFrame([_make_market_row(outcomes=("Texans", "Bears"))])
    # A different matchup on that day.
    schedules = pd.DataFrame(
        [_make_schedule(game_id="2024_02_KC_BUF", season=2024, week=2,
                        gameday="2024-09-15", away="KC", home="BUF")]
    )
    out = match_mod.reconcile(markets, schedules)
    assert out.iloc[0]["match_status"] == "teams_found_no_game"


def test_window_too_narrow_misses_game():
    """Game 5 days out should NOT match (window is ±3)."""
    markets = pd.DataFrame([_make_market_row(market_end="2024-09-10T18:00:00Z")])
    schedules = pd.DataFrame(
        [_make_schedule(game_id="2024_02_CHI_HOU", season=2024, week=2,
                        gameday="2024-09-15", away="HOU", home="CHI")]
    )
    out = match_mod.reconcile(markets, schedules)
    assert out.iloc[0]["match_status"] == "teams_found_no_game"


def test_non_team_outcomes_not_matched():
    """Totals markets ('Over'/'Under') should not be treated as moneylines."""
    markets = pd.DataFrame([_make_market_row(outcomes=("Over", "Under"))])
    schedules = pd.DataFrame(
        [_make_schedule(game_id="2024_02_CHI_HOU", season=2024, week=2,
                        gameday="2024-09-15", away="HOU", home="CHI")]
    )
    out = match_mod.reconcile(markets, schedules)
    assert out.iloc[0]["match_status"] == "no_team_outcomes"


def test_non_moneyline_markets_marked_separately():
    """A futures market (not 'moneyline' type) gets its own status."""
    markets = pd.DataFrame(
        [_make_market_row(market_type="futures", outcomes=("Chiefs", "Bills"))]
    )
    schedules = pd.DataFrame(
        [_make_schedule(game_id="2024_02_BUF_KC", season=2024, week=2,
                        gameday="2024-09-15", away="KC", home="BUF")]
    )
    out = match_mod.reconcile(markets, schedules)
    assert out.iloc[0]["match_status"] == "not_moneyline"
