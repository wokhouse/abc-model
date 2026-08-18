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


# --- Spread parsing & matching --------------------------------------------


def _make_spread_row(
    *,
    market_id="s1",
    question="Spread: Steelers (-5.5)",
    event_title="Steelers vs. Panthers",
    market_end="2024-09-15T18:00:00Z",
):
    return {
        "market_id": market_id,
        "question": question,
        "outcomes": '["Yes", "No"]',
        "event_title": event_title,
        "market_end": market_end,
        "event_end": market_end,
        "market_type": "spread",
    }


def test_parse_spread_favored_negative_margin():
    """'Spread: Steelers (-5.5)' => Steelers favored by 5.5."""
    parsed = match_mod.parse_spread("Spread: Steelers (-5.5)", "Steelers vs. Panthers")
    assert parsed["spread_favored_abbr"] == "PIT"
    assert parsed["spread_opponent_abbr"] == "CAR"
    assert parsed["spread_margin"] == 5.5


def test_parse_spread_underdog_positive_margin():
    """'Spread: Jets (1.5)' => Jets are getting points; opponent is favored."""
    parsed = match_mod.parse_spread("Spread: Jets (1.5) ", "Eagles vs. Jets")
    # Jets are the underdog; the Eagles (from the title) are favored by 1.5.
    assert parsed["spread_favored_abbr"] == "PHI"
    assert parsed["spread_opponent_abbr"] == "NYJ"
    assert parsed["spread_margin"] == 1.5


def test_parse_spread_margin_of_victory_phrasing():
    """'Will the Chiefs win by 4 or more points?' => Chiefs favored by 4."""
    parsed = match_mod.parse_spread(
        "Will the Chiefs win by 4 or more points?",
        "NFL Kickoff: Chiefs vs. Ravens",
    )
    assert parsed["spread_favored_abbr"] == "KC"
    assert parsed["spread_opponent_abbr"] == "BAL"
    assert parsed["spread_margin"] == 4.0


def test_parse_spread_generic_title_returns_no_opponent():
    """A 'NFL Week 1: Spreads' title carries no opponent identity.

    The named favorite and margin still parse, but without an opponent the
    market can't be matched to a game downstream.
    """
    parsed = match_mod.parse_spread(
        "Will the Falcons win by 4 or more points?", "NFL Week 1: Spreads"
    )
    assert parsed["spread_favored_abbr"] == "ATL"  # named team resolves
    assert parsed["spread_margin"] == 4.0
    assert parsed["spread_opponent_abbr"] is None  # no opponent in title


def test_spread_market_matches_to_game():
    """A spread market joins to the same game a moneyline would."""
    markets = pd.DataFrame([_make_spread_row(
        question="Spread: Chiefs (-3.5)", event_title="Chiefs vs. Raiders",
        market_end="2024-10-01T17:00:00Z",
    )])
    schedules = pd.DataFrame([_make_schedule(
        game_id="2024_04_LV_KC", season=2024, week=4,
        gameday="2024-09-30", away="KC", home="LV",
    )])
    out = match_mod.reconcile(markets, schedules)
    row = out.iloc[0]
    assert row["match_status"] == "matched"
    assert row["game_id"] == "2024_04_LV_KC"
    assert row["spread_favored_abbr"] == "KC"
    assert row["spread_margin"] == 3.5


def test_spread_market_no_game_flagged():
    """A spread whose teams don't meet in the window is 'teams_found_no_game'."""
    markets = pd.DataFrame([_make_spread_row(
        question="Spread: Chiefs (-3.5)", event_title="Chiefs vs. Raiders",
    )])
    schedules = pd.DataFrame([_make_schedule(
        game_id="2024_04_BUF_MIA", season=2024, week=4,
        gameday="2024-09-15", away="MIA", home="BUF",
    )])
    out = match_mod.reconcile(markets, schedules)
    assert out.iloc[0]["match_status"] == "teams_found_no_game"


def test_spread_covered_favorite_covers():
    """Favorite wins by more than the margin => covered (True)."""
    # Home team KC favored by 3.5; won 27-20 (margin 7 > 3.5).
    assert match_mod.spread_covered("KC", 3.5, "KC", "BUF", 27, 20) is True


def test_spread_covered_favorite_fails_to_cover():
    """Favorite wins but by less than the margin => not covered (False)."""
    # Home team KC favored by 7; won 27-20 (margin 7, not > 7) => push => False.
    assert match_mod.spread_covered("KC", 7.0, "KC", "BUF", 27, 20) is False


def test_spread_covered_away_favorite():
    """Favorite is the away team; margin computed from away-home score."""
    # Away team KC favored by 3.5; away score 27, home 20 => +7 > 3.5 => cover.
    assert match_mod.spread_covered("KC", 3.5, "BUF", "KC", 20, 27) is True


def test_spread_covered_favorite_loses():
    """Favorite loses outright => not covered."""
    assert match_mod.spread_covered("KC", 3.5, "KC", "BUF", 10, 20) is False


def test_spread_covered_unknown_favorite_returns_none():
    """If the favorite isn't in the game, return None (data-quality flag)."""
    assert match_mod.spread_covered("NE", 3.5, "KC", "BUF", 27, 20) is None
    assert match_mod.spread_covered(None, 3.5, "KC", "BUF", 27, 20) is None
