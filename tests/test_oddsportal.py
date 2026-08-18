"""Tests for the OddsPortal ingestion module."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from abcm.oddsportal import ingest, loader, teams


# --- team normalization ---------------------------------------------------

def test_oddsportal_to_abbr_known_teams():
    assert teams.oddsportal_to_abbr("Kansas City Chiefs") == "KC"
    assert teams.oddsportal_to_abbr("Philadelphia Eagles") == "PHI"
    assert teams.oddsportal_to_abbr("San Francisco 49ers") == "SF"
    # Case-insensitive
    assert teams.oddsportal_to_abbr("buffalo bills") == "BUF"


def test_oddsportal_to_abbr_unknown_returns_none():
    assert teams.oddsportal_to_abbr("Some Unknown Team") is None
    assert teams.oddsportal_to_abbr(None) is None


# --- market slug generation -----------------------------------------------

@pytest.mark.parametrize("line,expected", [
    (-4.5, "asian_handicap_-4_5"),
    (-3.0, "asian_handicap_-3_0"),
    (1.5, "asian_handicap_+1_5"),
    (7.0, "asian_handicap_+7_0"),
    (0, "asian_handicap_0"),
    (-14.0, "asian_handicap_-14_0"),
])
def test_handicap_slug(line, expected):
    assert loader._handicap_slug(line) == expected


@pytest.mark.parametrize("line,expected", [
    (47.5, "over_under_47_5"),
    (39.0, "over_under_39_0"),
    (33.5, "over_under_33_5"),
])
def test_total_slug(line, expected):
    assert loader._total_slug(line) == expected


# --- link parsing ---------------------------------------------------------

def test_teams_from_link_parses_both_teams():
    link = "https://www.oddsportal.com/american-football/h2h/kansas-city-chiefs-vienoNr7/philadelphia-eagles-dGgVhRLK/#OW"
    away, home = loader._teams_from_link(link)
    assert "Chiefs" in away
    assert "Eagles" in home


def test_slug_to_name_strips_id_hash():
    assert loader._slug_to_name("kansas-city-chiefs-vienoNr7") == "Kansas City Chiefs"
    assert loader._slug_to_name("buffalo-bills-t2lcn4tk") == "Buffalo Bills"


# --- normalization (using realistic synthetic records) -------------------

def _ml_record(home="Philadelphia Eagles", away="Kansas City Chiefs"):
    return {
        "home_team": home, "away_team": away,
        "home_score": "40", "away_score": "22",
        "match_date": "2025-02-09 23:30:00 UTC",
        "1x2_market": [{
            "1": "2.00", "X": "8.50", "2": "2.20",
            "bookmaker_name": "bet365.us",
            "odds_history_data": [
                {"opening_odds": {"timestamp": "2026-01-26T19:02:00", "odds": 2.1},
                 "odds_history": [{"timestamp": "2026-02-09T12:24:00", "odds": 2.0}]},
                {"opening_odds": {"timestamp": "2026-01-26T19:02:00", "odds": 9.0},
                 "odds_history": [{"timestamp": "2026-02-07T23:19:00", "odds": 8.5}]},
                {"opening_odds": {"timestamp": "2026-01-26T19:02:00", "odds": 2.0},
                 "odds_history": [{"timestamp": "2026-02-09T12:24:00", "odds": 2.2}]},
            ],
        }],
    }


def test_normalize_match_extracts_moneyline_open_close():
    row = ingest.normalize_match(_ml_record(), spread_line=None, total_line=None)
    assert row["home_team"] == "PHI"
    assert row["away_team"] == "KC"
    # Home side "1": opened 2.1, closed 2.0 (last history point).
    assert row["home_open_ml"] == pytest.approx(2.1)
    assert row["home_close_ml"] == pytest.approx(2.0)
    # Away side "2": opened 2.0, closed 2.2.
    assert row["away_open_ml"] == pytest.approx(2.0)
    assert row["away_close_ml"] == pytest.approx(2.2)
    assert row["home_score"] == 40 and row["away_score"] == 22


def test_normalize_match_returns_none_for_unknown_team():
    rec = _ml_record(home="Some City Unknowns", away="Kansas City Chiefs")
    assert ingest.normalize_match(rec, spread_line=None, total_line=None) is None


def test_normalize_match_handles_spread_and_total():
    rec = _ml_record()
    rec["asian_handicap_-2_5_market"] = [{
        "1": "2.15", "2": "1.67", "bookmaker_name": "BetMGM.us",
        "odds_history_data": [
            {"opening_odds": {"timestamp": "x", "odds": 2.25},
             "odds_history": [{"timestamp": "y", "odds": 2.15}]},
            {"opening_odds": {"timestamp": "x", "odds": 1.62},
             "odds_history": [{"timestamp": "y", "odds": 1.67}]},
        ],
    }]
    rec["over_under_48_5_market"] = [{
        "odds_over": "1.91", "odds_under": "1.91", "bookmaker_name": "bet365.us",
        "odds_history_data": [
            {"opening_odds": {"timestamp": "x", "odds": 1.76},
             "odds_history": [{"timestamp": "y", "odds": 1.91}]},
            {"opening_odds": {"timestamp": "x", "odds": 2.06},
             "odds_history": [{"timestamp": "y", "odds": 1.91}]},
        ],
    }]
    row = ingest.normalize_match(rec, spread_line=-2.5, total_line=48.5)
    assert row["spread_line"] == pytest.approx(-2.5)
    assert row["spread_home_open_odds"] == pytest.approx(2.25)
    assert row["spread_home_close_odds"] == pytest.approx(2.15)
    assert row["total_line"] == pytest.approx(48.5)
    assert row["over_open_odds"] == pytest.approx(1.76)
    assert row["over_close_odds"] == pytest.approx(1.91)


def test_open_close_uses_opening_field_and_last_history_point():
    block = {
        "opening_odds": {"timestamp": "t0", "odds": 2.1},
        "odds_history": [{"timestamp": "t1", "odds": 2.0}, {"timestamp": "t2", "odds": 2.2}],
    }
    open_odds, close_odds = ingest._open_close(block)
    assert open_odds == pytest.approx(2.1)
    assert close_odds == pytest.approx(2.2)  # last point, not first


def test_pick_book_prefers_sharp_books():
    books = [
        {"bookmaker_name": "BetMGM.us", "odds": 1.9},
        {"bookmaker_name": "Pinnacle", "odds": 1.95},
        {"bookmaker_name": "bet365.us", "odds": 1.92},
    ]
    picked = ingest._pick_book(books, ["Pinnacle", "bet365"])
    assert picked["bookmaker_name"] == "Pinnacle"


# --- range validation -----------------------------------------------------

def test_spread_range_validation_rejects_garbage(tmp_path, monkeypatch):
    """The SBR scraper produced spread=40.5 garbage; we reject out-of-range lines."""
    bad = pd.DataFrame([{"game_id": "g", "spread_line": 40.5, "total_line": 1.0}])
    # ingest.ingest_season applies between(SPREAD_MIN, SPREAD_MAX); 40.5 is outside.
    out = bad[bad["spread_line"].isna() | bad["spread_line"].between(ingest.SPREAD_MIN, ingest.SPREAD_MAX)]
    assert len(out) == 0  # garbage row dropped
