"""Tests for the SBR historical archive loader."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from abcm.sbr import loader


# --- team-name normalization ---------------------------------------------

@pytest.mark.parametrize("nick,expected", [
    ("Packers", "GB"), ("Fortyniners", "SF"), ("Washingtom", "WAS"),
    ("BuffaloBills", "BUF"), ("Bills", "BUF"), ("KCChiefs", "KC"),
    ("Oakland", "LV"), ("St.Louis", "LA"), ("SanDiego", "LAC"),
    ("NewYork", "NYJ"),
])
def test_sbr_nickname_to_abbr(nick, expected):
    assert loader._to_abbr(nick) == expected


def test_sbr_unknown_returns_none():
    assert loader._to_abbr("0") is None
    assert loader._to_abbr(0) is None
    assert loader._to_abbr("SomeUnknownTeam") is None


# --- date parsing ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (20110908.0, "2011-09-08"),
    (20210130.0, "2021-01-30"),
])
def test_parse_date(raw, expected):
    assert loader._parse_date(raw) == expected


def test_parse_date_invalid():
    assert loader._parse_date(None) is None
    assert loader._parse_date("abc") is None


# --- range validation drops garbage ---------------------------------------

def _make_raw(**kw):
    base = {
        "season": 2020, "date": 20200910.0, "home_team": "Packers", "away_team": "Bears",
        "home_final": "24", "away_final": "17",
        "home_close_ml": -200, "away_close_ml": 180,
        "home_open_spread": -3.5, "away_open_spread": 3.5,
        "home_close_spread": -3.5, "away_close_spread": 3.5,
        "home_2H_spread": 0, "away_2H_spread": 0, "2H_total": 0.0,
        "open_over_under": 45.5, "close_over_under": 45.5,
    }
    base.update(kw)
    return base


def test_load_lines_drops_garbage_spread(tmp_path, monkeypatch):
    """The SBR parser swap-bug produces spread=40.5; we drop it."""
    raw = pd.DataFrame([
        _make_raw(home_team="Packers", away_team="Bears"),  # clean
        _make_raw(home_close_spread=40.5, away_close_spread=-40.5),  # garbage
        _make_raw(home_team="0", away_team="Bears"),  # unknown team
    ])
    f = tmp_path / "nfl.json"
    raw.to_json(f, orient="records")
    monkeypatch.setattr(loader, "load_raw", lambda path=None: pd.read_json(f))
    out = loader.load_lines(validate=True)
    assert len(out) == 1  # only the clean row survives
    assert out.iloc[0]["home_abbr"] == "GB"


def test_load_lines_drops_garbage_total(tmp_path, monkeypatch):
    raw = pd.DataFrame([
        _make_raw(),  # clean
        _make_raw(open_over_under=1.0, close_over_under=1.0),  # garbage O/U
    ])
    f = tmp_path / "nfl.json"
    raw.to_json(f, orient="records")
    monkeypatch.setattr(loader, "load_raw", lambda path=None: pd.read_json(f))
    out = loader.load_lines(validate=True)
    assert len(out) == 1


# --- movement + cover label -----------------------------------------------

def test_spread_movement_and_cover_label(tmp_path, monkeypatch):
    raw = pd.DataFrame([_make_raw(
        home_open_spread=-3.0, home_close_spread=-5.0,  # movement = -2.0 (more favored)
        home_final="28", away_final="17",  # margin +11 > 5 -> covered
    )])
    f = tmp_path / "nfl.json"
    raw.to_json(f, orient="records")
    monkeypatch.setattr(loader, "load_raw", lambda path=None: pd.read_json(f))
    out = loader.load_lines(validate=True)
    assert out.iloc[0]["spread_movement"] == pytest.approx(-2.0)
    assert out.iloc[0]["home_covered_close"] == 1  # +11 margin beats -5 spread


# --- game_id join ---------------------------------------------------------

def test_join_game_ids_matches_on_date_and_teams():
    lines = pd.DataFrame([{
        "season": 2020, "gameday": "2020-09-13",
        "home_abbr": "LV", "away_abbr": "CAR",
    }])
    sched = pd.DataFrame([{
        "game_id": "2020_01_CAR_LV", "season": 2020, "gameday": "2020-09-13",
        "home_team": "LV", "away_team": "CAR",
    }])
    out = loader.join_game_ids(lines, sched)
    assert len(out) == 1
    assert out.iloc[0]["game_id"] == "2020_01_CAR_LV"


def test_join_game_ids_drops_unmatched():
    lines = pd.DataFrame([{
        "season": 2020, "gameday": "2020-09-13",
        "home_abbr": "LV", "away_abbr": "CAR",
    }])
    sched = pd.DataFrame([{
        "game_id": "2020_01_XYZ_LV", "season": 2020, "gameday": "2020-09-14",  # wrong date
        "home_team": "LV", "away_team": "XYZ",
    }])
    out = loader.join_game_ids(lines, sched)
    assert len(out) == 0  # no match -> dropped
