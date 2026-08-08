"""Tests for Polymarket nickname -> nflverse abbreviation normalization."""
from __future__ import annotations

import pytest

from abcm.reconcile import teams


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Texans", "HOU"),
        ("49ers", "SF"),
        ("Bears", "CHI"),
        ("Commanders", "WAS"),
        ("Rams", "LA"),
        ("Chargers", "LAC"),
        ("Chiefs", "KC"),
        ("Packers", "GB"),
    ],
)
def test_nickname_to_abbr_known_teams(label, expected):
    assert teams.nickname_to_abbr(label) == expected


def test_trailing_whitespace_is_stripped():
    """Polymarket labels often carry trailing spaces ('Bears ', 'Jets ')."""
    assert teams.nickname_to_abbr("Bears ") == "CHI"
    assert teams.nickname_to_abbr("Jets ") == "NYJ"


def test_spread_suffix_is_stripped():
    """A spread market outcome like 'Chiefs -1.5' should still resolve to the team."""
    assert teams.nickname_to_abbr("Chiefs -1.5") == "KC"
    assert teams.nickname_to_abbr("Eagles +1.5") == "PHI"


def test_non_team_outcomes_return_none():
    assert teams.nickname_to_abbr("Over") is None
    assert teams.nickname_to_abbr("Under") is None
    assert teams.nickname_to_abbr("Yes") is None
    assert teams.nickname_to_abbr("No") is None


def test_player_names_return_none():
    """A player-prop outcome (e.g. 'Mahomes') is not a team."""
    assert teams.nickname_to_abbr("Mahomes") is None
    assert teams.nickname_to_abbr("Allen ") is None


def test_historical_washington_names_normalize():
    assert teams.nickname_to_abbr("Washington") == "WAS"
    assert teams.nickname_to_abbr("Redskins") == "WAS"


def test_none_and_empty():
    assert teams.nickname_to_abbr(None) is None
    assert teams.nickname_to_abbr("") is None


def test_all_32_teams_present():
    """Every nflverse team_abbr must be reachable via some nickname."""
    nflverse_abbrs = {
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
        "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LA", "LAC", "MIA", "MIN",
        "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS",
    }
    assert nflverse_abbrs.issubset(teams.all_abbrs())
