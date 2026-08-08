"""Tests for the Kaggle betting-line loader (normalization logic only).

The live download requires credentials + network, so these tests exercise the
pure-normalization helpers against synthetic CSV content.
"""
from __future__ import annotations

import pandas as pd
import pytest

from abcm.kaggle import loader


def test_team_name_map_covers_32_active_teams():
    """Every active nflverse abbr must be reachable from some full name."""
    active = {
        "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB",
        "HOU","IND","JAX","KC","LV","LA","LAC","MIA","MIN","NE","NO","NYG","NYJ",
        "PHI","PIT","SF","SEA","TB","TEN","WAS",
    }
    assert active.issubset(set(loader.TEAM_NAME_TO_ABBR.values()))


def test_historical_names_normalize():
    """Relocated/rebranded franchises map to their current abbr."""
    assert loader.TEAM_NAME_TO_ABBR["san diego chargers"] == "LAC"
    assert loader.TEAM_NAME_TO_ABBR["oakland raiders"] == "LV"
    assert loader.TEAM_NAME_TO_ABBR["st. louis rams"] == "LA"
    assert loader.TEAM_NAME_TO_ABBR["washington redskins"] == "WAS"


def test_team_name_lookup_is_case_insensitive():
    """The loader lowercases the franchise name before lookup."""
    # Simulate the lookup the loader performs.
    name = "New England Patriots"
    assert loader.TEAM_NAME_TO_ABBR[name.strip().lower()] == "NE"


def test_week_normalization_drops_playoff_labels(tmp_path, monkeypatch):
    """Playoff-week labels ('Wildcard') are dropped; numeric weeks coerce to int."""
    csv = tmp_path / "spreadspoke_scores.csv"
    csv.write_text(
        "schedule_date,schedule_season,schedule_week,team_home,team_away,"
        "score_home,score_away,team_favorite_id,spread_favorite,over_under_line,"
        "stadium_neutral\n"
        "9/10/2023,2023,1,Atlanta Falcons,Carolina Panthers,24,10,ATL,-3.5,39.5,False\n"
        "1/13/2024,2023,Wildcard,Kansas City Chiefs,Miami Dolphins,26,7,KC,-10,48,False\n"
    )
    monkeypatch.setattr(loader, "download_dataset", lambda dest_dir=None: tmp_path)

    df = loader.load_betting_lines(years=[2023])
    assert len(df) == 1  # only the regular-season row survives
    row = df.iloc[0]
    assert row["schedule_week"] == 1
    assert int(row["schedule_week"]) == 1  # numeric, not a playoff label string
    assert row["home_team_abbr"] == "ATL"
    assert row["away_team_abbr"] == "CAR"
    assert row["spread_favorite"] == pytest.approx(-3.5)


def test_years_filter_applied(tmp_path, monkeypatch):
    csv = tmp_path / "spreadspoke_scores.csv"
    csv.write_text(
        "schedule_date,schedule_season,schedule_week,team_home,team_away,"
        "score_home,score_away,team_favorite_id,spread_favorite,over_under_line,"
        "stadium_neutral\n"
        + "\n".join([
            f"9/10/{y},{y},1,Detroit Lions,Kansas City Chiefs,21,20,KC,-1.5,52,False"
            for y in (2022, 2023, 2024)
        ]) + "\n"
    )
    monkeypatch.setattr(loader, "download_dataset", lambda dest_dir=None: tmp_path)
    df = loader.load_betting_lines(years=[2023, 2024])
    assert sorted(df["schedule_season"].unique()) == [2023, 2024]
