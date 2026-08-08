"""Tests for Polymarket market-type classification.

The classifier routes markets to downstream handling (moneyline vs spread vs
totals matching, prop/futures exclusion), so the patterns are pinned here against
real Polymarket phrasings and the adversarial cases that broke the old regex
(year false-positives, two-team spread outcomes, season win totals).
"""
from __future__ import annotations

import pytest

from abcm.polymarket import events


@pytest.mark.parametrize(
    "question,event_title,outcomes,expected",
    [
        # --- spread: explicit keyword + margin-of-victory forms ---
        ("Spread: Steelers (-5.5)", "Steelers vs. Panthers", ["Yes", "No"], "spread"),
        ("Spread: Jets (1.5) ", "Eagles vs. Jets", ["Yes", "No"], "spread"),
        ("Chiefs (-1.5) vs. Eagles Spread", "Chiefs vs. Eagles", ["Yes", "No"], "spread"),
        # Two team-name outcomes on a spread question must NOT fall through to moneyline.
        ("Spread: Eagles (-7.5)", "Cowboys vs. Eagles", ["Eagles", "Cowboys"], "spread"),
        ("Will the Chiefs win by 4 or more points?", "Chiefs vs. Raiders", ["Yes", "No"], "spread"),
        ("Will the Eagles beat the Chiefs by 9 or more points?", "Super Bowl LIX", ["Yes", "No"], "spread"),
        # --- game_total: O/U and combined-points forms ---
        ("Steelers vs Panthers: O/U 35.5", "Steelers vs. Panthers", ["Over", "Under"], "game_total"),
        ("Falcons Team Total: O/U 26.5", "Falcons Team Total", ["Over", "Under"], "game_total"),
        ("Will there be 47 or more combined points scored?", "NFL Week 1: Totals", ["Yes", "No"], "game_total"),
        # --- season_total: distinct from game_total ---
        ("Will the Cardinals win 9 or more regular season games?", "NFL Win Totals", ["Yes", "No"], "season_total"),
        # --- moneyline: genuine two-team game outcomes ---
        ("NFL: Houston Texans vs. Chicago Bears", "Texans vs Bears", ["Texans", "Bears"], "moneyline"),
        ("Cardinals vs. Seahawks", "Cardinals vs Seahawks", ["Cardinals", "Seahawks"], "moneyline"),
        # --- halftime: 1H moneyline / spread / O/U ---
        ("Jets vs. Ravens: 1H Moneyline", "Jets vs Ravens", ["Jets", "Ravens"], "halftime"),
        ("Browns vs. Raiders: 1H O/U 19.5", "Browns vs Raiders", ["Over", "Under"], "halftime"),
        ("1H Spread: Packers (-2.5)", "Packers vs Bears", ["Yes", "No"], "halftime"),
        # --- player_props ---
        ("Mahomes: Passing Yards O/U 275.5", "Chiefs props", ["Over", "Under"], "player_props"),
        ("First Touchdown: Davante Adams", "Packers props", ["Davante Adams", "No Touchdown"], "player_props"),
        ("Jacory Croskey-Merritt: Anytime Touchdown", "Commanders props", ["Yes", "No"], "player_props"),
        # --- futures ---
        ("Will the Chiefs win the Super Bowl?", "Super Bowl LVIII", ["Yes", "No"], "futures"),
        ("Will Caleb Williams be drafted 1st overall?", "2024 NFL Draft", ["Yes", "No"], "futures"),
    ],
)
def test_classify_market_real_phrasings(question, event_title, outcomes, expected):
    assert events.classify_market(question, event_title=event_title, outcomes=outcomes) == expected


@pytest.mark.parametrize(
    "question,event_title,outcomes",
    [
        # Bare years must not trigger the spread regex (the old ``[-+]\d`` bug).
        ("Will X win NFL MVP for the 2024-25 season?", "NFL Awards", ["Yes", "No"]),
        ("2024 NFL Draft: 2nd Pick", "2024 NFL Draft", ["Yes", "No"]),
        # A draft-pick player-vs-player market is a future, not a game moneyline.
        ("Drake Maye or Jayden Daniels drafted first?", "2024 NFL Draft", ["Maye", "Daniels"]),
    ],
)
def test_adversarial_non_moneyline(question, event_title, outcomes):
    """Cases that previously leaked into the wrong bucket."""
    label = events.classify_market(question, event_title=event_title, outcomes=outcomes)
    assert label != "moneyline", f"{question!r} should not be moneyline (got {label})"


def test_year_in_question_does_not_trigger_spread():
    r"""The old ``[-+]\d`` spread regex matched years like '2024-25'.

    Regression: a season-suffix string must not classify as a spread.
    """
    assert events.classify_market(
        "NFL MVP 2024-25", event_title="NFL Awards", outcomes=["Yes", "No"]
    ) != "spread"


def test_two_team_outcomes_with_spread_keyword_is_spread():
    """The two-team-outcome moneyline check must yield to an explicit spread.

    Polymarket lists spread markets with two team outcomes, e.g.
    ["Eagles","Cowboys"] on "Spread: Eagles (-7.5)"; without the keyword guard
    these were mislabeled moneyline.
    """
    label = events.classify_market(
        "Spread: Eagles (-7.5)",
        event_title="Cowboys vs. Eagles",
        outcomes=["Eagles", "Cowboys"],
    )
    assert label == "spread"


def test_empty_inputs_return_other():
    assert events.classify_market(None) == "other"
    assert events.classify_market("") == "other"
    assert events.classify_market("   ") == "other"


def test_returns_one_of_known_types():
    """All outputs must be in the documented type set."""
    allowed = {
        "moneyline", "spread", "game_total", "season_total",
        "player_props", "halftime", "futures", "other",
    }
    samples = [
        ("Texans vs Bears", ["Texans", "Bears"]),
        ("Random unknown market", ["Yes", "No"]),
        (None, None),
    ]
    for q, outs in samples:
        assert events.classify_market(q, outcomes=outs) in allowed
