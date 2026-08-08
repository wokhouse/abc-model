"""Derived feature engineering built on the raw Polymarket + nflverse tables.

Modules:
* ``travel``     — great-circle travel distance and timezone shift per game.
* ``elo``        — FiveThirtyEight-style team and QB-adjusted Elo ratings.
* ``efficiency`` — Stage 0: rolling team-efficiency + differential + situational
                  game features from deep nflverse history (1999-2025).
"""
