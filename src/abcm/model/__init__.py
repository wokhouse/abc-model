"""Modeling stages built on the feature frames.

Modules:
* ``outcome``  — Stage 1: pre-train a game-outcome (home win probability) model
                 on deep history; predictions feed Stage 2's ``market_delta``.
* ``calibrate``— Platt scaling + Brier/reliability helpers (shared by stages).
"""
