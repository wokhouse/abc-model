"""Line-movement features and Closing Line Value (CLV) from OddsPortal open/close.

Two derived signal classes built on the open/close odds scraped from OddsPortal:

1. **Line movement** — how much the market moved from open to close. This is a
   genuinely new signal for Stage 2 (the model never saw it before): sharp money
   moves lines, so the direction and magnitude of open->close movement carries
   information about where the "smart" side is. Movement is expressed as the
   change in the *favorite's* implied probability (positive = favorite money came
   in = market thinks the favorite is more likely than the opener implied).

2. **Closing Line Value (CLV)** — the de-vigged gap between a model probability
   and the closing line. The research is explicit that consistent positive CLV is
   a stronger validity signal than ROI (which is noisy at small n). CLV is the
   Stage 3 primary metric: ``clv = model_prob - implied_prob(close_line, de-vigged)``.

All odds here are decimal (as scraped). De-vigging removes the bookmaker margin:
the two sides' implied probs sum to >1 (the vig); normalize them to sum to 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decimal_to_implied(odds: float | np.ndarray) -> float | np.ndarray:
    """Decimal odds -> raw implied probability (with vig). 2.0 -> 0.5."""
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = np.where(odds > 1.0, 1.0 / odds, np.nan)
    return imp


def devig(side_a_odds: float, side_b_odds: float) -> tuple[float, float]:
    """Remove the vig from a two-way market, returning fair probabilities.

    Each side's implied prob is divided by the overround (the sum of both implied
    probs) so they sum to 1. E.g. 1.91/1.91 -> 0.521/0.521 -> 0.5/0.5 after devig.
    """
    pa = decimal_to_implied(side_a_odds)
    pb = decimal_to_implied(side_b_odds)
    overround = pa + pb
    if overround is None or np.isnan(overround) or overround == 0:
        return float("nan"), float("nan")
    return float(pa / overround), float(pb / overround)


def favorite_implied_prob(home_odds: float, away_odds: float) -> float:
    """The favorite's de-vigged win probability (the side with the lower decimal
    odds = higher implied prob = the favorite). Positive = favorite > 50%."""
    ph, pa = devig(home_odds, away_odds)
    if np.isnan(ph):
        return float("nan")
    # Return the larger of the two (the favorite's prob), signed so >0.5 = fav.
    return max(ph, pa)


def ml_movement(home_open: float, away_open: float,
                home_close: float, away_close: float) -> float:
    """Change in the favorite's de-vigged probability from open to close.

    Positive = money came in on the favorite (the market's perceived edge grew).
    Negative = underdog money / fade. This is the headline movement signal.
    """
    fav_open = favorite_implied_prob(home_open, away_open)
    fav_close = favorite_implied_prob(home_close, away_close)
    if np.isnan(fav_open) or np.isnan(fav_close):
        return float("nan")
    return float(fav_close - fav_open)


def compute_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add line-movement features to a frame carrying OddsPortal open/close columns.

    Adds:
    * ``ml_movement`` — favorite's de-vigged prob change open->close (the headline).
    * ``spread_odds_movement`` — favorite spread-odds movement (close - open, in
      decimal-odds space; negative = favorite got more expensive to bet).
    * ``total_odds_movement`` — over-odds movement (positive = over money came in).

    Null where the open/close odds are missing (the frame is left-joined, so
    not every game has OddsPortal data).
    """
    out = df.copy()
    out["ml_movement"] = np.vectorize(ml_movement, otypes=[float])(
        out.get("home_open_ml", np.nan), out.get("away_open_ml", np.nan),
        out.get("home_close_ml", np.nan), out.get("away_close_ml", np.nan),
    )
    # Spread-odds movement: use the home side's odds movement as a proxy for the
    # line pressure on whichever side home is. (A signed favorite-side movement
    # would require resolving the favorite per-game; the raw movement is enough
    # signal for the model, which can learn the home/away context itself.)
    if "spread_home_open_odds" in out and "spread_home_close_odds" in out:
        out["spread_odds_movement"] = (
            out["spread_home_close_odds"] - out["spread_home_open_odds"]
        )
    if "over_open_odds" in out and "over_close_odds" in out:
        out["total_odds_movement"] = out["over_close_odds"] - out["over_open_odds"]
    return out


def clv(model_prob: float, close_home_odds: float, close_away_odds: float,
        *, model_is_home: bool = True) -> float:
    """Closing Line Value: model probability minus the de-vigged closing implied prob.

    This is the Stage 3 validity metric. ``model_is_home`` selects which side the
    model probability refers to. Positive CLV = the model valued that side more
    than the closing market did (the holy grail: consistent positive CLV beats ROI
    as a validity signal because it's far less noisy at small n).
    """
    fair_home, fair_away = devig(close_home_odds, close_away_odds)
    if np.isnan(fair_home):
        return float("nan")
    market_prob = fair_home if model_is_home else fair_away
    return float(model_prob - market_prob)
