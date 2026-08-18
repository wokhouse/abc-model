"""Calibration and proper-scoring helpers shared across the modeling stages.

The plan is explicit on one point: **use Platt (sigmoid) scaling, not isotonic
regression**. Isotonic overfits below ~1,000-2,000 samples (Niculescu-Mizil &
Caruana 2005; sklearn docs). Our calibration sets are in the low hundreds, so
Platt is the least-bad choice. Calibration must always be fit on a temporally
held-out split, separate from the hyperparameter-tuning fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def brier_score(probas: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions (proper scoring rule).

    Thin wrapper over sklearn's ``brier_score_loss`` so callers can pair a model's
    score with the Elo baseline's on the identical rows. Labels must be {0, 1}.
    """
    return float(brier_score_loss(np.asarray(labels), np.asarray(probas)))


def reliability_bins(
    probas: np.ndarray, labels: np.ndarray, n_bins: int = 5
) -> pd.DataFrame:
    """Bucket mean predicted prob vs. observed frequency for reliability diagrams.

    The plan says 5-6 bins (not 10) at low n. Returns one row per bin with
    ``bin_low, bin_high, mean_prob, observed_freq, count`` so a reliability
    diagram can be drawn and the calibration-in-the-large assessed.
    """
    probas = np.asarray(probas, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Label each prediction with its bin index (right-inclusive on the last bin).
    idx = np.clip(np.digitize(probas, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        cnt = int(mask.sum())
        rows.append({
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "mean_prob": float(probas[mask].mean()) if cnt else float("nan"),
            "observed_freq": float(labels[mask].mean()) if cnt else float("nan"),
            "count": cnt,
        })
    return pd.DataFrame(rows)


def platt_scale(
    probas: np.ndarray,
    labels: np.ndarray,
    *,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> tuple[LogisticRegression, np.ndarray]:
    """Fit a Platt (sigmoid) calibrator on a held-out split; return (model, calibrated).

    Splits temporally-by-position (the inputs are assumed already sorted by time,
    as the feature frames are): the last ``val_frac`` rows form the calibration
    set, the rest fit the calibrator — never a random split, which would leak the
    future into the calibration of the past. Returns the fitted ``LogisticRegression``
    (on ``proba -> label``) and the calibrated probabilities on the full input.
    """
    probas = np.asarray(probas, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels, dtype=int)
    n = len(labels)
    cut = int(n * (1.0 - val_frac))
    fit_x, fit_y = probas[:cut], labels[:cut]
    if len(np.unique(fit_y)) < 2:
        # Degenerate calibration set (all one class): identity calibration.
        return LogisticRegression(), probas.ravel()
    cal = LogisticRegression(C=1.0, solver="lbfgs", random_state=random_state)
    cal.fit(fit_x, fit_y)
    return cal, cal.predict_proba(probas)[:, 1]
