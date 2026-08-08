"""Hypothesis test: does open->close line movement predict ATS outcomes?

Uses the SBR historical archive (2,386 clean games, 2011-2021) to answer the
question that justifies the slow OddsPortal scrape: is line movement a real
signal, or noise? This is a methodology de-risker — it does NOT touch our
2024-2025 Polymarket markets.

The "smart money" hypothesis: when the spread moves toward the favorite (the
favorite's number gets bigger in absolute terms -> market backed the favorite),
the favorite should cover more often than 50%. If this holds with a real edge
and tight CI, the OddsPortal scrape is worth finishing. If movement is centered
on no-edge with a wide CI, we stop the scrape and save ~19h.

Usage:
    uv run python -m scripts.analyze_sbr_movement
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from abcm import io, config
from abcm.sbr import loader


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([values[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cover_rate_by_bucket(df: pd.DataFrame, movement: str, buckets: list[tuple]) -> None:
    """Print cover rate (did the side the line moved toward cover?) per bucket."""
    print(f"\n  Cover rate by {movement} bucket (did the backed side cover?):")
    print(f"  {'bucket':>16s} {'n':>6s} {'cover_rate':>11s} {'95% CI':>20s} {'vs 50%':>8s}")
    for lo, hi in buckets:
        sub = df[(df[movement] >= lo) & (df[movement] < hi)]
        if len(sub) == 0:
            continue
        covers = sub["_backed_side_covered"].to_numpy()
        m, lo_ci, hi_ci = _bootstrap_ci(covers)
        star = "*" if not (lo_ci <= 0.5 <= hi_ci) else ""
        print(f"  [{lo:+.1f}, {hi:+.1f})  {len(sub):>6d} {m:>11.3f}  [{lo_ci:.3f}, {hi_ci:.3f}] {star:>8s}")


def main() -> None:
    lines = loader.load_lines(validate=True)
    sched = io.read_parquet(config.NFLVERSE_HISTORY_DIR / "schedules.parquet")
    df = loader.join_game_ids(lines, sched)
    print(f"[sbr-movement] {len(df)} clean games with open/close spreads, "
          f"{df.season.min()}-{df.season.max()}")

    # --- Construct "did the side the line moved toward cover?" ---------------
    # spread_movement = home_close - home_open. Negative => the home team's
    # spread got more negative (home became a bigger favorite) => money on home.
    # Positive => money on away (home's number shrank).
    # The "backed side" covered iff:
    #   - movement < 0 (home backed) and home_covered_close == 1, OR
    #   - movement > 0 (away backed) and home_covered_close == 0
    backed_home = df["spread_movement"] < 0
    backed_away = df["spread_movement"] > 0
    df["_backed_side_covered"] = (
        (backed_home & (df["home_covered_close"] == 1)) |
        (backed_away & (df["home_covered_close"] == 0))
    ).astype(int)

    # Only games where the line actually moved (exclude no-movement games).
    moved = df[df["spread_movement"] != 0].copy()
    print(f"[sbr-movement] {len(moved)} games with non-zero spread movement")

    overall = moved["_backed_side_covered"].to_numpy()
    m, lo, hi = _bootstrap_ci(overall)
    print(f"\n=== OVERALL: did the backed side cover? ===")
    print(f"  n={len(overall)}  cover_rate={m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  vs 50% baseline: edge = {m-0.5:+.3f}  {'SIGNIFICANT*' if not (lo<=0.5<=hi) else 'not significant'}")
    print(f"  (50% = no signal; >50% = line movement predicts covers)")

    # --- By movement magnitude (bigger moves should be stronger signal) ------
    print(f"\n=== By movement magnitude (|spread_movement|) ===")
    abs_mv = moved["spread_movement"].abs()
    for lo_b, hi_b, label in [(0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0"),
                               (2.0, 3.5, "2.0-3.5"), (3.5, 99, "3.5+")]:
        sub = moved[(abs_mv >= lo_b) & (abs_mv < hi_b)]
        if len(sub) == 0:
            continue
        cov = sub["_backed_side_covered"].to_numpy()
        mm, llo, hhi = _bootstrap_ci(cov)
        sig = "*" if not (llo <= 0.5 <= hhi) else ""
        print(f"  |move| {label:>7s}: n={len(sub):>5d}  cover={mm:.3f}  [{llo:.3f}, {hhi:.3f}] {sig}")

    # --- By direction: favorite money vs underdog money ----------------------
    print(f"\n=== By direction ===")
    fav_backed = moved[backed_home[moved.index]] if "home_open_spread" in moved else None
    # home is favorite when home_open_spread < 0; money on home (movement<0) backs the favorite.
    home_fav = moved["home_open_spread"] < 0
    fav_money = moved[home_fav & (moved["spread_movement"] < 0)]
    dog_money = moved[~home_fav & (moved["spread_movement"] > 0)]
    for label, sub in [("favorite money (line got bigger)", fav_money),
                       ("underdog money (line shrank)", dog_money)]:
        if len(sub) == 0:
            continue
        cov = sub["_backed_side_covered"].to_numpy()
        mm, llo, hhi = _bootstrap_ci(cov)
        sig = "*" if not (llo <= 0.5 <= hhi) else ""
        print(f"  {label}: n={len(sub):>5d}  backed-side cover={mm:.3f}  [{llo:.3f}, {hhi:.3f}] {sig}")

    # --- Total movement: does over/under movement predict the over? ----------
    print(f"\n=== Total line movement -> does the over hit? ===")
    tm = df[df["total_movement"] != 0].copy()
    total_score = tm["home_score"] + tm["away_score"]
    over_hit_close = (total_score > tm["close_over_under"]).astype(int)
    tm["_over_hit"] = over_hit_close
    # When total moved UP (over money), does the over hit more than 50%?
    over_backed = tm[tm["total_movement"] > 0]
    under_backed = tm[tm["total_movement"] < 0]
    for label, sub in [("total moved UP (over money)", over_backed),
                       ("total moved DOWN (under money)", under_backed)]:
        if len(sub) == 0:
            continue
        ov = sub["_over_hit"].to_numpy()
        mm, llo, hhi = _bootstrap_ci(ov)
        sig = "*" if not (llo <= 0.5 <= hhi) else ""
        print(f"  {label}: n={len(sub):>5d}  over_rate={mm:.3f}  [{llo:.3f}, {hhi:.3f}] {sig}")

    print(f"\n=== VERDICT ===")
    edge = m - 0.5
    if not (lo <= 0.5 <= hi) and edge > 0:
        print(f"  Spread movement IS predictive (backed side covers {m:.1%}, CI excludes 50%).")
        print(f"  -> The OddsPortal scrape for 2024-2025 is worth finishing.")
    elif not (lo <= 0.5 <= hi) and edge < 0:
        print(f"  Spread movement is CONTRARIAN (backed side covers only {m:.1%}).")
        print(f"  -> Fade-the-move signal; still worth scraping, opposite direction.")
    else:
        print(f"  Spread movement shows NO clear edge ({m:.1%}, CI [{lo:.3f},{hi:.3f}] includes 50%).")
        print(f"  -> Consider stopping the OddsPortal scrape; signal not confirmed on history.")


if __name__ == "__main__":
    main()
