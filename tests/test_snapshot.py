"""Tests for pre-game price snapshot + open→close delta materialization.

The snapshot logic reads per-market candle files; we build a temp candle
directory to exercise the cutoff / fallback / empty cases without touching the
real data dir.
"""
from __future__ import annotations

import pandas as pd
import pytest

from abcm.reconcile import snapshot as snap_mod


def _make_candles(rows):
    """rows: list of (ts_str, open, high, low, close)."""
    return pd.DataFrame(
        rows, columns=["ts", "open", "high", "low", "close"]
    ).assign(n_trades=1, volume=1.0)


@pytest.fixture
def candle_dir(tmp_path, monkeypatch):
    """Point prices.candles_for() at a temp dir and write candle files."""
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()

    def _write(condition_id, candles):
        d = prices_dir / condition_id
        d.mkdir(parents=True, exist_ok=True)
        candles.to_parquet(d / "candles.parquet", index=False)

    # Monkeypatch pm_prices.candles_for / candles_path to read from temp dir.
    from abcm.polymarket import prices as pm_prices

    def fake_candles_for(condition_id):
        p = prices_dir / str(condition_id) / "candles.parquet"
        if not p.exists():
            return pd.DataFrame(
                columns=["ts", "open", "high", "low", "close", "n_trades", "volume"]
            )
        return pd.read_parquet(p)

    monkeypatch.setattr(pm_prices, "candles_for", fake_candles_for)
    return _write


GAME_TIME = pd.Timestamp("2024-09-15 18:00:00")


def test_snapshot_cutoff_takes_last_close_before_game(candle_dir):
    """The snapshot is the last close at or before T-1h."""
    cond = "c1"
    candle_dir(cond, _make_candles([
        ("2024-09-14 13:00:00+00:00", 0.50, 0.50, 0.49, 0.49),
        ("2024-09-15 16:00:00+00:00", 0.51, 0.57, 0.51, 0.57),  # 2h before game
    ]))
    snap = snap_mod.snapshot_price(cond, GAME_TIME)
    assert snap["method"] == "cutoff"
    assert snap["snapshot_price"] == pytest.approx(0.57)
    assert snap["n_candles"] == 2


def test_snapshot_open_and_delta(candle_dir):
    """Open price is the first candle's open; delta = close - open."""
    cond = "c2"
    candle_dir(cond, _make_candles([
        ("2024-09-14 13:00:00+00:00", 0.51, 0.51, 0.49, 0.49),  # open 0.51
        ("2024-09-15 16:00:00+00:00", 0.51, 0.57, 0.51, 0.57),  # close 0.57
    ]))
    snap = snap_mod.snapshot_price(cond, GAME_TIME)
    assert snap["open_price"] == pytest.approx(0.51)
    assert snap["snapshot_price"] == pytest.approx(0.57)
    assert snap["price_delta"] == pytest.approx(0.57 - 0.51)


def test_snapshot_single_candle_delta_zero(candle_dir):
    """A one-candle market yields delta 0 (open == close)."""
    cond = "c3"
    candle_dir(cond, _make_candles([
        ("2024-09-15 16:00:00+00:00", 0.55, 0.55, 0.55, 0.55),
    ]))
    snap = snap_mod.snapshot_price(cond, GAME_TIME)
    assert snap["n_candles"] == 1
    assert snap["price_delta"] == pytest.approx(0.0)
    assert snap["open_price"] == pytest.approx(0.55)


def test_snapshot_empty_candles_returns_no_history(candle_dir):
    """An empty candle file yields no_history with n_candles 0."""
    cond = "c4"
    # Write an empty file (schema only).
    candle_dir(cond, pd.DataFrame(
        columns=["ts", "open", "high", "low", "close", "n_trades", "volume"]
    ))
    snap = snap_mod.snapshot_price(cond, GAME_TIME)
    assert snap["has_history"] is False
    assert snap["method"] == "no_history"
    assert snap["n_candles"] == 0
    assert snap["open_price"] is None
    assert snap["price_delta"] is None


def test_snapshot_missing_condition_id():
    snap = snap_mod.snapshot_price(None, GAME_TIME)
    assert snap["method"] == "no_condition_id"
    assert snap["snapshot_price"] is None


def test_snapshot_no_game_time_falls_back_to_last(candle_dir):
    """Without a game time, the last available close is used."""
    cond = "c5"
    candle_dir(cond, _make_candles([
        ("2024-09-14 13:00:00+00:00", 0.40, 0.40, 0.40, 0.40),
        ("2024-09-15 16:00:00+00:00", 0.50, 0.60, 0.50, 0.60),
    ]))
    snap = snap_mod.snapshot_price(cond, None)
    assert snap["method"] == "no_game_time"
    assert snap["snapshot_price"] == pytest.approx(0.60)


def test_snapshot_history_starts_after_cutoff(candle_dir):
    """If all candles are after the cutoff, take the earliest candle's close."""
    cond = "c6"
    # Game at 2024-09-15 18:00; cutoff is 17:00. First candle at 17:30.
    candle_dir(cond, _make_candles([
        ("2024-09-15 17:30:00+00:00", 0.52, 0.52, 0.52, 0.52),
        ("2024-09-15 17:45:00+00:00", 0.55, 0.55, 0.55, 0.55),
    ]))
    snap = snap_mod.snapshot_price(cond, GAME_TIME)
    assert snap["method"] == "last_fallback"
    assert snap["snapshot_price"] == pytest.approx(0.52)


def test_add_price_snapshots_writes_new_columns(candle_dir):
    """The batch function emits open_price / price_delta / n_candles.

    Kickoff = gameday 2024-09-15 + gametime 18:00 ET = 2024-09-15 22:00 UTC;
    cutoff (T-1h) = 21:00 UTC. The 16:00 UTC candle is the last <= cutoff.
    """
    cond = "c7"
    candle_dir(cond, _make_candles([
        ("2024-09-14 13:00:00+00:00", 0.50, 0.50, 0.49, 0.49),  # open 0.50
        ("2024-09-15 16:00:00+00:00", 0.49, 0.55, 0.49, 0.55),  # <= cutoff, close 0.55
        ("2024-09-15 23:00:00+00:00", 0.55, 0.60, 0.55, 0.60),  # after kickoff
    ]))
    markets = pd.DataFrame([{
        "condition_id": cond,
        "game_id": "g1",
        "market_end": "2024-09-15T22:00:00Z",
    }])
    schedules = pd.DataFrame([{
        "game_id": "g1",
        "gameday": "2024-09-15",
        "gametime": "18:00",  # ET -> 22:00 UTC; cutoff 21:00 UTC
    }])
    out = snap_mod.add_price_snapshots(markets, schedules)
    row = out.iloc[0]
    assert row["n_candles"] == 3
    # No yes_token_id => falls back to first candle's open (0.50).
    assert row["open_price"] == pytest.approx(0.50)
    assert row["snapshot_price"] == pytest.approx(0.55)  # 16:00 close <= 21:00 cutoff
    assert row["price_delta"] == pytest.approx(0.55 - 0.50)


def test_open_price_uses_first_trade_when_token_given(candle_dir, monkeypatch):
    """With yes_token_id, open_price is the first actual trade (tick-level)."""
    from abcm.polymarket import prices as pm_prices

    cond = "c8"
    yes_token = "yes-token-123"
    candle_dir(cond, _make_candles([
        ("2024-09-14 13:00:00+00:00", 0.50, 0.50, 0.49, 0.49),  # candle open 0.50
    ]))
    # Provide a trades_for stub where the first Yes trade is at 0.44 — distinct
    # from the candle open (0.50), proving we read the tick, not the candle.
    trades = pd.DataFrame([
        {"timestamp": int(pd.Timestamp("2024-09-14 12:30:00").timestamp()),
         "price": 0.44, "asset": yes_token, "side": "BUY", "size": 100.0,
         "outcome": "Yes", "outcomeIndex": 0},
        {"timestamp": int(pd.Timestamp("2024-09-14 13:05:00").timestamp()),
         "price": 0.49, "asset": yes_token, "side": "BUY", "size": 50.0,
         "outcome": "Yes", "outcomeIndex": 0},
    ])
    monkeypatch.setattr(pm_prices, "trades_for", lambda cid: trades)

    snap = snap_mod.snapshot_price(cond, GAME_TIME, yes_token_id=yes_token)
    assert snap["open_price"] == pytest.approx(0.44)  # first trade, not candle 0.50


def test_open_price_falls_back_to_candle_when_no_trades(candle_dir):
    """Without a yes_token_id, open_price falls back to the first candle open."""
    cond = "c9"
    candle_dir(cond, _make_candles([
        ("2024-09-14 13:00:00+00:00", 0.50, 0.50, 0.49, 0.49),
    ]))
    snap = snap_mod.snapshot_price(cond, GAME_TIME)  # no yes_token_id
    assert snap["open_price"] == pytest.approx(0.50)


# --- Kickoff-time reconstruction ------------------------------------------


def test_kickoff_utc_converts_eastern_to_utc():
    """20:20 ET -> 00:20 UTC the next day."""
    ko = snap_mod.kickoff_utc("2024-09-05", "20:20")
    assert ko == pd.Timestamp("2024-09-06 00:20:00")


def test_kickoff_utc_handles_dst_boundary():
    """A September game (EDT, UTC-4) vs a December game (EST, UTC-5).

    13:00 ET in September -> 17:00 UTC; 13:00 ET in December -> 18:00 UTC.
    """
    sep = snap_mod.kickoff_utc("2024-09-15", "13:00")
    dec = snap_mod.kickoff_utc("2024-12-15", "13:00")
    assert sep == pd.Timestamp("2024-09-15 17:00:00")  # EDT
    assert dec == pd.Timestamp("2024-12-15 18:00:00")  # EST


def test_kickoff_utc_returns_none_for_missing_fields():
    assert snap_mod.kickoff_utc(None, "13:00") is None
    assert snap_mod.kickoff_utc("2024-09-15", None) is None
    assert snap_mod.kickoff_utc("2024-09-15", "TBD") is None  # no colon


def test_add_price_snapshots_uses_true_kickoff_not_midnight(candle_dir):
    """Regression: a gameday-only cutoff (midnight) is ~17-23h too early.

    With kickoff 2024-09-15 18:00 ET = 22:00 UTC and T-1h cutoff = 21:00 UTC,
    a candle at 20:00 UTC is the pre-game snapshot (it would be AFTER a
    midnight-based cutoff of 2024-09-14 23:00... wait, 20:00 Sep15 is after
    23:00 Sep14). The point: the snapshot lands near kickoff, not ~18h early.
    """
    cond = "c10"
    candle_dir(cond, _make_candles([
        ("2024-09-14 23:00:00+00:00", 0.40, 0.40, 0.40, 0.40),  # ~23h before kickoff
        ("2024-09-15 20:00:00+00:00", 0.40, 0.55, 0.40, 0.55),  # ~2h before kickoff
    ]))
    markets = pd.DataFrame([{
        "condition_id": cond, "game_id": "g1",
        "market_end": "2024-09-15T22:00:00Z",
    }])
    schedules = pd.DataFrame([{
        "game_id": "g1", "gameday": "2024-09-15", "gametime": "18:00",
    }])
    out = snap_mod.add_price_snapshots(markets, schedules)
    row = out.iloc[0]
    # The T-1h cutoff (21:00 UTC) picks the 20:00 candle (close 0.55), NOT the
    # 23:00-previous-day candle (0.40) that a midnight cutoff would have picked.
    assert row["snapshot_price"] == pytest.approx(0.55)
    assert row["snapshot_method"] == "cutoff"
