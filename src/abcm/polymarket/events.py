"""Fetch and flatten resolved NFL events from Polymarket's Gamma API.

A Polymarket *event* (e.g. "Super Bowl LIX") contains one or more *markets*
(e.g. "Chiefs vs Eagles", "Chiefs -3.5", "Game total over 47.5"). Each market is
a tradable binary outcome with two CLOB token ids — one for "Yes", one for "No".

This module pages all *closed* NFL events, flattens them to one row per market,
and classifies each market's type (moneyline / spread / totals / props / futures)
so later phases can filter to the modeling target without re-pulling raw data.

Resolution is encoded in ``outcomePrices``: for ``outcomes=["Yes","No"]`` a
resolved Yes-win market has ``outcomePrices=["1","0"]`` and a Yes-loss market has
``["0","1"]``. We capture the resolved Yes-price as ``resolved_yes_price``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterator

import pandas as pd

from .. import config
from .client import PolymarketClient

# Pagination — Gamma caps page size; we page until an empty/short page arrives.
_PAGE = config.EVENTS_PAGE_SIZE


def fetch_all_closed_events(client: PolymarketClient, tag_id: str) -> list[dict[str, Any]]:
    """Page through every closed event under ``tag_id`` and return raw event dicts."""
    all_events: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = client.get_json(
            "/events",
            params={"tag_id": tag_id, "closed": "true", "limit": _PAGE, "offset": offset},
        )
        if not isinstance(page, list) or not page:
            break
        all_events.extend(page)
        if len(page) < _PAGE:
            break
        offset += _PAGE
    return all_events


# --- Market type classification --------------------------------------------

# Order matters: check more specific patterns before generic ones.
_SPREAD_RE = re.compile(r"[-+]\s?\d+\.?\d*|handicap|spread", re.IGNORECASE)
_TOTALS_RE = re.compile(r"\bover\b|\bunder\b|\btotal\b|o/u", re.IGNORECASE)
_PLAYER_PROP_RE = re.compile(
    r"\b(passing|rushing|receiving) yards\b|\byards\b|\btd(s)?\b|"
    r"\breceptions\b|\bcarries\b|\bcompletions\b|\binterceptions\b|"
    r"\bplayer\b",
    re.IGNORECASE,
)
_HALF_RE = re.compile(r"\b1st half\b|\b2nd half\b|\bhalftime\b|\bfirst half\b", re.IGNORECASE)


def classify_market(
    question: str | None,
    *,
    event_title: str | None = None,
    outcomes: list[str] | None = None,
) -> str:
    """Best-effort market-type label from the question/event/outcomes.

    Returns one of: moneyline, spread, totals, player_props, halftime, futures,
    other. Classification is heuristic — it drives filtering, not modeling, so
    false positives just get reviewed in the EDA rather than breaking anything.

    The strongest moneyline signal is *two team-name outcomes* (e.g.
    ``["Texans", "Bears"]``); a bare "vs" in the question is insufficient because
    many prop questions ("highest scoring game") also contain "vs".
    """
    text = f"{question or ''} {event_title or ''}".strip()
    if not text:
        return "other"

    # Two non-Yes/No outcomes that name teams => definitive game moneyline.
    # (Team-name validity is validated later in reconcile.teams; here we only
    # require that both outcomes are neither Yes/No nor Over/Under.)
    if outcomes and len(outcomes) == 2:
        lowered = {str(o).strip().lower() for o in outcomes}
        if not (lowered & {"yes", "no", "over", "under"}):
            # Still let spread/total suffixes on the outcome win out.
            if not any(_SPREAD_RE.search(str(o)) for o in outcomes):
                return "moneyline"

    if _HALF_RE.search(text):
        return "halftime"
    if _PLAYER_PROP_RE.search(text):
        return "player_props"
    if _SPREAD_RE.search(text):
        return "spread"
    if _TOTALS_RE.search(text):
        return "totals"
    # Single-team moneyline phrasings: "X to win", "X beat Y", "X defeat Y".
    # A bare "vs" is intentionally NOT enough — too many prop questions use it
    # (e.g. "Will Chiefs vs Ravens be the highest scoring game?").
    if re.search(r"\bto win\b|\bbeat\b|\bdefeat\b", text, re.IGNORECASE):
        return "moneyline"
    if re.search(r"\bchampion\b|\bwin (the )?super bowl\b|\bwin the\b|\bdivision winner\b",
                 text, re.IGNORECASE):
        return "futures"
    return "other"


# --- Flattening ------------------------------------------------------------

def _parse_clob_token_ids(raw: Any) -> list[str]:
    """``clobTokenIds`` is a JSON-encoded string like '["id1", "id2"]'."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _parse_outcome_prices(raw: Any) -> list[float]:
    """``outcomePrices`` is a JSON-encoded string of numeric strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return [float(x) for x in raw]
    return []


def _parse_outcomes(raw: Any) -> list[str]:
    """``outcomes`` is a JSON-encoded string like '["Yes", "No"]' or '["Texans", "Bears"]'."""
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _market_row(event: dict[str, Any], market: dict[str, Any]) -> dict[str, Any]:
    """Project a single market into a flat row."""
    tokens = _parse_clob_token_ids(market.get("clobTokenIds"))
    prices = _parse_outcome_prices(market.get("outcomePrices"))
    yes_token = tokens[0] if len(tokens) >= 1 else None
    no_token = tokens[1] if len(tokens) >= 2 else None
    # Yes resolution = outcomePrices[0] on resolved markets (outcomes=["Yes","No"]).
    resolved_yes_price = prices[0] if prices else None

    question = market.get("question")
    event_title = event.get("title")
    outcomes = _parse_outcomes(market.get("outcomes"))

    return {
        "event_id": event.get("id"),
        "event_slug": event.get("slug"),
        "event_title": event_title,
        "event_start": event.get("startDate"),
        "event_end": event.get("endDate"),
        "event_closed_time": event.get("closedTime"),
        "market_id": market.get("id"),
        "condition_id": market.get("conditionId"),
        "market_slug": market.get("slug"),
        "question": question,
        "market_type": classify_market(question, event_title=event_title, outcomes=outcomes),
        "yes_token_id": yes_token,
        "no_token_id": no_token,
        "outcomes": market.get("outcomes"),
        "resolved_yes_price": resolved_yes_price,
        "volume": market.get("volumeNum"),
        "liquidity": market.get("liquidityNum"),
        "best_bid": market.get("bestBid"),
        "best_ask": market.get("bestAsk"),
        "last_trade_price": market.get("lastTradePrice"),
        "market_start": market.get("startDate"),
        "market_end": market.get("endDate"),
        "market_closed_time": market.get("closedTime"),
    }


def flatten_markets(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten all events into one row per market."""
    rows = [_market_row(ev, m) for ev in events for m in (ev.get("markets") or [])]
    return pd.DataFrame(rows)


def events_to_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Project event-level metadata (one row per event) for provenance."""
    rows = [
        {
            "event_id": ev.get("id"),
            "event_slug": ev.get("slug"),
            "event_title": ev.get("title"),
            "event_start": ev.get("startDate"),
            "event_end": ev.get("endDate"),
            "event_closed_time": ev.get("closedTime"),
            "n_markets": len(ev.get("markets") or []),
            "volume": ev.get("volume"),
            "tags": [t.get("slug") for t in (ev.get("tags") or []) if isinstance(t, dict)],
        }
        for ev in events
    ]
    return pd.DataFrame(rows)
