"""Resolve Polymarket's NFL tag id(s) for event filtering.

The Gamma API filters events by numeric tag id, but tag ids are not stable
across environments, so we resolve them dynamically from the slug. We also pull
the ``/sports`` mapping as a cross-check and to surface related tag ids (e.g.
"NFL Playoffs") that may also carry game markets.
"""
from __future__ import annotations

from typing import Any

from .. import config, io
from .client import PolymarketClient


def resolve_nfl_tag(client: PolymarketClient) -> dict[str, Any]:
    """Fetch the NFL tag object ``{id, slug, label}`` from Gamma."""
    tag = client.get_json(f"/tags/slug/{config.NFL_TAG_SLUG}")
    if not isinstance(tag, dict) or "id" not in tag:
        raise RuntimeError(
            f"Unexpected response from /tags/slug/{config.NFL_TAG_SLUG}: {tag!r}"
        )
    return tag


def fetch_sports_mapping(client: PolymarketClient) -> list[dict[str, Any]]:
    """Fetch the ``/sports`` sport→tags mapping (used for cross-checking)."""
    data = client.get_json("/sports")
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data if isinstance(data, list) else []


def resolve_and_cache(client: PolymarketClient) -> dict[str, Any]:
    """Resolve the NFL tag and persist it (plus the sports map) for reproducibility.

    Returns the NFL tag dict. The sports mapping is written alongside it so the
    EDA notebook can show which tag ids belong to the NFL ecosystem.
    """
    nfl_tag = resolve_nfl_tag(client)
    sports = fetch_sports_mapping(client)

    payload = {"nfl_tag": nfl_tag, "sports": sports}
    io.write_json(payload, config.POLYMARKET_RAW_DIR / "tags.json")
    return payload
