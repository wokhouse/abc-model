"""HTTP client for Polymarket's public APIs (Gamma + CLOB).

Both APIs are public and rate-limited, so this client wraps an ``httpx.Client``
with exponential backoff on 429 / 5xx responses. Base URLs come from config so
tests can point at a stub.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from .. import config


class PolymarketAPIError(Exception):
    """Raised when a request exhausts its retries or returns a non-2xx status."""


class PolymarketClient:
    """Synchronous JSON client with retry/backoff for Polymarket endpoints."""

    # Retry on these statuses (429 = rate limited, 5xx = transient server error).
    _RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str | None = None,
        *,
        max_retries: int | None = None,
        timeout: float | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or config.GAMMA_BASE_URL).rstrip("/")
        self.max_retries = max_retries if max_retries is not None else config.HTTP_MAX_RETRIES
        self.timeout = timeout if timeout is not None else config.HTTP_TIMEOUT
        # Allow callers (tests) to inject a pre-configured client.
        self._client = client or httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> PolymarketClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_json(self, path_or_url: str, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON endpoint with retries.

        ``path_or_url`` may be either a path (joined to ``base_url``) or an
        absolute URL (used as-is).
        """
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"

        attempt = 0
        while True:
            try:
                resp = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                # Network blips are treated like a retryable 5xx.
                if attempt >= self.max_retries:
                    raise PolymarketAPIError(
                        f"Transport error after {attempt} retries: {exc}"
                    ) from exc
                attempt += 1
                self._sleep_backoff(attempt)
                continue

            if resp.status_code in self._RETRY_STATUS and attempt < self.max_retries:
                attempt += 1
                self._sleep_backoff(attempt, resp=resp)
                continue

            if resp.status_code != 200:
                raise PolymarketAPIError(
                    f"{resp.status_code} {resp.reason_phrase} for {url}"
                )

            return resp.json()

    def _sleep_backoff(self, attempt: int, *, resp: httpx.Response | None = None) -> None:
        """Exponential backoff. Honors Retry-After when the server provides it."""
        if resp is not None:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(min(int(retry_after), 30))
                return
        # 0.5s, 1s, 2s, 4s, ... capped at 30s.
        delay = min(0.5 * (2 ** (attempt - 1)), 30.0)
        time.sleep(delay)
