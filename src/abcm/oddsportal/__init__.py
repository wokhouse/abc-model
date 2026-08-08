"""OddsPortal historical NFL odds ingestion.

Scrapes opening + closing sportsbook lines (spread, total, moneyline) for
2024-2025 NFL games via the OddsHarvester library (Playwright-based, handles
OddsPortal's Cloudflare/JS rendering). This fills the gap left by the
sportsbookreviewsonline.com archive (frozen at 2021-22) and the single-line
Kaggle dataset: genuine opening AND closing lines for the Polymarket seasons.

Modules:
* ``teams``  — OddsPortal full team names -> nflverse abbrs.
* ``loader`` — wraps OddsHarvester with per-game caching + resume.
* ``ingest`` — orchestrates the pull and normalizes to one row per game.
"""
