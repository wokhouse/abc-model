"""Historical NFL odds from the sportsbookreviewsonline.com archive (2011-2021).

The pre-scraped ``nfl_archive_10Y.json`` ships 2,956 games (2011-2021) with
opening AND closing spreads and totals — the same open/close signal we are
slowly scraping from OddsPortal for 2024-2025. This module loads + validates it
for a *hypothesis test*: does open->close line movement predict outcomes on a
well-powered historical sample, before committing to the long OddsPortal scrape?

It does NOT overlap our 2024-2025 Polymarket markets, so it cannot directly
improve Stage 2's predictions on those markets — it's a methodology de-risker,
not new market signal.
"""
