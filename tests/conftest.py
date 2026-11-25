"""Shared pytest fixtures and marker registration."""
from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that call the QuantX Dataloader / Massive REST APIs"
    )


@pytest.fixture(scope="session")
def spy_bars_slice(request: pytest.FixtureRequest):
    """A short, cached slice of SPY daily bars for market-data tests.

    Skips when DATALOADER_API_KEY / MASSIVE_API_KEY are absent, per the
    massive-rest-market-data skill's rule that tests do not depend on a live
    call by default.
    """
    if not (os.environ.get("DATALOADER_API_KEY") and os.environ.get("MASSIVE_API_KEY")):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    cache = request.config.cache
    cached = cache.get("spy_bars_slice/2025-01", None)
    if cached is not None:
        import pandas as pd
        return pd.DataFrame(cached)
    from data.data_loader import load_spy_daily_bars
    bars = load_spy_daily_bars("2025-01-02", "2025-02-28")
    cache.set("spy_bars_slice/2025-01", bars.assign(date=bars["date"].astype(str)).to_dict("list"))
    return bars
