"""Market data access for the SPY calibration experiment (Sec. 2.4.2 /
Appendix A, page 29): daily SPY bars from the QuantX Dataloader, and SPY /
SPY-option NBBO quotes and option-contract reference data from the Massive
REST API.

Downloaded data is never written inside the repository; any on-disk cache
lives under ``data/cache/`` (ignored by git) or ``/tmp``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

DATALOADER_API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
MASSIVE_API_ROOT = "https://api.massive.com"
RETRY_STATUSES = {429, 503, 504}


class DataLoaderError(RuntimeError):
    """A failed Dataloader or Massive request."""


def _dataloader_headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not key or not massive_key:
        raise DataLoaderError("DATALOADER_API_KEY and MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {key}", "X-Massive-Api-Key": massive_key}


def _massive_headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise DataLoaderError("MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {key}"}


def _request(method: str, url: str, headers: dict, params: dict | None = None,
             json_body: dict | None = None) -> dict:
    for attempt in range(3):
        resp = requests.request(method, url, headers=headers, params=params,
                                 json=json_body, timeout=60)
        if resp.status_code not in RETRY_STATUSES or attempt == 2:
            break
        time.sleep(float(resp.headers.get("Retry-After") or 2 * 2 ** attempt))
    if resp.status_code != 200:
        raise DataLoaderError(f"{url}: HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def load_spy_daily_bars(start_date: str, end_date: str) -> pd.DataFrame:
    """Daily SPY bars (dividend-adjusted) between ``start_date`` and
    ``end_date`` inclusive, used for the closing price S_0 and the
    preceding-60-day log-return volatility of Appendix A (page 29).

    Market data: QuantX Dataloader — https://quantx.limex.pro/

    Price history in this project is loaded through the QuantX Dataloader, a
    research-ready market-data service available through a REST API and as an
    MCP server for AI agents. It is built on market data from Massive
    (https://massive.com/) and prepares every stock history for backtesting:
      - prices adjusted for dividends, ready for total-return research;
      - one continuous history when a company changes its ticker;
      - clean separation when a ticker has belonged to different companies;
      - careful corporate-action handling and fine-grained data polishing.

    Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    MASSIVE_API_KEY at https://massive.com/.
    """
    body = {
        "tickers": ["SPY"],
        "timeframe": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": True,
        "yahoo_backstop": True,
    }
    answer = _request("POST", DATALOADER_API_ROOT + "/fetch-ohlcvs",
                       _dataloader_headers(), json_body=body)
    result = answer["result"]
    if result.get("data_stored"):
        raise DataLoaderError("unexpected stored (non-inline) SPY daily result")
    wide = pd.DataFrame(result["ohlcv_df"]["rows"])
    details = result.get("ticker_details") or []
    if not wide.shape[0] or any(d.get("error") for d in details):
        raise DataLoaderError(f"empty or failed SPY daily load: {details}")
    stamps = pd.to_datetime(wide["timestamp"], utc=True)
    out = pd.DataFrame({
        "date": stamps.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None),
        "open": wide["open"].astype(float),
        "high": wide["high"].astype(float),
        "low": wide["low"].astype(float),
        "close": wide["close"].astype(float),
        "volume": wide["volume"].astype(float),
    }).sort_values("date").reset_index(drop=True)
    non_massive = [d for d in details if d.get("data_source") not in (None, "massive")]
    out.attrs["ticker_details"] = details
    out.attrs["non_massive_source"] = non_massive
    return out


def fetch_option_contracts(underlying: str, expiration_date: str, as_of: str,
                            contract_type: str = "call") -> list[dict]:
    """Reference list of listed option contracts for ``underlying`` expiring on
    ``expiration_date``, as known on ``as_of`` (point-in-time chain), via the
    Massive REST options-contracts-reference endpoint.
    """
    params = {
        "underlying_ticker": underlying,
        "expiration_date": expiration_date,
        "contract_type": contract_type,
        "as_of": as_of,
        "limit": 1000,
    }
    url = MASSIVE_API_ROOT + "/v3/reference/options/contracts"
    results: list[dict] = []
    headers = _massive_headers()
    next_url = url
    next_params = params
    while next_url:
        answer = _request("GET", next_url, headers, params=next_params)
        results.extend(answer.get("results", []))
        next_url = answer.get("next_url")
        next_params = None
        if next_url and "apiKey" not in next_url:
            next_url = next_url  # next_url already carries its own query string
    return results


@dataclass
class QuoteSample:
    """Stratified NBBO-quote sample for one ticker and one trading session."""
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray


def _session_windows(date: str, n_windows: int = 6) -> list[tuple[str, str]]:
    """``n_windows`` evenly spaced 5-minute windows across the 9:30-16:00
    New York trading session on ``date``, used to subsample NBBO quotes
    instead of downloading the full session (see Deviations in RESULTS.md)."""
    day = pd.Timestamp(date, tz="America/New_York")
    open_t = day + pd.Timedelta(hours=9, minutes=30)
    close_t = day + pd.Timedelta(hours=16)
    starts = pd.date_range(open_t, close_t - pd.Timedelta(minutes=5), periods=n_windows)
    windows = []
    for s in starts:
        e = s + pd.Timedelta(minutes=5)
        windows.append((
            s.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            e.tz_convert("UTC").isoformat().replace("+00:00", "Z"),
        ))
    return windows


def fetch_quote_sample(ticker: str, date: str, is_option: bool, n_windows: int = 6,
                        limit_per_window: int = 2000) -> QuoteSample:
    """Stratified sample of NBBO quotes for ``ticker`` on ``date``, taken from
    ``n_windows`` evenly spaced 5-minute windows across the trading session
    (an approximation to the full-session median of Appendix A, used because
    downloading every NBBO quote tick for every near-expiry SPY option over
    January-June 2025 is not feasible in the time available; see
    results/RESULTS.md).
    """
    headers = _massive_headers()
    path = f"/v3/quotes/{ticker}"
    bids, asks, bid_sizes, ask_sizes = [], [], [], []
    for start_iso, end_iso in _session_windows(date, n_windows):
        params = {
            "timestamp.gte": start_iso,
            "timestamp.lt": end_iso,
            "order": "asc",
            "limit": limit_per_window,
            "sort": "timestamp",
        }
        answer = _request("GET", MASSIVE_API_ROOT + path, headers, params=params)
        for row in answer.get("results", []):
            bid = row.get("bid_price")
            ask = row.get("ask_price")
            bs = row.get("bid_size")
            as_ = row.get("ask_size")
            if bid is None or ask is None or bs is None or as_ is None:
                continue
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            bids.append(bid)
            asks.append(ask)
            bid_sizes.append(bs)
            ask_sizes.append(as_)
    if not bids:
        raise DataLoaderError(f"no usable NBBO quotes for {ticker} on {date}")
    return QuoteSample(
        bid=np.asarray(bids, dtype=float), ask=np.asarray(asks, dtype=float),
        bid_size=np.asarray(bid_sizes, dtype=float), ask_size=np.asarray(ask_sizes, dtype=float),
    )
