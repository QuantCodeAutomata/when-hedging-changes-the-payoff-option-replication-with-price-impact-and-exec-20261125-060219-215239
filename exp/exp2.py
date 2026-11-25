"""Experiment 2: calibration of the one-period binomial model to SPY call
option market data (Sec. 2.4.2, Appendix A, page 29).

For a sample of near-expiry SPY call options, the preceding trading
session's median NBBO quoted spreads and median quoted sizes of SPY and of
each option are used to calibrate the price-impact/execution-cost parameter
phi (midpoint regime lambda = 2*phi), and the model-implied bid-ask spread
of the option is compared with the observed spread, across moneyness bands.

Run: python -m exp.exp2
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import (
    DataLoaderError,
    fetch_option_contracts,
    fetch_quote_sample,
    load_spy_daily_bars,
)
from src.calibration import (
    calibrate_phi_midpoint,
    daily_log_return_std,
    explained_spread_pct,
    implied_contracts,
    implied_n,
    implied_phi,
    model_implied_option_spread,
    up_down_prices,
)
from src.config import (
    EXP2_END_DATE,
    EXP2_LOOKBACK_DAYS,
    EXP2_MONEYNESS_BANDS,
    EXP2_N_EXPIRIES_PAPER,
    EXP2_N_EXPIRIES_RUN,
    EXP2_N_OPTIONS_PAPER,
    EXP2_START_DATE,
    EXP2_UNDERLYING,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
# Target moneyness offsets |S_0/K - 1| sampled for every expiry, spanning the
# bands of Table 1 (0.5%, 1%, 2%, 5%, all strikes).
MONEYNESS_TARGETS = (0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08)
N_WINDOWS = 3          # 5-minute NBBO sampling windows per trading session
LIMIT_PER_WINDOW = 300  # quotes retrieved per window


def previous_trading_day(bars: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    """The last SPY trading session strictly before ``date``."""
    prior = bars.loc[bars["date"] < date, "date"]
    return None if prior.empty else prior.max()


def select_expiry_candidates(bars: pd.DataFrame, n_expiries: int) -> list[pd.Timestamp]:
    """``n_expiries`` SPY trading sessions evenly spaced across the sample
    period, used as candidate option-expiration dates (SPY lists call
    options expiring on every trading day); each candidate is verified
    against the listed option chain in ``run_calibration``."""
    sample = bars[(bars["date"] >= EXP2_START_DATE) & (bars["date"] <= EXP2_END_DATE)]
    days = sample["date"].tolist()
    assert len(days) > 0, "no SPY trading sessions found in the sample period"
    idx = np.linspace(0, len(days) - 1, min(n_expiries, len(days))).astype(int)
    return [days[i] for i in sorted(set(idx))]


def run_calibration(n_expiries: int = EXP2_N_EXPIRIES_RUN,
                     moneyness_targets: tuple[float, ...] = MONEYNESS_TARGETS) -> dict:
    """Run the calibration of Appendix A over a sample of near-expiry SPY
    call options and aggregate the model-implied/observed spread comparison
    by moneyness band (Table 1, page 8)."""
    lookback_start = (pd.Timestamp(EXP2_START_DATE) - pd.Timedelta(days=4 * EXP2_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    bars = load_spy_daily_bars(lookback_start, EXP2_END_DATE)
    print(f"SPY daily bars: {bars.shape[0]} rows, {bars['date'].min()} to {bars['date'].max()}")

    expiry_candidates = select_expiry_candidates(bars, n_expiries)
    records: list[dict] = []
    skipped: list[dict] = []

    for expiry_ts in expiry_candidates:
        expiry = expiry_ts.strftime("%Y-%m-%d")
        t_i = previous_trading_day(bars, expiry_ts)
        if t_i is None:
            skipped.append({"expiry": expiry, "reason": "no preceding trading session"})
            continue
        t_i_str = t_i.strftime("%Y-%m-%d")
        lookback = bars[bars["date"] <= t_i].tail(EXP2_LOOKBACK_DAYS + 1)
        if lookback.shape[0] < EXP2_LOOKBACK_DAYS + 1:
            skipped.append({"expiry": expiry, "reason": "insufficient lookback history"})
            continue

        s0 = float(bars.loc[bars["date"] == t_i, "close"].iloc[0])
        sigma_d = daily_log_return_std(lookback["close"].to_numpy())
        st_up, st_down = up_down_prices(s0, sigma_d)

        try:
            contracts = fetch_option_contracts(EXP2_UNDERLYING, expiry, t_i_str, "call")
        except DataLoaderError as exc:
            skipped.append({"expiry": expiry, "reason": f"contracts fetch failed: {exc}"})
            continue
        if not contracts:
            skipped.append({"expiry": expiry, "reason": "no listed call contracts"})
            continue
        strikes = sorted({float(c["strike_price"]) for c in contracts})

        try:
            underlying_quotes = fetch_quote_sample(
                EXP2_UNDERLYING, t_i_str, is_option=False,
                n_windows=N_WINDOWS, limit_per_window=LIMIT_PER_WINDOW,
            )
        except DataLoaderError as exc:
            skipped.append({"expiry": expiry, "reason": f"underlying quotes failed: {exc}"})
            continue
        # S^S is defined as half the median quoted spread (Appendix A, page 29);
        # calibrate_phi_midpoint applies the formula's own additional 1/2 factor.
        half_spread_u = 0.5 * float(np.median(underlying_quotes.ask - underlying_quotes.bid))
        vol_ask_u = float(np.median(underlying_quotes.ask_size))
        vol_bid_u = float(np.median(underlying_quotes.bid_size))
        phi = calibrate_phi_midpoint(half_spread_u, vol_ask_u, vol_bid_u)
        lam = 2.0 * phi

        seen_strikes: set[float] = set()
        for target in moneyness_targets:
            k_target = s0 * (1.0 + target)
            k = min(strikes, key=lambda x: abs(x - k_target))
            if k in seen_strikes:
                continue
            seen_strikes.add(k)
            moneyness = abs(s0 / k - 1.0)
            ticker = next((c["ticker"] for c in contracts if float(c["strike_price"]) == k), None)
            if ticker is None:
                continue
            try:
                opt_quotes = fetch_quote_sample(
                    ticker, t_i_str, is_option=True,
                    n_windows=N_WINDOWS, limit_per_window=LIMIT_PER_WINDOW,
                )
            except DataLoaderError as exc:
                skipped.append({"expiry": f"{expiry}:{ticker}", "reason": f"option quotes failed: {exc}"})
                continue
            observed_spread = float(np.median(opt_quotes.ask - opt_quotes.bid))
            if not np.isfinite(observed_spread) or observed_spread <= 0.0:
                continue
            vol_ask_o = float(np.median(opt_quotes.ask_size))
            vol_bid_o = float(np.median(opt_quotes.bid_size))
            if vol_ask_o <= 0.0 or vol_bid_o <= 0.0:
                continue
            n_shares = implied_contracts(vol_ask_o, vol_bid_o)
            model_spread = model_implied_option_spread(n_shares, k, s0, st_up, st_down, lam, phi)
            explained = explained_spread_pct(model_spread, observed_spread)
            imp_n = implied_n(observed_spread, k, s0, st_up, st_down, lam, phi)
            imp_phi = implied_phi(observed_spread, n_shares, k, s0, st_up, st_down)
            records.append({
                "moneyness": moneyness,
                "phi": phi,
                "n_shares": n_shares,
                "observed_spread": observed_spread,
                "model_spread": model_spread,
                "explained_pct": explained,
                "implied_n": imp_n,
                "implied_phi": imp_phi,
            })

    assert records, f"no usable SPY option quote samples were obtained; skipped={skipped}"
    df = pd.DataFrame.from_records(records)

    band_rows = []
    for band in EXP2_MONEYNESS_BANDS:
        label = "all strikes" if band is None else f"delta<={band * 100:.1f}%"
        sub = df if band is None else df[df["moneyness"] <= band]
        row = {"band": label, "n_options": int(sub.shape[0])}
        for col, out_name in (
            ("explained_pct", "median_explained_pct"),
            ("observed_spread", "median_observed_spread_usd"),
            ("model_spread", "median_model_spread_usd"),
            ("implied_n", "median_implied_n_shares"),
            ("implied_phi", "median_implied_phi"),
            ("phi", "median_calibrated_phi"),
        ):
            row[out_name] = float(sub[col].median()) if not sub.empty else float("nan")
        band_rows.append(row)
    band_df = pd.DataFrame(band_rows)

    summary = {
        "data_source": "QuantX Dataloader (SPY daily bars) and Massive REST "
                       "(SPY/option NBBO quotes, option contract reference)",
        "sample_period": {"start": EXP2_START_DATE, "end": EXP2_END_DATE},
        "n_expiries_paper": None,
        "n_options_paper": EXP2_N_OPTIONS_PAPER,
        "n_expiries_run": len(expiry_candidates),
        "n_options_run": int(df.shape[0]),
        "n_skipped": len(skipped),
        "skipped_reasons": skipped[:20],
        "bands": band_df.to_dict(orient="records"),
    }
    return summary


def plot_bands(summary: dict, path: Path) -> None:
    bands = summary["bands"]
    labels = [b["band"] for b in bands]
    values = [b["median_explained_pct"] for b in bands]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color="steelblue")
    ax.set_ylabel("Median explained spread (%)")
    ax.set_xlabel("Moneyness band |S0/K - 1|")
    ax.set_title("Model-implied share of the observed SPY call bid-ask spread\n(Sec. 2.4.2 / Appendix A)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: SPY option calibration")
    parser.add_argument("--full", action="store_true",
                         help="use the paper-scale candidate expiration count (EXP2_N_EXPIRIES_PAPER)")
    args = parser.parse_args()
    n_expiries = EXP2_N_EXPIRIES_PAPER if args.full else EXP2_N_EXPIRIES_RUN

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = run_calibration(n_expiries=n_expiries)
    with open(RESULTS_DIR / "exp2_calibration.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_bands(summary, RESULTS_DIR / "exp2_calibration.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
