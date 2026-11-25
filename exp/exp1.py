"""Experiment 1: one-period binomial model, limit order book shape (Sec. 2.4.1).

Illustrates how the nonlinear execution-cost function (2.9) in the
underlying market is inherited by the call-option market: average
per-unit execution prices of the underlying and of a call option are
computed as the traded quantity N varies, for three (lambda, phi)
combinations, following the numerical example of Sec. 2.4.1 (page 7).

Run: python -m exp.exp1
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.binomial_one_period import average_execution_price_option, average_execution_price_underlying
from src.execution_costs import execution_cost_sqrt_linear
from src.config import (
    EXP1_K,
    EXP1_LAMBDA_PHI_SETS,
    EXP1_M_UNIT,
    EXP1_N_MULTIPLES,
    EXP1_N_UNIT,
    EXP1_R,
    EXP1_S0,
    EXP1_ST_DOWN,
    EXP1_ST_UP,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_table() -> dict:
    """Average execution prices of the underlying (per share) and of the
    call option (per contract) over a dense grid of N and over the
    table multiples of the illustrative transaction size, for every
    (lambda, phi) combination of Figure 2."""
    n_dense = np.concatenate([
        np.linspace(-5.0 * EXP1_N_UNIT, -0.05 * EXP1_N_UNIT, 200),
        np.linspace(0.05 * EXP1_N_UNIT, 5.0 * EXP1_N_UNIT, 200),
    ])
    table_rows = []
    curves = {}
    for lam, phi in EXP1_LAMBDA_PHI_SETS:
        underlying_dense = average_execution_price_underlying(
            n_dense, EXP1_S0, execution_cost_sqrt_linear, lam=lam, phi=phi
        )
        option_n_dense = n_dense  # one option contract <-> one underlying share, Sec. 2.4.1
        phi_func_bound = partial(execution_cost_sqrt_linear, lam=lam)
        option_dense = average_execution_price_option(
            option_n_dense, EXP1_K, EXP1_S0, EXP1_ST_UP, EXP1_ST_DOWN, lam, EXP1_R,
            phi_func_bound, phi=phi,
        )
        curves[f"lam{lam}_phi{phi}"] = {
            "n": n_dense.tolist(),
            "underlying_price": np.asarray(underlying_dense, dtype=float).tolist(),
            "option_price": np.asarray(option_dense, dtype=float).tolist(),
        }

        for mult in EXP1_N_MULTIPLES:
            if mult == 0:
                continue
            n_shares = mult * EXP1_N_UNIT
            m_contracts = mult * EXP1_M_UNIT
            s_price = float(average_execution_price_underlying(
                n_shares, EXP1_S0, execution_cost_sqrt_linear, lam=lam, phi=phi
            ))
            o_price = float(average_execution_price_option(
                m_contracts, EXP1_K, EXP1_S0, EXP1_ST_UP, EXP1_ST_DOWN, lam, EXP1_R,
                partial(execution_cost_sqrt_linear, lam=lam), phi=phi,
            ))
            table_rows.append({
                "lambda": lam, "phi": phi, "multiple": mult,
                "N_shares": n_shares, "m_contracts": m_contracts,
                "underlying_avg_price": s_price, "option_avg_price": o_price,
            })

    spreads = []
    for lam, phi in EXP1_LAMBDA_PHI_SETS:
        buy_s = float(average_execution_price_underlying(
            EXP1_N_UNIT, EXP1_S0, execution_cost_sqrt_linear, lam=lam, phi=phi
        ))
        sell_s = float(average_execution_price_underlying(
            -EXP1_N_UNIT, EXP1_S0, execution_cost_sqrt_linear, lam=lam, phi=phi
        ))
        buy_o = float(average_execution_price_option(
            EXP1_M_UNIT, EXP1_K, EXP1_S0, EXP1_ST_UP, EXP1_ST_DOWN, lam, EXP1_R,
            partial(execution_cost_sqrt_linear, lam=lam), phi=phi,
        ))
        sell_o = float(average_execution_price_option(
            -EXP1_M_UNIT, EXP1_K, EXP1_S0, EXP1_ST_UP, EXP1_ST_DOWN, lam, EXP1_R,
            partial(execution_cost_sqrt_linear, lam=lam), phi=phi,
        ))
        spreads.append({
            "lambda": lam, "phi": phi,
            "underlying_spread": buy_s - sell_s,
            "option_spread": buy_o - sell_o,
        })

    return {"table_rows": table_rows, "curves": curves, "spreads": spreads}


def make_plot(curves: dict, out_path: Path) -> None:
    """Figure 2: average execution price vs N for the underlying and the
    call option, one panel per (lambda, phi) combination (page 7)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False)
    for key, data in curves.items():
        n = np.asarray(data["n"])
        axes[0].plot(n, data["underlying_price"], label=key)
        axes[1].plot(n, data["option_price"], label=key)
    axes[0].axhline(EXP1_S0, color="gray", linestyle=":", linewidth=1, label="frictionless S0")
    axes[0].set_xlabel("N (shares)")
    axes[0].set_ylabel("average execution price")
    axes[0].set_title("Underlying market")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("N (contracts)")
    axes[1].set_ylabel("average execution price")
    axes[1].set_title("Call option market (inherited concavity)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Experiment 1: execution-cost shape of the limit order book (Sec. 2.4.1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run() -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    result = compute_table()
    make_plot(result["curves"], RESULTS_DIR / "exp1_execution_prices.png")

    summary_rows = []
    for row in result["table_rows"]:
        summary_rows.append({
            "lambda": row["lambda"], "phi": row["phi"], "multiple_of_n": row["multiple"],
            "underlying_avg_price": round(row["underlying_avg_price"], 6),
            "option_avg_price": round(row["option_avg_price"], 6),
        })
    with open(RESULTS_DIR / "exp1_execution_prices.json", "w") as f:
        json.dump({"table_rows": summary_rows, "spreads": result["spreads"]}, f, indent=2)

    print("Experiment 1 spreads (buy-sell average price) by (lambda, phi):")
    for s in result["spreads"]:
        print(
            f"  lambda={s['lambda']}, phi={s['phi']}: "
            f"underlying spread={s['underlying_spread']:.6f}, option spread={s['option_spread']:.6f}"
        )
    return result


if __name__ == "__main__":
    run()
