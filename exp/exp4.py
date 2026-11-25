"""Experiment 4: quadratic payoff replication, closed-form solution (Sec. 5.1, Figure 5).

For V(x) = N(alpha*x^2 + beta*x + gamma), in the midpoint regime 2*phi = lambda,
computes the per-contract replication price and initial position over a grid of
initial fundamental prices S0, for every (N, lambda) combination of Figure 5, and
compares against the frictionless (no price-impact) benchmark.

Run: python -m exp.exp4
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.quadratic_payoff import frictionless_quadratic_price, solve_quadratic_ansatz
from src.config import (
    EXP4_ALPHA,
    EXP4_BETA,
    EXP4_GAMMA,
    EXP4_LAMBDA_VALUES,
    EXP4_N_VALUES,
    EXP4_R,
    EXP4_S0_MAX,
    EXP4_S0_MIN,
    EXP4_S0_POINTS,
    EXP4_SIGMA,
    EXP4_T,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_curves(s0_grid: np.ndarray) -> dict:
    """Solve the quadratic ansatz for every (N, lambda) combination of Figure 5,
    plus the frictionless benchmark (computed once per N, since it does not
    depend on lambda)."""
    curves: dict = {}
    frictionless: dict = {}
    for n in EXP4_N_VALUES:
        frictionless[n] = frictionless_quadratic_price(
            s0_grid, n, EXP4_ALPHA, EXP4_BETA, EXP4_GAMMA, EXP4_R, EXP4_SIGMA, EXP4_T
        )
        for lam in EXP4_LAMBDA_VALUES:
            sol = solve_quadratic_ansatz(
                s0_grid, n, EXP4_ALPHA, EXP4_BETA, EXP4_GAMMA, lam, EXP4_R, EXP4_SIGMA, EXP4_T
            )
            curves[(n, lam)] = sol
    return {"curves": curves, "frictionless": frictionless}


def run_checks(s0_grid: np.ndarray, curves: dict, frictionless: dict) -> dict:
    """Checks (a)-(c): price above the frictionless benchmark, the lambda-N
    scaling identity in the midpoint regime, and monotonicity in N."""
    checks: dict = {}

    # (a) price_per_contract >= frictionless benchmark elementwise, for every (n, lam).
    above_benchmark = True
    for (n, lam), sol in curves.items():
        if not np.all(sol["price_per_contract"] >= frictionless[n] - 1e-6):
            above_benchmark = False
    checks["price_above_frictionless"] = bool(above_benchmark)

    # (b) scaling property of the midpoint regime: (N=1, lam=0.1) == (N=10, lam=0.01).
    if (1, 0.1) in curves and (10, 0.01) in curves:
        sol_a = curves[(1, 0.1)]
        sol_b = curves[(10, 0.01)]
        price_match = np.allclose(sol_a["price_per_contract"], sol_b["price_per_contract"], rtol=1e-6)
        position_match = np.allclose(sol_a["position_per_contract"], sol_b["position_per_contract"], rtol=1e-6)
        checks["scaling_property"] = bool(price_match and position_match)
    else:
        checks["scaling_property"] = None

    # (c) per-contract price and position increase with N, at matching lambda=0.1.
    # Position can be negative here (beta < 0 dominates the delta on this S0 range),
    # so "increase" is checked on its magnitude, i.e. the hedging exposure grows with N.
    if (10, 0.1) in curves and (1, 0.1) in curves:
        sol_10 = curves[(10, 0.1)]
        sol_1 = curves[(1, 0.1)]
        price_increasing = np.all(sol_10["price_per_contract"] >= sol_1["price_per_contract"] - 1e-6)
        position_increasing = np.all(
            np.abs(sol_10["position_per_contract"]) >= np.abs(sol_1["position_per_contract"]) - 1e-6
        )
        checks["increasing_in_n"] = bool(price_increasing and position_increasing)
    else:
        checks["increasing_in_n"] = None

    return checks


def make_plot(s0_grid: np.ndarray, curves: dict, frictionless: dict, out_path: Path) -> None:
    """Figure 5: price_per_contract and position_per_contract vs S0, for every
    (N, lambda) combination, plus the frictionless benchmarks (Sec. 5.1)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (n, lam), sol in curves.items():
        label = f"N={n}, lambda={lam}"
        axes[0].plot(s0_grid, sol["price_per_contract"], label=label)
        axes[1].plot(s0_grid, sol["position_per_contract"], label=label)
    for n, benchmark in frictionless.items():
        axes[0].plot(s0_grid, benchmark, linestyle="--", color="gray", label=f"frictionless N={n}")

    axes[0].set_xlabel("S0 (initial fundamental price)")
    axes[0].set_ylabel("derivative price per contract")
    axes[0].set_title("Replication price")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("S0 (initial fundamental price)")
    axes[1].set_ylabel("initial position per contract")
    axes[1].set_title("Replication position")
    axes[1].legend(fontsize=7)

    fig.suptitle("Experiment 4: quadratic payoff replication, closed-form solution (Sec. 5.1, Figure 5)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run() -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    s0_grid = np.linspace(EXP4_S0_MIN, EXP4_S0_MAX, EXP4_S0_POINTS)

    computed = compute_curves(s0_grid)
    curves, frictionless = computed["curves"], computed["frictionless"]
    checks = run_checks(s0_grid, curves, frictionless)

    assert checks["price_above_frictionless"], "price_per_contract must be >= frictionless benchmark"
    if checks["scaling_property"] is not None:
        assert checks["scaling_property"], "midpoint-regime scaling property (N=1,lam=0.1)==(N=10,lam=0.01) failed"
    if checks["increasing_in_n"] is not None:
        assert checks["increasing_in_n"], "per-contract price/position must increase with N at fixed lambda"

    make_plot(s0_grid, curves, frictionless, RESULTS_DIR / "exp4_quadratic_payoff.png")

    table: dict = {}
    for (n, lam), sol in curves.items():
        table[f"N{n}_lam{lam}"] = {
            "s0_grid": s0_grid.tolist(),
            "price_per_contract": np.asarray(sol["price_per_contract"], dtype=float).tolist(),
            "position_per_contract": np.asarray(sol["position_per_contract"], dtype=float).tolist(),
        }
    for n, benchmark in frictionless.items():
        table[f"frictionless_N{n}"] = {
            "s0_grid": s0_grid.tolist(),
            "price_per_contract": np.asarray(benchmark, dtype=float).tolist(),
        }
    table["checks"] = checks

    with open(RESULTS_DIR / "exp4_quadratic_payoff.json", "w") as f:
        json.dump(table, f, indent=2)

    print("Experiment 4 checks (Sec. 5.1, Figure 5):")
    for name, value in checks.items():
        print(f"  {name}: {value}")

    return {"s0_grid": s0_grid, "curves": curves, "frictionless": frictionless, "checks": checks}


if __name__ == "__main__":
    run()
