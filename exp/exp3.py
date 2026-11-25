"""Experiment 3: multiperiod binomial replication with linear price impact
and execution costs (Sec. 3, Theorem 3.2, Corollary 3.3).

Builds a recombining fundamental-price tree and runs the backward-induction
replication scheme of Theorem 3.2 for a call option, under two execution-cost
regimes: the midpoint regime 2*phi = lambda (Sec. 3.2) and a general regime
with 2*phi > lambda, which the paper's nondegeneracy condition (3.9) is also
checked numerically for. The replication cost X_0* is compared, for a range
of step counts M, against the frictionless no-arbitrage price of Corollary
3.3, and the initial hedge Delta*_{t_1}(S_0) is tracked as M grows.

Run: python -m exp.exp3
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    EXP3_K,
    EXP3_LAMBDA,
    EXP3_N,
    EXP3_PHI_GENERAL,
    EXP3_PHI_MIDPOINT,
    EXP3_R,
    EXP3_S0,
    EXP3_SIGMA,
    EXP3_T,
    EXP3_M_STEPS,
)
from src.multiperiod_binomial import replicate_call

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Step counts used for the convergence study; EXP3_M_STEPS is kept as the
# headline (largest) value of the list.
M_STEPS_LIST: tuple[int, ...] = tuple(sorted(set((5, 10, 20, EXP3_M_STEPS))))


def tree_params(m_steps: int) -> tuple[float, float, float]:
    """Up/down factors and the per-step interest rate for a given M.

    u = exp(sigma*sqrt(h)), d = exp(-sigma*sqrt(h)), h = T/M, consistent with
    Sec. 3.3; r_per_step = EXP3_R * h (annualized rate scaled to the step).
    """
    h = EXP3_T / m_steps
    u = float(np.exp(EXP3_SIGMA * np.sqrt(h)))
    d = float(np.exp(-EXP3_SIGMA * np.sqrt(h)))
    r_per_step = EXP3_R * h
    return u, d, r_per_step


def run_one(m_steps: int, lam: float, phi: float) -> dict:
    """Run the replication scheme for one (M, lambda, phi) configuration."""
    u, d, r_per_step = tree_params(m_steps)
    return replicate_call(
        s0=EXP3_S0, u=u, d=d, m_steps=m_steps, r_per_step=r_per_step,
        lam=lam, phi=phi, k=EXP3_K, n=EXP3_N,
    )


def compute_table() -> tuple[list[dict], dict[int, dict]]:
    """Run both execution-cost regimes over M_STEPS_LIST.

    Returns (table_rows, raw_outputs) where raw_outputs[m] holds the full
    replicate_call() dict for each regime, keyed by m, for plotting.
    """
    regimes = {"midpoint": EXP3_PHI_MIDPOINT, "general": EXP3_PHI_GENERAL}
    table_rows: list[dict] = []
    raw_outputs: dict[int, dict] = {}

    for m_steps in M_STEPS_LIST:
        raw_outputs[m_steps] = {}
        for regime, phi in regimes.items():
            out = run_one(m_steps, EXP3_LAMBDA, phi)
            raw_outputs[m_steps][regime] = out

            # Corollary 3.3: replication cost of a monotone call payoff must
            # dominate the frictionless no-arbitrage price.
            assert out["x0_star"] >= out["frictionless_price"] - 1e-6, (
                f"Corollary 3.3 violated at M={m_steps}, regime={regime}: "
                f"x0_star={out['x0_star']} < frictionless={out['frictionless_price']}"
            )
            if regime == "midpoint":
                assert out["nondegenerate"], (
                    f"nondegeneracy condition (3.9) failed at M={m_steps} in the midpoint regime"
                )
            # f is indexed by node of the recombining tree only (one value
            # per node, not per path): f[m] must have exactly m entries.
            for m in range(1, m_steps + 1):
                assert out["f"][m].shape[0] == m

            table_rows.append({
                "M": m_steps,
                "regime": regime,
                "lambda": EXP3_LAMBDA,
                "phi": phi,
                "x0_star": out["x0_star"],
                "frictionless_price": out["frictionless_price"],
                "cost_minus_frictionless": out["x0_star"] - out["frictionless_price"],
                "nondegenerate": out["nondegenerate"],
            })

    return table_rows, raw_outputs


def make_plot(raw_outputs: dict[int, dict], out_path: Path) -> None:
    """Two-panel figure: (a) replication cost X_0* and frictionless price vs
    M in the midpoint regime; (b) initial hedge Delta*_{t_1}(S_0) vs M in
    both regimes."""
    m_list = sorted(raw_outputs.keys())
    x0_mid = [raw_outputs[m]["midpoint"]["x0_star"] for m in m_list]
    frictionless_mid = [raw_outputs[m]["midpoint"]["frictionless_price"] for m in m_list]
    f1_mid = [raw_outputs[m]["midpoint"]["f"][1][0] for m in m_list]
    f1_gen = [raw_outputs[m]["general"]["f"][1][0] for m in m_list]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(m_list, x0_mid, "o-", label="Replication cost $X_0^*$ (midpoint regime)")
    axes[0].plot(m_list, frictionless_mid, "s--", label="Frictionless price (Cor. 3.3)")
    axes[0].set_xlabel("Number of steps M")
    axes[0].set_ylabel("Price")
    axes[0].set_title("Convergence of $X_0^*$ to the frictionless price")
    axes[0].legend(fontsize=8)

    axes[1].plot(m_list, f1_mid, "o-", label="midpoint regime ($2\\phi=\\lambda$)")
    axes[1].plot(m_list, f1_gen, "^-", label="general regime ($2\\phi>\\lambda$)")
    axes[1].set_xlabel("Number of steps M")
    axes[1].set_ylabel(r"Initial hedge $\Delta^*_{t_1}(S_0)$")
    axes[1].set_title("Initial replicating position vs number of steps")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Experiment 3: multiperiod binomial replication with price impact (Sec. 3)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run() -> dict:
    """Run the full experiment: compute the table, save JSON, save the plot,
    print a summary, and return everything as a dict."""
    RESULTS_DIR.mkdir(exist_ok=True)
    table_rows, raw_outputs = compute_table()

    make_plot(raw_outputs, RESULTS_DIR / "exp3_replication.png")

    json_rows = [
        {k: (bool(v) if k == "nondegenerate" else v) for k, v in row.items()}
        for row in table_rows
    ]
    with open(RESULTS_DIR / "exp3_replication_cost.json", "w") as fh:
        json.dump({"table_rows": json_rows}, fh, indent=2)

    print("Experiment 3: multiperiod binomial replication (Sec. 3, Theorem 3.2 / Corollary 3.3)")
    print(f"{'M':>4} {'regime':>9} {'lambda':>7} {'phi':>6} {'X_0*':>10} "
          f"{'frictionless':>13} {'diff':>10} {'nondeg':>7}")
    for row in table_rows:
        print(
            f"{row['M']:>4} {row['regime']:>9} {row['lambda']:>7.3f} {row['phi']:>6.3f} "
            f"{row['x0_star']:>10.5f} {row['frictionless_price']:>13.5f} "
            f"{row['cost_minus_frictionless']:>10.5f} {str(row['nondegenerate']):>7}"
        )

    return {"table_rows": table_rows, "raw_outputs": raw_outputs}


if __name__ == "__main__":
    run()
