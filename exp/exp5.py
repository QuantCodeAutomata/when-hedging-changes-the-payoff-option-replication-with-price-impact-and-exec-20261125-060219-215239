"""Experiment 5: continuous-time call replication (paper Sec. 4, Sec. 5.2,
Figures 6-9).

Solves the nonlinear midpoint-liquidation-value PDE (4.6) with the implicit
terminal condition (4.8), made explicit for a call by the quadratic
completion (5.2), using the finite-difference solver in src/pde_solver.py.
Reports: the modified terminal payoff (Figure 6), the replication identity
check under increasing grid resolution (Figure 7), the price/position per
contract versus S_0 for N in {1, 10} and lambda in {0.01, 0.1} (Figure 8),
and the "moving target" illustration where the hedger's own trading brings
an out-of-the-money call into the money (Figure 9).

Run with ``python -m exp.exp5``. The ``--full`` flag selects the paper-scale
PDE grid (EXP5_N_SPACE_PAPER / EXP5_N_TIME_PAPER) and the full discretisation
list (EXP5_MC_STEPS_LIST) for the Figure 7 check; by default a reduced,
time-limited configuration is used (see results/RESULTS.md).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from src import config
from src.pde_solver import (
    call_terminal_delta_tilde,
    call_terminal_payoff_tilde,
    option_price_and_hedge_at_t0,
    price_and_hedge_at,
    solve_call_pde,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def bs_call_price_delta(s: np.ndarray, k: float, r: float, sigma: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Black-Scholes call price and delta (standard closed form, used only as
    the frictionless/no-impact benchmark of Sec. 5.2; not the paper's model)."""
    s = np.asarray(s, dtype=float)
    tau = max(tau, 1e-12)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = s * norm.cdf(d1) - k * np.exp(-r * tau) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return price, delta


def solve_base_pde(n_space: int, n_time: int) -> dict:
    """Solve PDE (4.6) for the Figure 6-9 base case: K, lambda, phi, r, sigma,
    N, T from src/config.py (page 24 caption values)."""
    return solve_call_pde(
        k=config.EXP5_K, n=float(config.EXP5_N), lam=config.EXP5_LAMBDA, phi=config.EXP5_PHI,
        r=config.EXP5_R, sigma=config.EXP5_SIGMA, t_mat=config.EXP5_T,
        x_min=config.EXP5_SPATIAL_MIN, x_max=config.EXP5_SPATIAL_MAX,
        n_space=n_space, n_time=n_time,
    )


def figure6_terminal_condition() -> dict:
    """Modified terminal payoff V~ and terminal position V~' versus the raw
    call payoff N(x-K)+ (Equation (5.2))."""
    k, n, lam, phi = config.EXP5_K, float(config.EXP5_N), config.EXP5_LAMBDA, config.EXP5_PHI
    x = np.linspace(k - 4.0 * phi * n - 2.0, k + 20.0, 400)
    v_tilde = call_terminal_payoff_tilde(x, k, n, lam, phi)
    delta_tilde = call_terminal_delta_tilde(x, k, n, lam, phi)
    raw_payoff = n * np.maximum(x - k, 0.0)
    return {"x": x, "v_tilde": v_tilde, "delta_tilde": delta_tilde, "raw_payoff": raw_payoff}


def figure7_replication_check(resolutions: list[tuple[int, int]], n_samples: int, seed: int = 0) -> list[dict]:
    """For increasing (n_space, n_time) grid resolution, verify the
    replication identity V(P_T^{Delta*}) = X*_T, Equation preceding (4.9):

        P_T^{Delta*} = S_T + lambda * Delta*_T
        X*_T = u(T, S_T) - (2*phi-lambda)/2 * (Delta*_T)^2

    Delta*_T = partial_x u(T, S_T) is read off the finite-difference solution
    at tau=0 (t=T); as the grid refines this numerical derivative converges
    to the exact terminal position V~'(S_T), so the replication error shrinks.
    """
    k, n, lam, phi = config.EXP5_K, float(config.EXP5_N), config.EXP5_LAMBDA, config.EXP5_PHI
    rln = (2.0 * phi - lam) / 2.0
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n_samples)
    s_t = config.EXP5_S0 * np.exp(
        (config.EXP5_R - 0.5 * config.EXP5_SIGMA**2) * config.EXP5_T
        + config.EXP5_SIGMA * np.sqrt(config.EXP5_T) * z
    )
    rows = []
    for n_space, n_time in resolutions:
        sol = solve_base_pde(n_space, n_time)
        errors = []
        for s in s_t:
            s = float(np.clip(s, sol["x"][1], sol["x"][-2]))
            u_t, delta_t = price_and_hedge_at(sol, 0.0, s)
            p_t = s + lam * delta_t
            x_t = u_t - rln * delta_t**2
            payoff = n * max(p_t - k, 0.0)
            errors.append(abs(payoff - x_t))
        rows.append({
            "n_space": n_space, "n_time": n_time,
            "mean_abs_error": float(np.mean(errors)), "max_abs_error": float(np.max(errors)),
        })
    return rows


def figure8_price_position_vs_s0(n_space: int, n_time: int) -> dict:
    """Option price and initial position per contract versus S_0, for
    N in {1,10} and lambda in {0.01,0.1} with phi fixed (Figure 8)."""
    out: dict = {}
    s0_grid = np.linspace(config.EXP5_SPATIAL_MIN + 20.0, config.EXP5_SPATIAL_MAX - 20.0, 60)
    bs_price, bs_delta = bs_call_price_delta(s0_grid, config.EXP5_K, config.EXP5_R, config.EXP5_SIGMA, config.EXP5_T)
    out["frictionless"] = {
        "s0_grid": s0_grid.tolist(),
        "price_per_contract": bs_price.tolist(),
        "position_per_contract": bs_delta.tolist(),
    }
    for n_contracts in config.EXP5_N_VALUES:
        for lam in config.EXP5_LAMBDA_VALUES:
            sol = solve_call_pde(
                k=config.EXP5_K, n=float(n_contracts), lam=lam, phi=config.EXP5_PHI,
                r=config.EXP5_R, sigma=config.EXP5_SIGMA, t_mat=config.EXP5_T,
                x_min=config.EXP5_SPATIAL_MIN, x_max=config.EXP5_SPATIAL_MAX,
                n_space=n_space, n_time=n_time,
            )
            price = np.empty_like(s0_grid)
            position = np.empty_like(s0_grid)
            for i, s0 in enumerate(s0_grid):
                p, d = option_price_and_hedge_at_t0(sol, float(s0), 2.0 * config.EXP5_PHI - lam)
                price[i] = p / n_contracts
                position[i] = d / n_contracts
            out[f"N{n_contracts}_lam{lam}"] = {
                "s0_grid": s0_grid.tolist(),
                "price_per_contract": price.tolist(),
                "position_per_contract": position.tolist(),
            }
    return out


def figure9_moving_target(n_space: int, n_time: int, n_paths: int, n_steps: int = 252, seed: int = 1) -> dict:
    """Among simulated GBM paths under Q, find those with S_T < K < P_T^{Delta*}
    (the option would expire worthless but the hedger's own trading brings it
    into the money), and record one example path with S_t, the impacted price
    P_t^{Delta*}=S_t+lambda*Delta*_t, the Black-Scholes delta and the
    impact-model delta Delta*_t = partial_x u(t,S_t)."""
    k, lam = config.EXP5_K, config.EXP5_LAMBDA
    sol = solve_base_pde(n_space, n_time)
    rng = np.random.default_rng(seed)
    dt = config.EXP5_T / n_steps
    moving_target_count = 0
    example = None
    for path_idx in range(n_paths):
        z = rng.normal(size=n_steps)
        log_increments = (config.EXP5_R - 0.5 * config.EXP5_SIGMA**2) * dt + config.EXP5_SIGMA * np.sqrt(dt) * z
        s_path = config.EXP5_S0 * np.exp(np.cumsum(np.concatenate([[0.0], log_increments])))
        s_t = s_path[-1]
        s_clamped = float(np.clip(s_t, sol["x"][1], sol["x"][-2]))
        _, delta_t = price_and_hedge_at(sol, 0.0, s_clamped)
        p_t = s_t + lam * delta_t
        if s_t < k < p_t:
            moving_target_count += 1
            if example is None:
                times = np.linspace(0.0, config.EXP5_T, n_steps + 1)
                deltas = np.empty(n_steps + 1)
                bs_deltas = np.empty(n_steps + 1)
                for i, (t_i, s_i) in enumerate(zip(times, s_path)):
                    s_i_clamped = float(np.clip(s_i, sol["x"][1], sol["x"][-2]))
                    tau = config.EXP5_T - t_i
                    _, deltas[i] = price_and_hedge_at(sol, tau, s_i_clamped)
                    _, bs_delta_i = bs_call_price_delta(np.array([s_i]), k, config.EXP5_R, config.EXP5_SIGMA, max(tau, 1e-6))
                    bs_deltas[i] = float(bs_delta_i[0])
                impacted_path = s_path + lam * deltas
                example = {
                    "times": times.tolist(), "s_path": s_path.tolist(),
                    "impacted_path": impacted_path.tolist(),
                    "impact_delta": deltas.tolist(), "bs_delta": bs_deltas.tolist(),
                }
    return {"n_paths": n_paths, "moving_target_count": moving_target_count, "example": example}


def make_plots(fig6: dict, fig7: list[dict], fig8: dict, fig9: dict) -> None:
    k, n, lam, phi = config.EXP5_K, float(config.EXP5_N), config.EXP5_LAMBDA, config.EXP5_PHI

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fig6["x"], fig6["raw_payoff"], "--", label="raw call payoff N(x-K)+")
    ax.plot(fig6["x"], fig6["v_tilde"], label=r"modified terminal payoff $\tilde V$")
    ax.axvline(k, color="gray", linewidth=0.8, label="strike K")
    ax.set_xlabel("terminal fundamental price x")
    ax.set_ylabel("terminal value")
    ax.set_title("Figure 6: modified terminal payoff (Eq. 5.2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_figure6_terminal_condition.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    n_space_vals = [r["n_space"] for r in fig7]
    mean_err = [r["mean_abs_error"] for r in fig7]
    max_err = [r["max_abs_error"] for r in fig7]
    ax.loglog(n_space_vals, mean_err, "o-", label="mean |V(P_T)-X_T*|")
    ax.loglog(n_space_vals, max_err, "s--", label="max |V(P_T)-X_T*|")
    ax.set_xlabel("spatial grid points (discretisation)")
    ax.set_ylabel("replication error")
    ax.set_title("Figure 7: replication identity error vs grid resolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_figure7_replication_check.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fig8["frictionless"]["s0_grid"], fig8["frictionless"]["price_per_contract"], "k--", label="no-impact (Black-Scholes)")
    axes[1].plot(fig8["frictionless"]["s0_grid"], fig8["frictionless"]["position_per_contract"], "k--", label="no-impact (Black-Scholes)")
    for n_contracts in config.EXP5_N_VALUES:
        for lamv in config.EXP5_LAMBDA_VALUES:
            key = f"N{n_contracts}_lam{lamv}"
            axes[0].plot(fig8[key]["s0_grid"], fig8[key]["price_per_contract"], label=f"N={n_contracts}, λ={lamv}")
            axes[1].plot(fig8[key]["s0_grid"], fig8[key]["position_per_contract"], label=f"N={n_contracts}, λ={lamv}")
    axes[0].set_xlabel("S_0"); axes[0].set_ylabel("option price per contract"); axes[0].legend()
    axes[1].set_xlabel("S_0"); axes[1].set_ylabel("initial position per contract"); axes[1].legend()
    fig.suptitle("Figure 8: price and position per contract vs S_0")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp5_figure8_price_position.png", dpi=150)
    plt.close(fig)

    if fig9["example"] is not None:
        ex = fig9["example"]
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
        axes[0].plot(ex["times"], ex["s_path"], label="fundamental price S_t")
        axes[0].plot(ex["times"], ex["impacted_path"], label=r"impacted price $P_t^{\Delta^*}$")
        axes[0].axhline(k, color="gray", linewidth=0.8, label="strike K")
        axes[0].set_ylabel("price"); axes[0].legend()
        axes[1].plot(ex["times"], ex["impact_delta"], label="impact-model delta Δ*_t")
        axes[1].plot(ex["times"], ex["bs_delta"], label="Black-Scholes delta")
        axes[1].set_xlabel("time t"); axes[1].set_ylabel("hedge position"); axes[1].legend()
        fig.suptitle("Figure 9: hedging moves an OTM call into the money")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "exp5_figure9_moving_target.png", dpi=150)
        plt.close(fig)


def run(full: bool = False) -> dict:
    """Run experiment 5 end to end and save results/metrics to results/."""
    n_space = config.EXP5_N_SPACE_PAPER if full else config.EXP5_N_SPACE_RUN
    n_time = config.EXP5_N_TIME_PAPER if full else config.EXP5_N_TIME_RUN
    steps_list = config.EXP5_MC_STEPS_LIST if full else config.EXP5_MC_STEPS_LIST_RUN
    mc_paths = config.EXP5_MC_PATHS_PAPER

    fig6 = figure6_terminal_condition()

    resolutions = [(max(20, s // 4), s) for s in steps_list]
    fig7 = figure7_replication_check(resolutions, n_samples=40)
    assert fig7[-1]["mean_abs_error"] <= fig7[0]["mean_abs_error"] + 1e-9, (
        "replication error should not increase as the grid is refined"
    )

    fig8 = figure8_price_position_vs_s0(n_space, n_time)
    for key, series in fig8.items():
        if key == "frictionless":
            continue
        price = np.array(series["price_per_contract"])
        bs_price = np.array(fig8["frictionless"]["price_per_contract"])
        assert np.mean(price >= bs_price - 1e-3) > 0.9, (
            f"{key}: impact-model price per contract should exceed the Black-Scholes benchmark"
        )
        position = np.array(series["position_per_contract"])
        bs_delta = np.array(fig8["frictionless"]["position_per_contract"])
        assert np.mean(position >= bs_delta - 1e-3) > 0.9, (
            f"{key}: impact-model delta should exceed the Black-Scholes delta for a call"
        )

    fig9 = figure9_moving_target(n_space, n_time, n_paths=mc_paths)

    make_plots(fig6, fig7, fig8, fig9)

    results = {
        "grid": {"n_space": n_space, "n_time": n_time, "full": full},
        "figure7_replication_check": fig7,
        "figure8_price_position": {k: v for k, v in fig8.items()},
        "figure9_moving_target": {
            "n_paths": fig9["n_paths"], "moving_target_count": fig9["moving_target_count"],
        },
    }
    with open(RESULTS_DIR / "exp5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Figure 7 replication error (grid -> mean/max abs error): {fig7}")
    print(f"Figure 9: {fig9['moving_target_count']} of {fig9['n_paths']} paths show S_T<K<P_T")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: continuous-time call replication")
    parser.add_argument("--full", action="store_true", help="use the paper-scale PDE grid and discretisation list")
    args = parser.parse_args()
    run(full=args.full)
