"""Finite-difference solver for the midpoint liquidation value PDE (4.6) with
the implicit terminal condition (4.8)/(5.2) for a European call payoff.

Custom implementation (paper Eq. (4.6), (4.8), (5.2), Sec. 4-5.2): no
standard option-pricing library solves a nonlinear PDE with this reaction
structure, so a semi-implicit finite-difference scheme is implemented from
scratch. The linear (convection-diffusion-discount) part of the operator is
treated implicitly on a uniform spatial grid with a constant tridiagonal
matrix (prefactorised once); the nonlinear terms in (partial_x u)^2 and
(partial_xx u)^2 are evaluated explicitly from the previous time level
(a standard IMEX scheme for semilinear/quasilinear parabolic PDEs).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def call_terminal_payoff_tilde(
    x: np.ndarray, k: float, n: float, lam: float, phi: float
) -> np.ndarray:
    """Modified terminal payoff V~ for a call, Equation (5.2):

        V~(x) = 0                                   for x < K - 2*phi*N
        V~(x) = (x - K + 2*phi*N)^2 / (2*(2*phi-lam))  for K-2*phi*N <= x < K-lam*N
        V~(x) = N(x-K) + (2*phi+lam)/2 * N^2        for x >= K - lam*N
    """
    x = np.asarray(x, dtype=float)
    k_lo = k - 2.0 * phi * n
    k_hi = k - lam * n
    out = np.zeros_like(x)
    mid = (x >= k_lo) & (x < k_hi)
    high = x >= k_hi
    out[mid] = (x[mid] - k + 2.0 * phi * n) ** 2 / (2.0 * (2.0 * phi - lam))
    out[high] = n * (x[high] - k) + 0.5 * (2.0 * phi + lam) * n**2
    return out


def call_terminal_delta_tilde(
    x: np.ndarray, k: float, n: float, lam: float, phi: float
) -> np.ndarray:
    """Terminal replicating position V~'(x) (Sec. 5.2, paragraph following (5.2)):
    0 below K-2*phi*N, N above K-lam*N, linear interpolation in between."""
    x = np.asarray(x, dtype=float)
    k_lo = k - 2.0 * phi * n
    k_hi = k - lam * n
    out = np.zeros_like(x)
    mid = (x >= k_lo) & (x < k_hi)
    high = x >= k_hi
    out[mid] = (x[mid] - k + 2.0 * phi * n) / (2.0 * phi - lam)
    out[high] = n
    return out


def _build_linear_operator(x: np.ndarray, r: float, sigma: float, dx: float):
    """Tridiagonal matrix for L[u]_i = -r u_i + r x_i (u_{i+1}-u_{i-1})/(2dx)
    + (1/2) sigma^2 x_i^2 (u_{i+1}-2u_i+u_{i-1})/dx^2, Equation (4.6) linear part.

    Row 0 enforces u_0 = 0 (deep out-of-the-money Dirichlet condition); the
    last row enforces a linear right boundary (zero curvature), both
    time-independent so the same operator is reused at every time step.
    """
    nx = x.shape[0]
    lower = np.zeros(nx - 1)
    diag = np.zeros(nx)
    upper = np.zeros(nx - 1)
    for i in range(1, nx - 1):
        xi = x[i]
        lower[i - 1] = -r * xi / (2.0 * dx) + 0.5 * sigma**2 * xi**2 / dx**2
        diag[i] = -r - sigma**2 * xi**2 / dx**2
        upper[i] = r * xi / (2.0 * dx) + 0.5 * sigma**2 * xi**2 / dx**2
    return lower, diag, upper


def _solve_pde(
    x: np.ndarray,
    terminal: np.ndarray,
    r: float,
    sigma: float,
    lam: float,
    phi: float,
    rln: float,
    t_mat: float,
    n_time: int,
) -> np.ndarray:
    """Backward-in-time (forward in tau = T - t) solution of (4.6) on a uniform
    grid, returning the full array u[time_index, space_index] with
    u[0] = terminal condition (t=T) and u[n_time] = price at t=0.

    rln is (2*phi - lam)/2, the coefficient of (partial_xx u)^2 in (4.6);
    it is zero in the midpoint regime 2*phi=lam, reducing the scheme to the
    semilinear case used as a consistency check.
    """
    nx = x.shape[0]
    dx = x[1] - x[0]
    dtau = t_mat / n_time
    lower, diag, upper = _build_linear_operator(x, r, sigma, dx)

    # A = I - dtau * L, tridiagonal, boundary rows enforce u_0=0 and a linear
    # (zero-curvature) right boundary u_{N-1} - 2 u_{N-2} + u_{N-3} = 0.
    a_full = np.zeros((nx, nx))
    a_full[0, 0] = 1.0
    a_full[-1, -1] = 1.0
    a_full[-1, -2] = -2.0
    a_full[-1, -3] = 1.0
    for i in range(1, nx - 1):
        a_full[i, i - 1] = -dtau * lower[i - 1]
        a_full[i, i] = 1.0 - dtau * diag[i]
        a_full[i, i + 1] = -dtau * upper[i]
    lu, piv = lu_factor(a_full)

    u = np.empty((n_time + 1, nx))
    u[0] = terminal
    for step in range(n_time):
        u_prev = u[step]
        dux = np.zeros(nx)
        duxx = np.zeros(nx)
        dux[1:-1] = (u_prev[2:] - u_prev[:-2]) / (2.0 * dx)
        duxx[1:-1] = (u_prev[2:] - 2.0 * u_prev[1:-1] + u_prev[:-2]) / dx**2
        nonlinear = np.zeros(nx)
        nonlinear[1:-1] = (
            0.5 * r * lam * dux[1:-1] ** 2
            + rln * sigma**2 * x[1:-1] ** 2 * duxx[1:-1] ** 2
        )
        rhs = u_prev + dtau * nonlinear
        rhs[0] = 0.0
        rhs[-1] = 0.0
        u[step + 1] = lu_solve((lu, piv), rhs)
    return u


def solve_call_pde(
    k: float,
    n: float,
    lam: float,
    phi: float,
    r: float,
    sigma: float,
    t_mat: float,
    x_min: float,
    x_max: float,
    n_space: int,
    n_time: int,
) -> dict:
    """Solve PDE (4.6) with terminal condition (5.2) for a call option.

    Returns a dict with the spatial grid `x`, the time-to-maturity grid
    `tau` (tau=0 is t=T, tau=T is t=0), the full solution surface `u`
    (shape (n_time+1, n_space)), and convenience arrays `du_dx` of the same
    shape with the finite-difference first derivative (hedge Delta*_t).
    """
    assert x_min > 0.0 and x_max > x_min
    assert phi >= lam / 2.0 - 1e-12, "execution-cost regime requires 2*phi >= lambda (Eq. 2.3/3.3)"
    x = np.linspace(x_min, x_max, n_space)
    terminal = call_terminal_payoff_tilde(x, k, n, lam, phi)
    rln = (2.0 * phi - lam) / 2.0
    u = _solve_pde(x, terminal, r, sigma, lam, phi, rln, t_mat, n_time)

    dx = x[1] - x[0]
    du_dx = np.zeros_like(u)
    du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    du_dx[:, 0] = du_dx[:, 1]
    du_dx[:, -1] = du_dx[:, -2]

    tau = np.linspace(0.0, t_mat, n_time + 1)
    return {"x": x, "tau": tau, "u": u, "du_dx": du_dx, "rln": rln}


def price_and_hedge_at(
    solution: dict, tau_value: float, s_value: float
) -> tuple[float, float]:
    """Bilinearly interpolate the PDE solution's price level u(t,x) and hedge
    partial_x u(t,x) at a given time-to-maturity tau_value=T-t and price s_value.
    """
    x = solution["x"]
    tau = solution["tau"]
    u = solution["u"]
    du_dx = solution["du_dx"]
    it = np.clip(np.searchsorted(tau, tau_value) - 1, 0, len(tau) - 2)
    ix = np.clip(np.searchsorted(x, s_value) - 1, 0, len(x) - 2)
    wt = (tau_value - tau[it]) / (tau[it + 1] - tau[it])
    wx = (s_value - x[ix]) / (x[ix + 1] - x[ix])
    wt = np.clip(wt, 0.0, 1.0)
    wx = np.clip(wx, 0.0, 1.0)

    def bilerp(field: np.ndarray) -> float:
        f00, f01 = field[it, ix], field[it, ix + 1]
        f10, f11 = field[it + 1, ix], field[it + 1, ix + 1]
        return float(
            (1 - wt) * ((1 - wx) * f00 + wx * f01) + wt * ((1 - wx) * f10 + wx * f11)
        )

    return bilerp(u), bilerp(du_dx)


def option_price_and_hedge_at_t0(solution: dict, s0: float, two_phi_minus_lam: float) -> tuple[float, float]:
    """Initial option price X_0* = u(0,S_0) + (2phi-lam)/2 * (partial_x u(0,S_0))^2
    and initial hedge Delta*_{0+} = partial_x u(0,S_0) (Sec. 4.2, Eq. preceding (4.9))."""
    tau_t0 = solution["tau"][-1]
    u0, du0 = price_and_hedge_at(solution, tau_t0, s0)
    price = u0 + 0.5 * two_phi_minus_lam * du0**2
    return price, du0
