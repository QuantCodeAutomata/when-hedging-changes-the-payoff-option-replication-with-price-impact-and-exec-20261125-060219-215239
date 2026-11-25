"""Price impact and execution cost primitives shared by every experiment.

Custom implementation: these functions are the paper's primitive objects
(Sec. 2.1-2.2), not available in any standard quantitative-finance library.
"""
from __future__ import annotations

import numpy as np


def linear_impact(x: np.ndarray | float, lam: float) -> np.ndarray | float:
    """Permanent price impact I(x) = lambda * x (Sec. 2.1)."""
    return lam * x


def execution_cost_sqrt_linear(x: np.ndarray | float, lam: float, phi: float) -> np.ndarray | float:
    """Signed execution cost function (2.9).

    phi(x) = phi * sign(x) * sqrt(|x|)   for |x| <= 4*phi^2/lambda^2
    phi(x) = (lambda / 2) * x            otherwise

    The square-root branch is used for |x| below the threshold so that the
    no-price-manipulation condition (2.3) holds globally; for larger trades
    the cost switches to the linear branch.
    """
    x = np.asarray(x, dtype=float)
    threshold = 4.0 * phi**2 / lam**2
    sqrt_branch = phi * np.sign(x) * np.sqrt(np.abs(x))
    linear_branch = (lam / 2.0) * x
    out = np.where(np.abs(x) <= threshold, sqrt_branch, linear_branch)
    return out if out.ndim else float(out)


def execution_cost_linear(x: np.ndarray | float, phi: float) -> np.ndarray | float:
    """Linear (midpoint-type) execution cost phi(x) = phi * x, used in Sec. 3-5."""
    return phi * x


def round_trip_cost(x: np.ndarray | float, lam: float, r: float, phi_func, **phi_kwargs):
    """Round-trip cost function C(x) = phi(x)(1+r) - phi(-x) - I(x), Sec. 2.3 (Eq. preceding 2.7).

    C captures the gap between the cash raised when establishing the position x
    at time 0 (paying phi(x)(1+r) once compounded to maturity) and the cash
    recovered at maturity from closing out at phi(-x) net of the permanent
    impact I(x) that already shifted the terminal price.
    """
    return phi_func(x, **phi_kwargs) * (1.0 + r) - phi_func(-x, **phi_kwargs) - linear_impact(x, lam)


def no_manipulation_condition_linear(lam: float, phi: float) -> bool:
    """Absence of price manipulation condition for the linear regime: 2*phi >= lambda (Eq. 2.3/3.3)."""
    return 2.0 * phi >= lam - 1e-12
