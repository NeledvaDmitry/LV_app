from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Parameters:
    alpha: float
    beta: float
    gamma: float
    delta: float


@dataclass
class InitialConditions:
    x0: float
    y0: float
    t_start: float
    t_end: float
    n_points: int = 2000
    method: str = "DOP853"
    rtol: float = 1e-9
    atol: float = 1e-12


def lotka_volterra_rhs(t, z, p: Parameters):
    x, y = z
    dxdt = p.alpha * x - p.beta * x * y
    dydt = -p.gamma * y + p.delta * x * y
    return [dxdt, dydt]


def solve_lotka_volterra(p: Parameters, ic: InitialConditions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_eval = np.linspace(ic.t_start, ic.t_end, ic.n_points)
    sol = solve_ivp(
        lambda t, z: lotka_volterra_rhs(t, z, p),
        (ic.t_start, ic.t_end),
        [ic.x0, ic.y0],
        t_eval=t_eval,
        method=ic.method,
        rtol=ic.rtol,
        atol=ic.atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    t = sol.t
    x = sol.y[0]
    y = sol.y[1]
    return t, x, y


def invariant_values(x: np.ndarray, y: np.ndarray, p: Parameters) -> np.ndarray:
    eps = 1e-12
    x_safe = np.maximum(x, eps)
    y_safe = np.maximum(y, eps)
    return (
        p.delta * x_safe
        - p.gamma * np.log(x_safe)
        + p.beta * y_safe
        - p.alpha * np.log(y_safe)
    )


def equilibrium_point(p: Parameters) -> Tuple[float, float]:
    x_eq = p.gamma / p.delta
    y_eq = p.alpha / p.beta
    return x_eq, y_eq