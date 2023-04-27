"""Numpy rel. perm. functions for plotting etc."""

import numpy as np


def error_func_deriv(
    x: np.ndarray, yscale: float = 1.0, xscale: float = 200, offset: float = 0.5
) -> np.ndarray:
    return yscale * np.exp(-xscale * (x - offset) ** 2)


def brookscorey_w(
    S_w: np.ndarray,
    n_1: float = 2.0,
    n_2: float = 3.0,
    n_3: float = 1.0,
    residual_sat_w: float = 0.1,
    residual_sat_n: float = 0.0,
) -> np.ndarray:
    normalized_s = (S_w - residual_sat_w) / (1 - residual_sat_n - residual_sat_w)
    k_rw = normalized_s ** (n_1 + n_2 * n_3)
    return k_rw


def brookscorey_n(
    S_w: np.ndarray,
    n_1: float = 2.0,
    n_2: float = 3.0,
    n_3: float = 1.0,
    residual_sat_w: float = 0.1,
    residual_sat_n: float = 0.0,
) -> np.ndarray:
    normalized_s = (S_w - residual_sat_w) / (1 - residual_sat_n - residual_sat_w)
    k_rn = (1 - normalized_s) ** n_1 * (1 - normalized_s**n_2) ** n_3
    return k_rn
