import numpy as np
import torch
import torch.nn as nn


class RelPermNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 30)
        self.fc4 = nn.Linear(30, 1)
        self.act1 = nn.ReLU()
        # Use sigmoid for the final layer, to enforce :math:0\leq S_w\leq1.
        self.act2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.act1(self.fc3(x))
        return self.act2(self.fc4(x))


def error_func_deriv(
    x: np.ndarray, yscale: float = 1.0, xscale: float = 200, offset: float = 0.5
) -> np.ndarray:
    return yscale * np.exp(-xscale * (x - offset) ** 2)


def brookscorey_w(
    s: np.ndarray,
    n1: float = 2.0,
    n2: float = 3.0,
    n3: float = 1.0,
    residual_sat_w: float = 0.1,
    residual_sat_n: float = 0.0,
) -> np.ndarray:
    normalized_s = (s - residual_sat_w) / (1 - residual_sat_n - residual_sat_w)
    return normalized_s ** (n1 + n2 * n3)


def brookscorey_n(
    s: np.ndarray,
    n1: float = 2.0,
    n2: float = 3.0,
    n3: float = 1.0,
    residual_sat_w: float = 0.1,
    residual_sat_n: float = 0.0,
) -> np.ndarray:
    normalized_s = (s - residual_sat_w) / (1 - residual_sat_n - residual_sat_w)
    return normalized_s ** (n1 + n2 * n3)
