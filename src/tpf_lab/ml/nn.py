"""Example neural network classes that can be wrapped and used in PorePy."""

import torch
import torch.nn as nn
from typing import Optional


class BaseNN(nn.Module):
    """Base nn with one input and one output. Depth can be chosen."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__()
        if params is None:
            params = {}
        self._depth: int = params.get("depth", 0)
        self._hidden_size: int = params.get("hidden_size", 30)
        self.fcs = nn.ModuleList(
            [
                nn.Linear(self._hidden_size, self._hidden_size)
                for _ in range(self._depth - 1)
            ]
        )
        self.fcs.insert(0, nn.Linear(1, self._hidden_size))
        self.fcs.append(nn.Linear(self._hidden_size, 1))
        self.act1 = nn.Sigmoid()
        # Use sigmoid for the final layer, to enforce :math:`0\leq S_w\leq1`.
        if params.get("final_act", "sigmoid") == "linear":
            self.act2: nn.Identity | nn.Sigmoid = nn.Identity()
        else:
            self.act2 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs[:-1]:
            x = self.act1(fc(x))
        return self.act2(self.fcs[-1](x))
