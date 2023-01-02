import torch
import torch.nn as nn
from typing import Optional


class BaseNN(nn.Module):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__()
        if params is None:
            params = {}
        self._depth: int = params.get("depth", 1)
        self._hidden_size: int = params.get("hidden_size", 30)
        # TODO: Implement a layer list with the given depth.
        self.fc1 = nn.Linear(1, self._hidden_size)
        self.fc2 = nn.Linear(self._hidden_size, 1)
        # Use sigmoid for the final layer, to enforce :math:0\leq S_w\leq1.
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.act(self.fc2(x))
