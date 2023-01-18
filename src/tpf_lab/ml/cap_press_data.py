import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader


class CapPressDataset(torch.utils.data.Dataset):
    def __init__(self, len: int = 1000, model: str = "Brooks-Corey") -> None:
        super().__init__()
        self.len = len
        self.model: str = model
        self.mean = torch.tensor([1.5] * self.len).unsqueeze(-1)
        self.std = torch.tensor([1.5] * self.len).unsqueeze(-1)
        self.s_w = torch.rand([self.len, 1])
        noise = torch.normal(self.mean, self.mean)
        if self.model == "Brooks-Corey":
            self.gen_func = BrooksCorey()
            self.target: torch.Tensor = self.gen_func(self.s_w) + noise

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.s_w[index], self.target[index]


class IterableRelPermDataset(torch.utils.data.IterableDataset):
    def __init__(self, model: str = "power_w") -> None:
        super().__init__()
        self.model: str = model
        self.mean = torch.tensor([0.0])
        self.std = torch.tensor([0.1])
        if self.model == "brooks_corey":
            self.gen_func = BrooksCorey()

    def __iter__(self):
        def iterator(self) -> tuple[torch.Tensor, torch.Tensor]:
            while True:
                s_w = torch.rand([1])
                noise = torch.normal(self.mean, self.mean)
                target = self.gen_func(s_w) + noise
                yield s_w, target

        return iterator(self)


class BrooksCorey(nn.Module):
    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.w_psi: float = params.get("w_psi", 0.5)
        """Wetting fluid pore size index."""
        self.n_psi: float = params.get("n_psi", 0.5)
        """Nonwetting fluid pore size index."""
        self.w_thresh_press: float = params.get("w_thresh_press", 1.5)
        """Wetting threshold pressure."""
        self.n_thresh_press: float = params.get("n_thresh_press", -1.5)
        """Nonwetting threshold pressure."""
        self.w_res_sat: float = params.get("w_res_sat", 0.3)
        """Wetting residual saturation."""
        self.n_res_sat: float = params.get("n_res_sat", 0.3)
        """Nonwetting residual saturation."""

    def forward(self, s_w: torch.Tensor) -> torch.Tensor:
        normalized_s_w = (s_w - self.w_res_sat) / (1 - self.n_res_sat - self.w_res_sat)
        normalized_s_n = (1 - s_w - self.w_res_sat) / (
            1 - self.n_res_sat - self.w_res_sat
        )
        p_cw = self.w_thresh_press / (normalized_s_w**self.w_psi)
        p_cn = self.n_thresh_press / (normalized_s_n**self.n_psi)
        return p_cw + p_cn


brooks_corey = BrooksCorey()
w_train_data = CapPressDataset(len=200)
w_train_dataloader = DataLoader(w_train_data, batch_size=200)
x = torch.arange(0, 1, 0.01).unsqueeze(-1)
truth = brooks_corey(x)
plt.plot(x.numpy(force=True), truth.numpy(force=True), label="Ground Truth")
x, y = next(iter(w_train_dataloader))
plt.scatter(
    x.numpy(force=True),
    y.numpy(force=True),
    label="Data",
)
plt.legend()
plt.savefig(os.path.join("saved_models", "BaseNN_CapPressHysteresis_data.png"))
