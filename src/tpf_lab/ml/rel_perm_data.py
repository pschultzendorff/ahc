import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from pp_nn import BaseNN

from train import train


# Direct input
plt.rcParams["text.latex.preamble"] = [r"\usepackage{lmodern}"]
# Options
params = {
    "text.usetex": True,
    "font.size": 11,
    "font.family": "lmodern",
    "text.latex.unicode": True,
}


class RelPermDatasetW(torch.utils.data.Dataset):
    def __init__(self, len: int = 1000, model: str = "Brooks-Corey") -> None:
        super().__init__()
        self.len = len
        self.model: str = model
        self.mean = torch.tensor([0.3] * self.len).unsqueeze(-1)
        self.std = torch.tensor([0.2] * self.len).unsqueeze(-1)
        self.s_w = torch.rand([self.len, 1])
        noise = torch.normal(self.mean, self.std) * self.s_w
        biased_noise = torch.where(self.s_w >= 0.5, torch.zeros_like(noise), noise)
        if self.model == "Brooks-Corey":
            self.gen_func = BrooksCoreyW()
            self.target: torch.Tensor = self.gen_func(self.s_w) + biased_noise

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.s_w[index], self.target[index]


class RelPermDatasetN(torch.utils.data.Dataset):
    def __init__(self, len: int = 1000, model: str = "Brooks-Corey") -> None:
        super().__init__()
        self.len = len
        self.model: str = model
        self.mean = torch.tensor([0.1] * self.len).unsqueeze(-1)
        self.std = torch.tensor([0.2] * self.len).unsqueeze(-1)
        self.s_w = torch.rand([self.len, 1])
        noise = torch.normal(self.mean, self.std)
        if self.model == "Brooks-Corey":
            self.gen_func = BrooksCoreyN()
            self.target: torch.Tensor = self.gen_func(self.s_w) + noise

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.s_w[index], self.target[index]


class BrooksCoreyW(nn.Module):
    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.n_1: int = int(params.get("n_1", 2))
        self.n_2: int = int(params.get("n_3", 3))
        self.n_3: int = int(params.get("n_2", 1))
        self.w_res_sat: float = params.get("w_res_sat", 0.1)
        """Wetting residual saturation."""
        self.n_res_sat: float = params.get("n_res_sat", 0.0)
        """Nonwetting residual saturation."""

    def forward(self, s_w: torch.Tensor) -> torch.Tensor:
        normalized_s_w = (s_w - self.w_res_sat) / (1 - self.n_res_sat - self.w_res_sat)
        k_rw = normalized_s_w ** (self.n_1 + self.n_2 * self.n_3)
        return k_rw


class BrooksCoreyN(nn.Module):
    def __init__(self, params: Optional[dict] = None) -> None:  #
        super().__init__()
        if params is None:
            params = {}
        self.n_1: int = int(params.get("n_1", 2))
        self.n_2: int = int(params.get("n_3", 3))
        self.n_3: int = int(params.get("n_2", 1))
        self.w_res_sat: float = params.get("w_res_sat", 0.1)
        """Wetting residual saturation."""
        self.n_res_sat: float = params.get("n_res_sat", 0.0)
        """Nonwetting residual saturation."""

    def forward(self, s_w: torch.Tensor) -> torch.Tensor:
        normalized_s_w = (s_w - self.w_res_sat) / (1 - self.n_res_sat - self.w_res_sat)
        k_rn = (1 - normalized_s_w) ** self.n_1 * (
            1 - normalized_s_w**self.n_2
        ) ** self.n_3
        return k_rn


# Wetting
brooks_corey_w = BrooksCoreyW()
w_train_data = RelPermDatasetW(len=64)
w_train_dataloader = DataLoader(w_train_data, batch_size=64)

model = BaseNN({"depth": 1, "act": "linear"})
trainer = optim.Adam(model.parameters())
train(w_train_dataloader, model, trainer, epochs=10000)
torch.save(
    model.state_dict(),
    os.path.join(
        "saved_models", "BaseNN_RelPermW_BrooksCorey_1HiddenLayers_BiasedNoise.pt"
    ),
)

x = torch.arange(0, 1, 0.01).unsqueeze(-1)
truth = brooks_corey_w(x)
y = model(x)
plt.figure()
plt.plot(x.numpy(force=True), truth.numpy(force=True), label="Ground truth")
plt.plot(x.numpy(force=True), y.numpy(force=True), label="NN")
x, y = next(iter(w_train_dataloader))
plt.scatter(
    x.numpy(force=True),
    y.numpy(force=True),
    label="Data",
)
plt.legend()
plt.xlabel(r"$S_w$")
plt.ylabel(r"$k_{r,w}$")
plt.savefig(
    os.path.join(
        "saved_models", "BaseNN_RelPermW_BrooksCorey_1HiddenLayers_BiasedNoise.png"
    )
)

# Nonwetting
# brooks_corey_n = BrooksCoreyN()
# n_train_data = RelPermDatasetN(len=64)
# n_train_dataloader = DataLoader(n_train_data, batch_size=64)

# model = BaseNN({"depth": 1, "act": "linear"})
# trainer = optim.Adam(model.parameters())
# train(n_train_dataloader, model, trainer)
# torch.save(
#     model.state_dict(),
#     os.path.join("saved_models", "BaseNN_RelPermN_BrooksCorey_1HiddenLayers.pt"),
# )

# x = torch.arange(0, 1, 0.01).unsqueeze(-1)
# truth = brooks_corey_n(x)
# y = model(x)
# plt.figure()
# plt.plot(x.numpy(force=True), truth.numpy(force=True), label="Ground truth")
# plt.plot(x.numpy(force=True), y.numpy(force=True), label="NN")
# x, y = next(iter(n_train_dataloader))
# plt.scatter(
#     x.numpy(force=True),
#     y.numpy(force=True),
#     label="Data",
# )
# plt.legend()
# plt.xlabel(r"$S_w$")
# plt.ylabel(r"$k_{r,n}$")
# plt.savefig(
#     os.path.join("saved_models", "BaseNN_RelPermN_BrooksCorey_1HiddenLayers.png")
# )
