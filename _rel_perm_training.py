import os
from functools import partial
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from tpf_lab.ml.nn import BaseNN
from tpf_lab.ml.datasets import (
    DatasetWithNoise,
    RelPermW_BrooksCorey,
    RelPermN_BrooksCorey,
)
from src.tpf_lab.ml.train import train

# Direct input
plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
# Options
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 11,
        "font.family": "lmodern",
        "text.latex.unicode": True,
    }
)


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
