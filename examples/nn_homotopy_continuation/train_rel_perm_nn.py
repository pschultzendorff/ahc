import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data

from tpf_lab.ml.datasets import DatasetWithNoise
from tpf_lab.ml.nn import BaseNN
from tpf_lab.ml.train import train
from tpf_lab.utils import fix_generator_seed, fix_seeds, seed_worker

# Reproducability
fix_seeds(0)
g = fix_generator_seed(0)

foldername: str = os.path.join("results", "rel_perm_nn")
try:
    os.makedirs(foldername)
except Exception:
    pass

# Generate very few data points
wetting_dataset = DatasetWithNoise(
    len=20,
    model="Corey_w",
    model_params={
        "power": 3,
        "linear_param": 1.0,
        "residual_saturation_w": 0.3,
        "residual_saturation_n": 0.3,
    },
    mean=0.1,
    std=0.2,
)
wetting_dataloader = data.DataLoader(
    wetting_dataset,
    batch_size=4,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)

plt.figure()
for batch in wetting_dataloader:
    x, y = batch
    plt.scatter(x.numpy(), y.numpy())
plt.xlabel(r"$S_w$")
plt.ylabel(r"$k_{r,w}$")
plt.savefig(os.path.join(foldername, "wetting_data.png"))

w_model = BaseNN({"depth": 5, "final_act": "linear"})
trainer = optim.Adam(w_model.parameters())
train(wetting_dataloader, w_model, trainer, epochs=4000)
torch.save(w_model.state_dict(), os.path.join(foldername, "wetting.pt"))

w_model.eval()
with torch.no_grad():
    xx = torch.linspace(0, 1, 300).unsqueeze(-1)
    yy = w_model(xx)

plt.figure()
plt.plot(xx.squeeze().numpy(), yy.squeeze().numpy())
plt.xlabel(r"$S_w$")
plt.ylabel(r"$k_{r,w}$")
plt.savefig(os.path.join(foldername, "wetting_nn.png"))


# Generate very few data points
nonwetting_dataset = DatasetWithNoise(
    len=16,
    model="Corey_n",
    model_params={
        "power": 3,
        "linear_param": 1.0,
        "residual_saturation_w": 0.3,
        "residual_saturation_n": 0.3,
    },
    mean=0.0,
    std=0.2,
)
nonwetting_dataloader = data.DataLoader(
    nonwetting_dataset,
    batch_size=4,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
)

plt.figure()
for batch in nonwetting_dataloader:
    x, y = batch
    plt.scatter(x.numpy(), y.numpy())
plt.xlabel(r"$S_w$")
plt.ylabel(r"$k_{r,n}$")
plt.savefig(os.path.join(foldername, "nonwetting_data.png"))


n_model = BaseNN({"depth": 7, "final_act": "linear"})
trainer = optim.Adam(n_model.parameters())
train(nonwetting_dataloader, n_model, trainer, epochs=4000)
torch.save(n_model.state_dict(), os.path.join(foldername, "nonwetting.pt"))

n_model.eval()
with torch.no_grad():
    xx = torch.linspace(0, 1, 300).unsqueeze(-1)
    yy = n_model(xx)

plt.figure()
plt.plot(xx.squeeze().numpy(), yy.squeeze().numpy())
plt.xlabel(r"$S_w$")
plt.ylabel(r"$k_{r,n}$")
plt.savefig(os.path.join(foldername, "nonwetting_nn.png"))


with torch.no_grad():
    xx = torch.linspace(0, 1, 300).unsqueeze(-1)
    yy = w_model(xx) / (w_model(xx) + n_model(xx))

plt.figure()
plt.plot(xx.squeeze().numpy(), yy.squeeze().numpy())
plt.xlabel(r"$S_w$")
plt.ylabel(r"$f_w$")
plt.savefig(os.path.join(foldername, "f_w_nn.png"))
