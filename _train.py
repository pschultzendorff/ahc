import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader


from tpf_lab.ml.nn import BaseNN


from tpf_lab.ml.datasets import (
    DatasetWithNoise,
    IterableDatasetWithNoise,
    RelPermW_BrooksCorey,
    RelPermN_BrooksCorey,
)

model = BaseNN()
trainer = optim.Adam(model.parameters())


w_train_data: RelPermDataset = RelPermDataset(len=200)
w_train_dataloader: DataLoader = DataLoader(w_train_data, batch_size=64)
nw_train_data: RelPermDataset = RelPermDataset(len=200, model="power_nw")
nw_train_dataloader: DataLoader = DataLoader(nw_train_data, batch_size=64)

# w_train_data: IterableRelPermDataset = IterableRelPermDataset()
# w_train_dataloader: DataLoader = DataLoader(w_train_data, batch_size=64)
# nw_train_data: IterableRelPermDataset = IterableRelPermDataset(model="power_nw")
# nw_train_dataloader: DataLoader = DataLoader(nw_train_data, batch_size=64)


# Wetting
# train(w_train_dataloader, model, trainer)
# torch.save(
#     model.state_dict(),
#     os.path.join("saved_models", "BaseNN_RelPerm_Wetting_200_datapoints.pt"),
# )

x = torch.arange(0, 1, 0.01).unsqueeze(-1)
truth = power(x)
y = model(x)
plt.plot(x.numpy(force=True), truth.numpy(force=True), label="Ground truth")
plt.plot(x.numpy(force=True), y.numpy(force=True), label="NN")
plt.legend()
plt.savefig(os.path.join("saved_models", "BaseNN_RelPerm_Wetting_200_datapoints.png"))

# Nonwetting
# model = BaseNN()
# trainer = optim.Adam(model.parameters())

# train(nw_train_dataloader)
# torch.save(
#     model.state_dict(), os.path.join("saved_models", "BaseNN_RelPerm_NonWetting.pt")
# )

# x = torch.arange(0, 1, 0.01).unsqueeze(-1)
# truth = power(1 - x)
# y = model(x)
# plt.figure()
# plt.plot(x.numpy(force=True), truth.numpy(force=True), label="Ground truth")
# plt.plot(x.numpy(force=True), y.numpy(force=True), label="NN")
# plt.legend()
# plt.savefig(os.path.join("saved_models", "BaseNN_RelPerm_Nonwetting.png"))
