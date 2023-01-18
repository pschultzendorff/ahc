import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader


from pp_nn import BaseNN

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class RelPermDataset(torch.utils.data.Dataset):
    def __init__(self, len: int = 1000, model: str = "power_w") -> None:
        super().__init__()
        self.len = len
        self.model: str = model
        self.mean = torch.tensor([0.0] * self.len).unsqueeze(-1)
        self.std = torch.tensor([0.1] * self.len).unsqueeze(-1)
        self.s_w = torch.rand([self.len, 1])
        noise = torch.normal(self.mean, self.mean)
        if self.model == "power_w":
            self.target: torch.Tensor = power(self.s_w) + noise
        if self.model == "power_nw":
            self.target = power(1 - self.s_w) + noise

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

    def __iter__(self):
        def iterator(self) -> tuple[torch.Tensor, torch.Tensor]:
            while True:
                s_w = torch.rand([1])
                noise = torch.normal(self.mean, self.mean)
                if self.model == "power_w":
                    target = power(s_w) + noise
                if self.model == "power_nw":
                    target = power(1 - s_w) + noise
                yield s_w, target

        return iterator(self)


def power(s_w: torch.Tensor) -> torch.Tensor:
    return s_w**3


def brooks_corey(s_w: torch.Tensor) -> torch.Tensor:
    pass


w_train_data: RelPermDataset = RelPermDataset(len=200)
w_train_dataloader: DataLoader = DataLoader(w_train_data, batch_size=64)
nw_train_data: RelPermDataset = RelPermDataset(len=200, model="power_nw")
nw_train_dataloader: DataLoader = DataLoader(nw_train_data, batch_size=64)

# w_train_data: IterableRelPermDataset = IterableRelPermDataset()
# w_train_dataloader: DataLoader = DataLoader(w_train_data, batch_size=64)
# nw_train_data: IterableRelPermDataset = IterableRelPermDataset(model="power_nw")
# nw_train_dataloader: DataLoader = DataLoader(nw_train_data, batch_size=64)

model = BaseNN()
trainer = optim.Adam(model.parameters())


def train(data, model: nn.Module, trainer: optim.Optimizer, epochs: int = 5000) -> None:
    """Train for a number of epochs. Since an IterableDataset is used, each epoch is one
    batch.
    """
    loss_func = nn.MSELoss()
    for epoch in range(epochs):
        logger.info(f"epoch {epoch}")
        progress_bar = tqdm.tqdm(data)
        for x, y in progress_bar:
            trainer.zero_grad()
            y_bar = model(x)
            loss = loss_func(y_bar, y)
            loss.backward()
            trainer.step()
            progress_bar.set_description_str(f"loss {loss.numpy(force=True)}")


def train_iterable(data, epochs: int = 5000) -> None:
    """Train for a number of epochs. Since an IterableDataset is used, each epoch is one
    batch.
    """
    progress_bar = tqdm.tqdm(range(epochs))
    for epoch in progress_bar:
        trainer.zero_grad()
        x, y = next(iter(data))
        y_bar = model(x)
        loss = loss_func(y_bar, y)
        loss.backward()
        trainer.step()
        progress_bar.set_description_str(f"epoch {epoch} loss {loss.numpy(force=True)}")
        # logger.info(f"epoch {epoch} loss {loss.numpy(force=True)}")


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
