"""Training functions."""

import torch.nn as nn
import torch.optim as optim
import tqdm


def train(data, model: nn.Module, trainer: optim.Optimizer, epochs: int = 5000) -> None:
    """Train for a number of epochs."""
    loss_func = nn.MSELoss()
    epoch_bar = tqdm.tqdm(range(epochs), position=0)
    for epoch in epoch_bar:
        epoch_bar.set_description(f"epoch {epoch}")
        progress_bar = tqdm.tqdm(data, position=1, leave=None)
        for x, y in progress_bar:
            trainer.zero_grad()
            y_bar = model(x)
            loss = loss_func(y_bar, y)
            loss.backward()
            trainer.step()
            progress_bar.set_description_str(f"loss {loss.numpy(force=True)}")


def train_iterable(
    data, model: nn.Module, trainer: optim.Optimizer, epochs: int = 5000
) -> None:
    """Train for a number of epochs.

    Since an :obj:`~torch.utils.data.IterableDataset` is used, each epoch is one batch.

    """
    loss_func = nn.MSELoss()
    progress_bar = tqdm.tqdm(range(epochs))
    for epoch in progress_bar:
        trainer.zero_grad()
        x, y = next(iter(data))
        y_bar = model(x)
        loss = loss_func(y_bar, y)
        loss.backward()
        trainer.step()
        progress_bar.set_description_str(f"epoch {epoch} loss {loss.numpy(force=True)}")
