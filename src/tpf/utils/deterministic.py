"""Provides functionality to fix rng for reproducibility.

Details are explained here: https://pytorch.org/docs/stable/notes/randomness.html

"""

import random

import numpy as np
import torch


def fix_seeds(id: int = 0) -> None:
    random.seed(id)
    np.random.seed(id)
    torch.manual_seed(id)


def fix_generator_seed_torch(id: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(id)
    return g


def fix_generator_seed_numpy(id: int = 0) -> np.random.Generator:
    return np.random.default_rng(id)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
