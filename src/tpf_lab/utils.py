import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch
from porepy.models import run_models

# Get module wide logger.
logger = logging.getLogger(__name__)


def save_params_and_run_model(model, params: dict[str, Any]) -> None:
    """Save the model parameters in a ``*.json`` file and run the model."""
    # Create save folder and save params.
    folder_name = params["folder_name"]
    try:
        os.makedirs(folder_name)
    except Exception:
        pass
    params_json = {}
    # Some values might not be JSON serializable. Create a new dict to avoid this issue.
    for key, value in params.items():
        try:
            json.dumps(value)
            params_json[key] = value
        except Exception:
            # ``pp.TimeManager`` is not json serializable, so we save its most important
            # attributes manually.
            if key == "time_manager":
                params_json[key] = {
                    "schedule": [float(value.schedule[0]), float(value.schedule[1])],
                    "dt_init": value.dt_init,
                    "constant_dt": value.is_constant,
                }
            pass
    with open(os.path.join(folder_name, "model_params.json"), "w") as f:
        json.dump(params_json, f, indent=2)
    # Run model.
    if model.is_time_dependent():
        run_models.run_time_dependent_model(model, params)
    else:
        run_models.run_stationary_model(model, params)


# We provide the functionality to fix rng for reproducibility.
# Details are explained here: https://pytorch.org/docs/stable/notes/randomness.html
def fix_seeds(id: int = 0) -> None:
    random.seed(id)
    np.random.seed(id)
    torch.manual_seed(id)


def fix_generator_seed(id: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(id)
    return g


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
