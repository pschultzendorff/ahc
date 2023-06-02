import json
import os
import logging
from typing import Any

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
            pass
    with open(os.path.join(folder_name, "model_params.json"), "w") as f:
        json.dump(params_json, f, indent=2)
    # Run model.
    if model.is_time_dependent():
        run_models.run_time_dependent_model(model, params)
    else:
        run_models.run_stationary_model(model, params)
