import logging
import os
from datetime import date
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp

from src.tpflab.models.run_models import run_time_dependent_model
from src.tpflab.models.two_phase_flow import TwoPhaseFlow
from src.tpflab.utils import logging_redirect_tqdm, rm_out_padding

# rm_out_padding()

cap_pressure_model = "Brooks-Corey"
params = {
    "formulation": "n_pressure_w_saturation",
    "file_name": f"pcap_{cap_pressure_model}_gravity_on",
    "folder_name": os.path.join(
        "results",
        "setup_tests",
        f"{date.today().strftime('%Y-%m-%d')}_pcap_{cap_pressure_model}",
    ),
}

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logger.handlers.clear()
# try:
#     os.makedirs(params["folder_name"])
# except OSError:
#     pass
# fh = logging.FileHandler(
#     os.path.join(params["folder_name"], ".".join([params["file_name"], "txt"]))
# )
# fh.setLevel(logging.DEBUG)
# # formatter = jsonlogger.JsonFormatter()
# # file_handler.setFormatter(formatter)
# sh = logging.StreamHandler()
# sh.setLevel(logging.DEBUG)
# logger.addHandler(sh)
# logger.addHandler(fh)
# logger.setLevel(logging.DEBUG)

w_source_cell_index = 209


class ModifiedModel(TwoPhaseFlow):
    # def _source_w(self, g: pp.Grid) -> np.ndarray:
    #     array: np.ndarray = super()._source_w(g)
    #     array[w_source_cell_index] = 0.5
    #     return array

    # NOTE: The density of both phases is equal, thus the gravity should have no effect.

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Zero vector source (gravity). Corresponds to the wetting buoyancy flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * self._w_density
        """
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        vals[-1] = pp.GRAVITY_ACCELERATION * self._density_w
        return vals

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Zero vector volume source (gravity). Corresponds to the nonwetting buoyancy
        flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * self._n_density
        """
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        vals[-1] = pp.GRAVITY_ACCELERATION * self._density_n
        return vals


model = ModifiedModel(params)

model._grid_size = 20
model._phys_size = 2
model._cap_pressure_model = cap_pressure_model
model._time_step = 0.1
model._density_w = 5.0
model._schedule = np.array([0, 100.0])
model.prepare_simulation()
with logging_redirect_tqdm([logger]):
    run_time_dependent_model(
        model, {"nl_convergence_tol": 1e-10, "max_iterations": 100}
    )
