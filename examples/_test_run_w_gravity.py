import logging
import os
from datetime import date
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp

from tpf_lab.models.run_models import run_time_dependent_model
from tpf_lab.models.two_phase_flow import TwoPhaseFlow
from tpf_lab.utils import logging_redirect_tqdm

w_source_cell_index = 209
density_w = 1.4
density_n = 1.0

cap_pressure_model = "Brooks-Corey"
params = {
    "formulation": "n_pressure_w_saturation",
    "file_name": f"pcap_{cap_pressure_model}_gravity_on_density_w{density_w}_density_n{density_n}_w_source_{w_source_cell_index}",
    "folder_name": os.path.join(
        "results",
        "setup_tests",
        f"{date.today().strftime('%Y-%m-%d')}_pcap_{cap_pressure_model}",
    ),
}

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
            vals[-1] = pp.GRAVITY_ACCELERATION * self._w_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * self._density_w
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Zero vector volume source (gravity). Corresponds to the nonwetting buoyancy
        flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = pp.GRAVITY_ACCELERATION * self._n_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * self._density_n
        return vals.ravel()

    def _source_w(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[w_source_cell_index] = 0.5
        return array


model = ModifiedModel(params)

model._grid_size = 20
model._phys_size = 2
model._time_step = 0.1
model._schedule = np.array([0, 100.0])
model._density_w = density_w
model._density_n = density_n
model._cap_pressure_model = cap_pressure_model
model.prepare_simulation()
with logging_redirect_tqdm([logger]):
    run_time_dependent_model(
        model, {"nl_convergence_tol": 1e-10, "max_iterations": 100}
    )
