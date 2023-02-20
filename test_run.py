import os
from datetime import date
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp

from src.tpf_lab.models.run_models import run_time_dependent_model

# from porepy.models.run_models import run_time_dependent_model
from src.tpf_lab.models.two_phase_flow import TwoPhaseFlow


w_source_cell_index = 209


class ModifiedModel(TwoPhaseFlow):
    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._w_source(g)
        array[w_source_cell_index] = 0.0
        return array

    def _source_w(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[w_source_cell_index] = 0.5
        return array


cap_pressure_model = "Brooks-Corey"

model = ModifiedModel(
    {
        "formulation": "n_pressure_w_saturation",
        "file_name": f"{date.today().strftime('%Y-%m-%d')}",
        "folder_name": os.path.join(
            "results",
            "setup_tests",
            f"{date.today().strftime('%Y-%m-%d')}_pcap_{cap_pressure_model}",
        ),
    }
)

model._grid_size = 20
model._phys_size = 2
model._cap_pressure_model = cap_pressure_model
model._time_step = 0.1
model._schedule = np.array([0, 30.0])
model.prepare_simulation()
run_time_dependent_model(model, {"nl_convergence_tol": 1e-10, "max_iterations": 30})
