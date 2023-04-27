import logging
import os
from datetime import date

import numpy as np
import porepy as pp

from src.tpflab.models.run_models import run_time_dependent_model
from src.tpflab.models.two_phase_flow import TwoPhaseFlow
from src.tpflab.utils import logging_redirect_tqdm, rm_out_padding

# rm_out_padding()

cap_pressure_model = "Brooks-Corey"
params = {
    "formulation": "n_pressure_w_saturation",
    "file_name": f"pcap_{cap_pressure_model}_gravity_off",
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
    def _source_w(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[w_source_cell_index] = 0.5
        return array


model = ModifiedModel(params)

model._grid_size = 20
model._phys_size = 2
model._time_step = 0.1
model._schedule = np.array([0, 120.0])
model._cap_pressure_model = cap_pressure_model
model.prepare_simulation()
with logging_redirect_tqdm([logger]):
    run_time_dependent_model(
        model, {"nl_convergence_tol": 1e-10, "max_iterations": 100}
    )
