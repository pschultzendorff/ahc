"""Some test runs with the ``two_phase_flow_model.TwoPhaseFlow`` class. 

If not noted otherwise, the runs were performed on the 2D unit square with a cell size
of 0.01.

"""

import os

import numpy as np
import porepy as pp
from tpf_lab.models.two_phase_flow import (
    BoundaryConditionsTPF,
    EquationsTPF,
    SolutionStrategyTPF,
    VariablesTPF,
)
from tpf_lab.utils import save_params_and_run_model


class WettingInflux(EquationsTPF):
    def _source_w(self, g: pp.Grid) -> np.ndarray:
        array = super()._source_w(g)
        array[209] = 1 / 0.025
        return array


class Setup(  # type: ignore
    WettingInflux,
    VariablesTPF,
    BoundaryConditionsTPF,
    SolutionStrategyTPF,
    #
    pp.ModelGeometry,
    #
    pp.DataSavingMixin,
): ...


# Simple test run
# Neumann bc on three sides, Dirichlet bc on one side.
params = {
    "progressbars": True,
    "meshing_arguments": {"cell_size": 0.05},
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 0.1]),
        dt_init=0.1 * 0.0025,
        constant_dt=True,
    ),
    "cap_pressure_model": "linear",
    "limit_rel_perm": True,
    "file_name": "setup",
    "folder_name": os.path.join("two_phase_flow_setup_runs", "wetting_injection"),
}
save_params_and_run_model(Setup(params), params)
