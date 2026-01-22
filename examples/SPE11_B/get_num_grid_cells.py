r"""Run the SPE11 A example with small time steps for plotting.


Model description:
- Constant CO2 injection in the center.
- No flow boundary condition on the sides and bottom. Homogeneous Dirichlet on top.
- Simulation time: 10 days
- Solid properties:
    - Porosity: SPE11, case A.
    - Permeability: SPE11, case A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.1.
    - CO2: From the NIST database, taken at 20°C and atmospheric pressure. Residual
      saturation is 0.1.
- Initial values:
    - Pressure: Atmospheric pressure.
    - Saturation: Varying between 0.8 and 0.9.
- Rel. perm. models:
    - linear
    - Brooks-Corey
- Capillary pressure model:
    - None

"""

import logging
import os
import pathlib
import sys
import warnings

import numpy as np
import porepy as pp
from run import cp_models, refinement_factors, rp_models, run_simulation

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig

# region SETUP


# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow, which may occur when calculating estimators.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

# endregion


# region RUN

time_manager_params = {
    "schedule": np.array([0.0, 10.0 * pp.DAY]),
    "dt_init": 10.0 * pp.DAY,
    "constant_dt": True,
}

if __name__ == "__main__":
    results_dir = dirname / "results"
    results_dir.mkdir(exist_ok=True)
    for refinement_factor in refinement_factors:
        config = SimulationConfig(
            file_name="num_grid_cells",
            folder_name=results_dir / f"num_grid_cells_{refinement_factor}",
            solver_name="NewtonAppleyard",  # NewtonAppleyard requires too many time steps
            adaptive_error_ratio=1000.0,  # Doesn't matter
            refinement_factor=refinement_factor,
            init_s=0.8,
            rp_model_1=rp_models["linear"],
            rp_model_2=rp_models["Brooks-Corey_nb_4"],
            cp_model_1=cp_models["None"],
            cp_model_2=cp_models["Brooks-Corey_nb_4"],
        )
        run_simulation(config, time_manager_params=time_manager_params)

# endregion
