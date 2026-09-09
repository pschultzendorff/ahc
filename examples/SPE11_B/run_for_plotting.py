"""Run the SPE11 B example with small time steps for plotting.


Model description:
- Constant CO2 injection in the center.
- No flow boundary condition on the sides and bottom. Homogeneous Dirichlet on top.
- Simulation time: 3000 days
- Solid properties:
    - Porosity: SPE11, case B.
    - Permeability: SPE11, case B.
- Fluid properties:
    - Water: ``pp.fluid_values.water``. Residual saturation is 0.15.
    - CO2: Reservoir conditions (350 bar, 70°C). Residual saturation is 0.1.
- Initial values:
    - Pressure: 30 MPa (reservoir pressure).
    - Saturation: 0.8 (water-filled domain).
- Rel. perm. model (Brooks-Corey-Mualem with n_b=4, eta=2).
- Capillary pressure model (Brooks-Corey with n_b=4).

"""

import logging
import os
import pathlib
import sys
import warnings

import numpy as np
import porepy as pp
from run import (
    SPE11_ENTRY_PRESSURE,
    ZERO_BUOYANCY_MODEL,
    cp_models,
    rp_models,
    run_simulation,
)

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig

# region SETUP


# Limit number of threads to one to ensure that pypardiso is deterministic.
N_THREADS = "1"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow, which may occur when calculating estimators.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Directories for results.
dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()
results_dir = dirname / "results_plotting"

# endregion


# region RUN

time_manager_params = {
    "schedule": np.array([0.0, 3000.0 * pp.DAY]),
    "dt_init": 10.0 * pp.DAY,
    "constant_dt": True,
}

if __name__ == "__main__":
    results_dir.mkdir(exist_ok=True)

    config = SimulationConfig(
        results_dir=results_dir,
        regime="viscous",
        study="plotting",
        case="plotting",
        solver_name="NewtonAppleyard",
        hc_tol=0.0,  # Disregarded
        nl_tol=1e-5,  # Almost disregarded
        init_s=0.8,
        rp_model_1=rp_models["linear"],
        rp_model_2=rp_models["Brooks-Corey_nb_4"],
        cp_model_1=cp_models["None"],
        cp_model_2=cp_models["Brooks-Corey_nb_4"],
        buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
        buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
        spe11_refinement_factor=1.0,
        spe11_entry_pressure=SPE11_ENTRY_PRESSURE,  # [Pa]
    )
    run_simulation(config, time_manager_params=time_manager_params)
# endregion
