"""Run the simulation with small uniform time steps for fancy plotting.

We loosely follow the setup of Wang and Tchelepi (2013). The considered model is similar
to the heterogeneous 3D models in the article (section 4.6.4), but on a 2D domain.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 1200x2200 ft domain
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 30 days
- Solid properties:
    - Porosity: Layers 10 and 55 of SPE10, case 2A.
    - Permeability: Layers 10 and 55 of SPE10, case 2A.
- Fluid properties:
    - Water: ``pp.fluid_values.water``. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Saturation: 0.3.
- Rel. perm. model:
    - Brooks-Corey-Mualem.
- Capillary pressure model:
    - Brooks-Corey-Mualem, entry pressure 200 Pa.

"""

import logging
import os
import pathlib
import sys
import warnings

import numpy as np
import porepy as pp
from run import (
    CELL_SIZE,
    LINEAR_RP_MODEL,
    WATER_DENSITY,
    ZERO_BUOYANCY_MODEL,
    ZERO_CP_MODEL,
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

# Setup logging level.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Directories for results.
dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()
results_dir = dirname / "results_plotting"

# endregion


# region RUN
time_manager_params = {
    "schedule": np.array([0.0, 30.0 * pp.DAY]),
    "dt_init": 1.0 * pp.DAY,
    "constant_dt": True,
}


if __name__ == "__main__":
    results_dir.mkdir(exist_ok=True)
    for spe10_layer in [10, 55]:
        config = SimulationConfig(
            results_dir=results_dir,
            regime="five_spot",
            study="plotting",
            case=f"layer_{spe10_layer}",
            solver_name="ReferenceSolution",
            hc_tol=0.01,
            nl_tol=0.01,
            init_s=0.3,
            rp_model_1=LINEAR_RP_MODEL,
            rp_model_2=rp_models["Brooks-Corey_nb_4"],
            cp_model_1=ZERO_CP_MODEL,
            cp_model_2=cp_models["Brooks-Corey_nb_4"],
            buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
            buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
            spe10_cell_size=CELL_SIZE,
            spe10_layer=spe10_layer,
            spe10_case="five_spot",
            spe10_water_density=WATER_DENSITY,
        )
        run_simulation(config, time_manager_params=time_manager_params)

# endregion
