r"""Study behavior during AHC/Newton iterations. We only run one time step, even if it
 fails.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton
- Newton
- Newton with Appleyard chopping


We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
continuation. The considered model is similar to the heterogeneous 3D models in the
article (section 4.6.4), but on a 2D domain for now.

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
    - Porosity: SPE10 case 2A, layer 55
    - Permeability: SPE10 case 2A, layer 55
- Fluid properties:
    - Water: ``pp.fluid_values.water``. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 4000 psi (= BHP; initial guess for Newton, no influence on the solution)
    - Saturation: 0.2, 0.3, and 0.5.
- Rel. perm. models (model_2; model_1 is linear for AHC):
    - Brooks-Corey-Mualem with n_b=4, eta=2.
- Capillary pressure model (model_2; model_1 is None for AHC):
    - Brooks-Corey with n_b=4, entry pressure 200 Pa.

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
    SPE10_CASE,
    SPE10_LAYER,
    WATER_DENSITY,
    ZERO_BUOYANCY_MODEL,
    cp_models,
    results_dir,
    rp_models,
    run_simulation,
    solvers_and_tols,
)

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

# Setup logging level.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# endregion


# region RUN
spe10_layer = 55
time_manager_params = {
    "schedule": np.array([0.0, 30.0 * pp.DAY]),
    "dt_init": 30.0 * pp.DAY,
    "constant_dt": True,
}


def generate_cases() -> list[SimulationConfig]:
    """Generate all simulation configurations."""

    cases = []

    # Varying init_s for the Brooks-Corey model.
    for init_s in [0.2, 0.3, 0.5]:
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            if solver_name in ["HC", "ReferenceSolution"]:
                # HC solver is not of interest here.
                continue
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous",
                    study="iteration_plotting",
                    case=f"init_s_{init_s:.2f}",
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=init_s,
                    rp_model_1=rp_models["linear"],
                    rp_model_2=rp_models["Brooks-Corey_nb_4"],
                    cp_model_1=cp_models["None"],
                    cp_model_2=cp_models["Brooks-Corey_nb_4"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                    spe10_cell_size=CELL_SIZE,
                    spe10_layer=SPE10_LAYER,
                    spe10_case=SPE10_CASE,
                    spe10_water_density=WATER_DENSITY,
                )
            )

    return cases


if __name__ == "__main__":
    results_dir.mkdir(exist_ok=True)

    study = generate_cases()
    for config in study:
        run_simulation(
            config, time_manager_params=time_manager_params, iteration_exporting=True
        )

# endregion
