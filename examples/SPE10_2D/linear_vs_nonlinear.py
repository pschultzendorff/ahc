r"""Run the simulation once with linear and once with nonlinear constitutive laws to
 compare the difference.

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
    - Pressure: 6000 psi (initial guess for Newton, no influence on the solution)
    - Saturation: 0.3
- Rel. perm. models:
    - linear
    - Brooks-Corey-Mualem with :math:`n_b = 4, \eta = 2`
- Capillary pressure model:
    - None
    - Brooks-Corey with :math:`n_b = 4`, entry pressure 200 Pa

"""

import copy
import logging
import os
import pathlib
import sys
import warnings

import numpy as np
from run import (
    CELL_SIZE,
    LINEAR_RP_MODEL,
    SPE10_CASE,
    SPE10_LAYER,
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

# Directories for results.
dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()
results_dir = dirname / "results_linear_vs_nonlinear"

# endregion


# region RUN


if __name__ == "__main__":
    results_dir.mkdir(exist_ok=True)

    init_s: float = 0.3

    # Linear
    config = SimulationConfig(
        results_dir=results_dir,
        regime="viscous",
        study="linear_vs_nonlinear",
        case=f"linear_{init_s}",
        solver_name="AHC",
        hc_tol=1e-4,
        nl_tol=0.1,
        init_s=init_s,
        rp_model_1=LINEAR_RP_MODEL,
        rp_model_2=LINEAR_RP_MODEL,
        cp_model_1=ZERO_CP_MODEL,
        cp_model_2=ZERO_CP_MODEL,
        buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
        buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
        spe10_cell_size=CELL_SIZE,
        spe10_layer=SPE10_LAYER,
        spe10_case=SPE10_CASE,
        spe10_water_density=WATER_DENSITY,
    )
    run_simulation(config)

    # Nonlinear
    config = SimulationConfig(
        results_dir=results_dir,
        regime="viscous",
        study="linear_vs_nonlinear",
        case=f"nonlinear_{init_s}",
        solver_name="AHC",
        hc_tol=1e-4,
        nl_tol=0.1,
        init_s=init_s,
        rp_model_1=copy.deepcopy(LINEAR_RP_MODEL),
        rp_model_2=copy.deepcopy(rp_models["Brooks-Corey_nb_4"]),
        cp_model_1=copy.deepcopy(ZERO_CP_MODEL),
        cp_model_2=copy.deepcopy(cp_models["Brooks-Corey_nb_4"]),
        buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
        buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
        spe10_cell_size=CELL_SIZE,
        spe10_layer=SPE10_LAYER,
        spe10_case=SPE10_CASE,
        spe10_water_density=WATER_DENSITY,
    )
    run_simulation(config)

    # Nonlinear but stop at the first homotopy step
    config = SimulationConfig(
        results_dir=results_dir,
        regime="viscous",
        study="linear_vs_nonlinear",
        case=f"nonlinear_stop_early_{init_s}",
        solver_name="AHC",
        hc_tol=1e-4,
        nl_tol=0.1,
        init_s=init_s,
        rp_model_1=LINEAR_RP_MODEL,
        rp_model_2=rp_models["Brooks-Corey_nb_4"],
        cp_model_1=ZERO_CP_MODEL,
        cp_model_2=cp_models["Brooks-Corey_nb_4"],
        buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
        buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
        spe10_cell_size=CELL_SIZE,
        spe10_layer=SPE10_LAYER,
        spe10_case=SPE10_CASE,
        spe10_water_density=WATER_DENSITY,
    )
    run_simulation(config)


# endregion
