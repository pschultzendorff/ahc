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
    - Brooks-Corey with :math:`n_b = 4`

"""

import copy
import logging
import os
import pathlib
import sys
import warnings

import numpy as np
from .run import cp_models, rp_models, run_simulation

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from ..utils import SimulationConfig

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


if __name__ == "__main__":
    results_dir = dirname / "results"
    results_dir.mkdir(exist_ok=True)

    spe10_layer: int = 55
    init_s: float = 0.3
    # Linear
    config = SimulationConfig(
        file_name="linear",
        folder_name=results_dir / "linear_vs_nonlinear" / f"linear_{init_s}",
        solver_name="AHC",
        adaptive_error_ratio=1e-4,
        init_s=init_s,
        rp_model_1=rp_models["linear"],
        rp_model_2=rp_models["linear"],
        cp_model_1=cp_models["None"],
        cp_model_2=cp_models["None"],
        spe10_layer=spe10_layer,
    )
    run_simulation(config)

    # Nonlinear
    config = SimulationConfig(
        file_name="nonlinear",
        folder_name=results_dir / "linear_vs_nonlinear" / f"nonlinear_{init_s}",
        solver_name="AHC",
        adaptive_error_ratio=1e-4,
        init_s=init_s,
        rp_model_1=copy.deepcopy(rp_models["linear"]),
        rp_model_2=copy.deepcopy(rp_models["Brooks-Corey_nb_4"]),
        cp_model_1=copy.deepcopy(cp_models["None"]),
        cp_model_2=copy.deepcopy(cp_models["linear"]),
        spe10_layer=spe10_layer,
    )
    run_simulation(config)

    # Nonlinear but stop at the first homotopy step
    config = SimulationConfig(
        file_name="nonlinear_stop_early",
        folder_name=results_dir
        / "linear_vs_nonlinear"
        / f"nonlinear_stop_early_{init_s}",
        solver_name="AHC",
        adaptive_error_ratio=1.0,  # Stop early
        init_s=init_s,
        rp_model_1=rp_models["linear"],
        rp_model_2=rp_models["Brooks-Corey_nb_4"],
        cp_model_1=cp_models["None"],
        cp_model_2=cp_models["linear"],
        spe10_layer=spe10_layer,
    )
    run_simulation(config)


# endregion
