r"""Run the simulation once with linear and once with nonlinear constitutive laws to
 compare the difference.


We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
continuation. The considered model is similar to the heterogeneous 3D models in the
article (section 4.6.4), but on a 2D domain for now.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 600x1100 ft domain (we just take a quarter of the original SPE10 domain)
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 10 days
- Solid properties:
    - Porosity: Uppermost layer of the SPE10, case 2A.
    - Permeability: Uppermost layer of the SPE10, case 2A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 6000 psi
    - Saturation: Varying between 0.2 and 0.3.
- Rel. perm. models:
    - linear
    - Corey with power .
    - Corey with power 3
    - Brooks-Corey
- Capillary pressure model:
    - None
    - Brooks-Corey

"""

import copy
import importlib
import logging
import os
import pathlib
import sys
import warnings

import numpy as np
import run
from run import cp_models, rp_models

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig

# region SETUP


# NOTE Limit number of threads for NREC to 1 to ensure nothing weird with parallelism.
N_THREADS = "1"
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

    for init_s in [0.3]:
        config = SimulationConfig(
            file_name="run_1",
            folder_name=results_dir / "mwe_status_bug" / "run_1",
            solver_name="AHC",
            adaptive_error_ratio=1e-4,
            init_s=init_s,
            # NOTE deepcopy dictionaries to make sure nothing is changed here.
            rp_model_1=copy.deepcopy(rp_models["linear"]),
            rp_model_2=copy.deepcopy(rp_models["Brooks-Corey_nb_4"]),
            cp_model_1=copy.deepcopy(cp_models["None"]),
            cp_model_2=copy.deepcopy(cp_models["linear"]),
            spe10_layer=spe10_layer,
        )
        run.run_simulation(config)

        # NOTE Reimport to make sure nothing persists.
        importlib.reload(run)

        config = SimulationConfig(
            file_name="run_2",
            folder_name=results_dir / "mwe_status_bug" / "run_2",
            solver_name="AHC",
            adaptive_error_ratio=1e-4,
            init_s=init_s,
            rp_model_1=copy.deepcopy(rp_models["linear"]),
            rp_model_2=copy.deepcopy(rp_models["Brooks-Corey_nb_4"]),
            cp_model_1=copy.deepcopy(cp_models["None"]),
            cp_model_2=copy.deepcopy(cp_models["linear"]),
            spe10_layer=spe10_layer,
        )
        run.run_simulation(config)


# endregion
