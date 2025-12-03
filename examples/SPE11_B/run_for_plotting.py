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
from typing import Any

import numpy as np
from run import run_simulation

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


if __name__ == "__main__":
    rp_model: dict[str, Any] = {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 1.0,
        "eta": 2.0,
    }  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1

    cp_model: dict[str, Any] = {
        "model": "Brooks-Corey",
        "n_b": 2.0,
    }
    config = SimulationConfig(
        file_name="plotting",
        folder_name=dirname / "plotting",
        solver_name="NewtonAppleyard",
        adaptive_error_ratio=0.0,  # Disregarded
        refinement_factor=1.0,
        init_s=0.8,
        rp_model_1=rp_model,
        rp_model_2=rp_model,
        cp_model_1=cp_model,
        cp_model_2=cp_model,
    )
    run_simulation(config)
# endregion
