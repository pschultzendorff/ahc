r"""Run the simulation with small uniform time steps for fancy plotting.

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
- Simulation time: 10 days
- Solid properties:
    - Porosity: Layers 10 and 80 of SPE10, case 2A.
    - Permeability: Layers 10 and 80 of SPE10, case 2A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Saturation: 0.3.
- Rel. perm. model:
    - Brooks-Corey-Mualem
- Capillary pressure model:
    - Brooks-Corey

"""

import logging
import os
import pathlib
import sys
import warnings

import numpy as np
import porepy as pp
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
rp_model = {"model": "Brooks-Corey-Mualem", "n_b": 4.0, "eta": 2.0, "limit": True}
cp_model = {
    "model": "Brooks-Corey",
    "n_b": 4.0,
    "entry_pressure": 50 * pp.PASCAL,
    "limit": True,
    "max": 1e6 * pp.PASCAL,
}

if __name__ == "__main__":
    for spe10_layer in [10, 55]:
        config = SimulationConfig(
            file_name=f"plotting_layer_{spe10_layer}",
            folder_name=dirname / f"plotting_layer_{spe10_layer}",
            solver_name="Newton",
            adaptive_error_ratio=0.1,
            init_s=0.3,
            rp_model_1=rp_model,
            rp_model_2=rp_model,
            cp_model_1=cp_model,
            cp_model_2=cp_model,
            spe10_layer=spe10_layer,
        )
        run_simulation(config)

# endregion
