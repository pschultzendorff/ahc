r"""Run the simulation with small uniform time steps for fancy plotting.


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

import os
import pathlib
import warnings
from typing import Any

import numpy as np
import porepy as pp
from run import run_simulation

from ..utils import SimulationConfig, plot_estimators, read_data

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

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()


# endregion


# region RUN
spe10_layer: int = 55

if __name__ == "__main__":
    rp_model_1 = {"model": "linear", "limit": False}
    cp_model_1 = {
        "model": None,
    }
    rp_model_2 = {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 1.0,
        "eta": 2.0,
    }  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1

    cp_model_2: dict[str, Any] = {
        "model": "Brooks-Corey",
        "n_b": 2.0,
        "entry_pressure": 30 * pp.PASCAL,
    }
    config = SimulationConfig(
        file_name="ahc_for_plots",
        folder_name=dirname / "ahc_for_plots",
        solver_name="AHC",
        adaptive_error_ratio=0.00000001,
        init_s=0.3,
        rp_model_1=rp_model_1,
        rp_model_2=rp_model_2,
        cp_model_1=cp_model_1,
        cp_model_2=cp_model_2,
        spe10_layer=spe10_layer,
    )

    run_simulation(config)

    statistics = read_data(config)
    fig = plot_estimators(statistics, combine_disc_est=True)
    fig.savefig(dirname / "convergence_estimators.png")

# endregion
