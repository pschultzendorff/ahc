r"""Statistical analysis of nonlinear convergence on all layers of the SPE10, case 2A.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton
- Homotopy continuation (HC) with Newton
- Adaptive Newton
- Adaptive Newton with Appleyard chopping


We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
continuation. The considered model is similar to the heterogeneous 3D models in the
article (section 4.6.4), but on a 2D domain for now.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 1200x2200 ft domain, single layers of SPE10, case 2A.
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
    - Saturation: 0.3
- Rel. perm. models:
    - Brooks-Corey
- Capillary pressure model:
    - Brooks-Corey, entry pressure 50 Pa.

"""

import os
import pathlib
import sys
import warnings

import numpy as np
import porepy as pp
from run import run_simulation

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig, clean_up_after_simulation

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
solvers = [
    "AHC",
    "Newton",
    "NewtonAppleyard",
    "HC",
]
adaptive_error_ratios = [0.001, 0.1, 0.1, 0.1]

rp_model = {
    "model": "Brooks-Corey-Mualem",
    "limit": True,
    "n_b": 4.0,
    "eta": 2.0,
}  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1


cp_model = {
    "model": "Brooks-Corey",
    "n_b": 4.0,
    "entry_pressure": 50 * pp.PASCAL,
    "limit": True,
    "max": 1e6 * pp.PASCAL,
}


def generate_configs() -> list[SimulationConfig]:
    """Generate all simulation configurations."""
    configs = []

    # Varying rel. perm. models at init_s = 0.2 and init_s = 0.3.
    for spe10_layer in range(85):
        for solver_name, adaptive_error_ratio in zip(solvers, adaptive_error_ratios):
            folder_name = (
                dirname
                / f"{solver_name}_{adaptive_error_ratio:.3f}"
                / f"layer_{spe10_layer:02d}"
            )
            configs.append(
                SimulationConfig(
                    file_name=f"{solver_name}_{adaptive_error_ratio:.3f}",
                    folder_name=folder_name,
                    solver_name=solver_name,
                    adaptive_error_ratio=adaptive_error_ratio,
                    init_s=0.3,
                    rp_model_1={"model": "linear", "limit": True},
                    rp_model_2=rp_model,
                    cp_model_1={"model": None},
                    cp_model_2=cp_model,
                    spe10_layer=spe10_layer,
                )
            )

    return configs


if __name__ == "__main__":
    configs = generate_configs()
    for config in configs:
        run_simulation(config)
        # Keep only 2 simulations from the upper and lower layers each.
        if config.spe10_layer not in [10, 20, 50, 65]:
            clean_up_after_simulation(config)

# endregion
