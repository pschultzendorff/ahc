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
- Simulation time: 30 days
- Solid properties:
    - Porosity: All 85 layers of SPE10 case 2A (one layer per simulation).
    - Permeability: All 85 layers of SPE10 case 2A (one layer per simulation).
- Fluid properties:
    - Water: ``pp.fluid_values.water``. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Saturation: 0.3
- Rel. perm. models (model_2; model_1 is always linear for HC/AHC):
    - Brooks-Corey-Mualem with n_b=4, eta=2.
- Capillary pressure model:
    - Linear, entry pressure 50 Pa.

"""

import os
import pathlib
import sys
import warnings

import numpy as np
from run import (
    LINEAR_RP_MODEL,
    ZERO_BUOYANCY_MODEL,
    ZERO_CP_MODEL,
    cp_models,
    rp_models,
    run_simulation,
)

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
results_dir = dirname / "results"

# endregion


# region RUN
solvers_and_tols: list[tuple[str, float, float]] = [
    ("AHC", 0.1, 0.1),
    ("AHC", 0.01, 0.1),
    ("HC", 0.01, 1e-3),
    ("HC", 0.01, 1e-5),
    ("Newton", 0.0, 0.1),
    ("NewtonAppleyard", 0.0, 0.1),
]


def generate_cases() -> list[SimulationConfig]:
    """Generate simulation configurations for all layers and solvers."""
    cases = []

    for spe10_layer in range(85):
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous",
                    study="all_layers",
                    case=f"layer_{spe10_layer:02d}",
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=0.3,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_models["Brooks-Corey_nb_4"],
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_models["Brooks-Corey_nb_4"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                    spe10_layer=spe10_layer,
                )
            )

    return cases


if __name__ == "__main__":
    results_dir.mkdir(exist_ok=True)
    study = generate_cases()
    for config in study:
        run_simulation(config)
        # Keep full simulation results for 2 simulations from the upper and lower
        # layers, respectively.
        if config.spe10_layer not in [10, 20, 50, 65]:
            clean_up_after_simulation(config)

# endregion
