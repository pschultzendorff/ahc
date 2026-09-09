"""Run layers 36-52 of SPE10 model (complex lower layers)."""

import os
import pathlib
import sys
import warnings

import numpy as np
from run import (
    CELL_SIZE,
    LINEAR_RP_MODEL,
    SPE10_CASE,
    WATER_DENSITY,
    ZERO_BUOYANCY_MODEL,
    ZERO_CP_MODEL,
    cp_models,
    results_dir,
    rp_models,
    run_simulation,
    solvers_and_tols,
)

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig, clean_up_after_simulation

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


# endregion


# region RUN


def generate_cases() -> list[SimulationConfig]:
    """Generate simulation configurations for the selected layers and solvers."""
    cases = []

    for spe10_layer in range(35, 52):
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
                    spe10_cell_size=CELL_SIZE,
                    spe10_layer=spe10_layer,
                    spe10_case=SPE10_CASE,
                    spe10_water_density=WATER_DENSITY,
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
