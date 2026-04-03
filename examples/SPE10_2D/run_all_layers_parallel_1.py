"""Run layers 36-52 of SPE10 model (complex lower layers)."""

import os
import pathlib
import sys
import warnings

import numpy as np
from .run import cp_models, rp_models, run_simulation, solvers_and_ratios

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from ..utils import SimulationConfig, clean_up_after_simulation

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


def generate_configs() -> list[SimulationConfig]:
    """Generate all simulation configurations."""
    results_dir = dirname / "results"
    results_dir.mkdir(exist_ok=True)

    configs = []

    for spe10_layer in range(35, 52):
        for solver_name, adaptive_error_ratio in solvers_and_ratios:
            folder_name = (
                results_dir
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
                    rp_model_1=rp_models["linear"],
                    rp_model_2=rp_models["Brooks-Corey_nb_4"],
                    cp_model_1=cp_models["None"],
                    cp_model_2=cp_models["linear"],
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
