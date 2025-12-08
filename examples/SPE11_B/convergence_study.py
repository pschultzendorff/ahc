r"""Study convergence of spatial and temporal error estimators on SPE11, case B.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton

"""

import logging
import os
import pathlib
import sys
import warnings

import numpy as np
from run import cp_models, rp_models, run_simulation

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import (
    SimulationConfig,
    clean_up_after_simulation,
    plot_convergence,
    read_data,
)

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
refinement_factors: list[float] = [20, 15, 10, 5, 1, 0.5]
time_step_sizes: list[float] = [1000, 500, 250, 100, 50, 25, 12.5]


def generate_configs() -> list[SimulationConfig]:
    """Generate all simulation configurations."""
    configs = []

    for time_step_size in time_step_sizes:
        folder_name = (
            dirname / "temporal_estimator_convergence" / f"dt_{time_step_size:.1f}"
        )
        configs.append(
            SimulationConfig(
                file_name="temporal_estimator_convergence",
                folder_name=folder_name,
                solver_name="AHC",
                adaptive_error_ratio=1e-10,  # Fixed for temporal study.
                refinement_factor=refinement_factors[3],
                init_s=0.8,
                rp_model_1=rp_models["linear"],
                rp_model_2=rp_models["Brooks-Corey_nb_4"],
                cp_model_1=cp_models["None"],
                cp_model_2=cp_models["Brooks-Corey_nb_4"],
            )
        )

    for refinement_factor in refinement_factors:
        folder_name = (
            dirname / "spatial_estimator_convergence" / f"r_{refinement_factor:.1f}"
        )
        configs.append(
            SimulationConfig(
                file_name="spatial_estimator_convergence",
                folder_name=folder_name,
                solver_name="AHC",
                adaptive_error_ratio=1e-5,  # Fixed for spatial study.
                refinement_factor=refinement_factor,
                init_s=0.8,
                rp_model_1=rp_models["linear"],
                rp_model_2=rp_models["Brooks-Corey_nb_4"],
                cp_model_1=cp_models["None"],
                cp_model_2=cp_models["Brooks-Corey_nb_4"],
            )
        )

    return configs


if __name__ == "__main__":
    configs = generate_configs()

    for config in configs:
        continue
        if "temporal" in config.folder_name.parent.name:
            time_step_size = float(config.folder_name.name.split("_")[1])
        else:
            time_step_size = 30.0  # Default for spatial study.
        time_manager_params = {
            "schedule": np.array([0.0, time_step_size]),
            "dt_init": time_step_size,
            "constant_dt": True,
        }
        run_simulation(config, time_manager_params=time_manager_params)
        clean_up_after_simulation(config)

    data = []
    for config in configs[:7]:
        data.append(read_data(config))
    fig_temporal = plot_convergence(data, time_step_sizes, "time_step_size")
    fig_temporal.savefig(dirname / "temporal_estimator_convergence.png", dpi=300)

    data = []
    for config in configs[7:]:
        data.append(read_data(config))
    fig_spatial = plot_convergence(data, refinement_factors, "refinement_factor")
    fig_spatial.savefig(dirname / "spatial_estimator_convergence.png", dpi=300)

# endregion
