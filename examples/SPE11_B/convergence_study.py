"""Study convergence of spatial, temporal, HC, and linearization error estimators on
SPE11, case B.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton

"""

import logging
import os
import pathlib
import sys
import warnings

import numpy as np
import porepy as pp
from run import cp_models, rp_models, run_simulation

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import (
    SimulationConfig,
    clean_up_after_simulation,
    plot_convergence,
    plot_estimators,
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
refinement_factors: list[float] = [10, 5, 1, 0.5]
time_step_sizes: list[float] = [1000, 500, 250, 100, 50, 25, 12.5]
time_step_sizes = [ts * pp.DAY for ts in time_step_sizes]


def generate_configs() -> list[SimulationConfig]:
    """Generate all simulation configurations."""
    results_dir = dirname / "results"
    results_dir.mkdir(exist_ok=True)

    configs = []

    for time_step_size in time_step_sizes:
        folder_name = (
            results_dir / "temporal_estimator_convergence" / f"dt_{time_step_size:.1f}"
        )
        configs.append(
            SimulationConfig(
                file_name="temporal_estimator_convergence",
                folder_name=folder_name,
                solver_name="AHC",
                adaptive_error_ratio=1e-5,  # Fixed for temporal study.
                refinement_factor=refinement_factors[2],
                init_s=0.8,
                rp_model_1=rp_models["linear"],
                rp_model_2=rp_models["Brooks-Corey_nb_4"],
                cp_model_1=cp_models["None"],
                cp_model_2=cp_models["Brooks-Corey_nb_4"],
            )
        )

    for refinement_factor in refinement_factors:
        folder_name = (
            results_dir / "spatial_estimator_convergence" / f"r_{refinement_factor:.1f}"
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

    configs.append(
        SimulationConfig(
            file_name="hc_estimator_convergence",
            folder_name=results_dir / "hc_estimator_convergence",
            solver_name="AHC",
            adaptive_error_ratio=0.01,
            refinement_factor=refinement_factors[2],
            init_s=0.8,
            rp_model_1=rp_models["linear"],
            rp_model_2=rp_models["Brooks-Corey_nb_4"],
            cp_model_1=cp_models["None"],
            cp_model_2=cp_models["Brooks-Corey_nb_4"],
        )
    )

    configs.append(
        SimulationConfig(
            file_name="hc_estimator_divergence",
            folder_name=results_dir / "hc_estimator_divergence",
            solver_name="AHC",
            adaptive_error_ratio=0.01,
            refinement_factor=refinement_factors[2],
            init_s=0.9,
            rp_model_1=rp_models["linear"],
            rp_model_2=rp_models["Brooks-Corey_nb_4"],
            cp_model_1=cp_models["None"],
            cp_model_2=cp_models["Brooks-Corey_nb_4"],
            spe11_entry_pressure=200 * pp.PASCAL,  # Larger to ensure divergence.
        )
    )

    return configs


if __name__ == "__main__":
    configs = generate_configs()

    for i, config in enumerate(configs):
        if "temporal" in config.folder_name.parent.name:
            time_step_size = float(config.folder_name.name.split("_")[1])
        elif "spatial" in config.folder_name.parent.name:
            time_step_size = 30.0 * pp.DAY  # Default for spatial study.
        else:
            time_step_size = 3000.0 * pp.DAY  # Default for HC study.
        time_manager_params = {
            "schedule": np.array([0.0, time_step_size]),
            "dt_init": time_step_size,
            "constant_dt": True,
        }
        run_simulation(
            config,
            time_manager_params=time_manager_params,
            extrapolate_temp_estimator_after_cutting=False,
        )
        clean_up_after_simulation(config)

    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    data = []
    for config in configs[:7]:
        time_step_size = float(config.folder_name.name.split("_")[1])
        data.append(read_data(config, time_step_size))
    fig_temporal = plot_convergence(data, time_step_sizes, "time_step_size")
    fig_temporal.savefig(fig_dir / "temporal_estimator_convergence.png", dpi=300)

    data = []
    for config in configs[7:11]:
        time_step_size = 30.0 * pp.DAY  # Default for spatial study.
        data.append(read_data(config, time_step_size))
    num_grid_cells = [stats.num_grid_cells for stats in data]
    fig_spatial = plot_convergence(data, num_grid_cells, "num_grid_cells")
    fig_spatial.savefig(fig_dir / "spatial_estimator_convergence.png", dpi=300)

    time_step_size = 3000.0 * pp.DAY  # Default for HC study.
    stats = read_data(configs[11], expected_final_time=time_step_size)
    fig = plot_estimators(stats, combine_disc_est=True)
    fig.savefig(fig_dir / "hc_estimator_convergence.png", dpi=300)

    stats = read_data(configs[12], expected_final_time=time_step_size)
    fig = plot_estimators(stats, combine_disc_est=True, legend_loc="upper right")
    fig.savefig(fig_dir / "hc_estimator_divergence.png", dpi=300)


# endregion

# endregion
