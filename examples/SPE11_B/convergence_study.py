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
from run import (
    LINEAR_RP_MODEL,
    SPE11_ENTRY_PRESSURE,
    SPE11_REFINEMENT_FACTOR,
    ZERO_BUOYANCY_MODEL,
    ZERO_CP_MODEL,
    cp_models,
    rp_models,
    run_simulation,
)

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import (
    SimulationConfig,
    clean_up_after_simulation,
    plot_convergence,
    plot_estimators,
    read_solver_stats,
)

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

# Setup logging.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Directories for results.
dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()
results_dir = dirname / "results_convergence_study"

# endregion


# region RUN


def generate_temporal_convergence_cases() -> list[SimulationConfig]:
    global time_step_sizes
    time_step_sizes = [1000.0, 500.0, 250.0, 100.0, 50.0, 25.0, 12.5]
    time_step_sizes = [ts * pp.DAY for ts in time_step_sizes]

    cases = []
    for time_step_size in time_step_sizes:
        cases.append(
            SimulationConfig(
                results_dir=results_dir,
                regime="viscous",
                study="temporal_estimator_convergence",
                case=f"time_step_size_{time_step_size:.1f}",
                solver_name="AHC",
                hc_tol=1e-5,  # Fixed for temporal study.
                nl_tol=0.1,
                init_s=0.8,
                rp_model_1=LINEAR_RP_MODEL,
                rp_model_2=rp_models["Brooks-Corey_nb_4"],
                cp_model_1=ZERO_CP_MODEL,
                cp_model_2=cp_models["Brooks-Corey_nb_4"],
                buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                spe11_refinement_factor=SPE11_REFINEMENT_FACTOR,
                spe11_entry_pressure=SPE11_ENTRY_PRESSURE,
            )
        )
    return cases


def generate_spatial_convergence_cases() -> list[SimulationConfig]:
    cases = []
    for refinement_factor in [5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75]:
        cases.append(
            SimulationConfig(
                results_dir=results_dir,
                regime="viscous",
                study="spatial_estimator_convergence",
                case=f"refinement_factor_{refinement_factor:.1f}",
                solver_name="AHC",
                hc_tol=1e-5,  # Fixed for spatial study.
                nl_tol=0.1,
                init_s=0.8,
                rp_model_1=LINEAR_RP_MODEL,
                rp_model_2=rp_models["Brooks-Corey_nb_4"],
                cp_model_1=ZERO_CP_MODEL,
                cp_model_2=cp_models["Brooks-Corey_nb_4"],
                buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                spe11_refinement_factor=refinement_factor,
                spe11_entry_pressure=SPE11_ENTRY_PRESSURE,
            )
        )
    return cases


def generate_hc_convergence_cases() -> list[SimulationConfig]:
    cases = []
    cases.append(
        SimulationConfig(
            results_dir=results_dir,
            regime="viscous",
            study="hc_estimator_convergence",
            case="SPE11_B",
            solver_name="AHC",
            hc_tol=0.01,
            nl_tol=0.1,
            init_s=0.8,
            rp_model_1=LINEAR_RP_MODEL,
            rp_model_2=rp_models["Brooks-Corey_nb_4"],
            cp_model_1=ZERO_CP_MODEL,
            cp_model_2=cp_models["Brooks-Corey_nb_4"],
            buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
            buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
            spe11_refinement_factor=SPE11_REFINEMENT_FACTOR,
            spe11_entry_pressure=SPE11_ENTRY_PRESSURE,
        )
    )
    return cases


def generate_hc_divergence_cases() -> list[SimulationConfig]:
    r"""Run with high entry pressure :math:`10000\,\mathrm{Pa}` to ensure divergence."""
    cases = []
    cases.append(
        SimulationConfig(
            results_dir=results_dir,
            regime="viscous",
            study="hc_estimator_divergence",
            case="SPE11_B",
            solver_name="AHC",
            hc_tol=0.01,
            nl_tol=0.1,
            init_s=0.9,
            rp_model_1=LINEAR_RP_MODEL,
            rp_model_2=rp_models["Brooks-Corey_nb_4"],
            cp_model_1=ZERO_CP_MODEL,
            cp_model_2=cp_models["Brooks-Corey_nb_4"],
            buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
            buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
            spe11_refinement_factor=SPE11_REFINEMENT_FACTOR,
            spe11_entry_pressure=10000 * pp.PASCAL,  # Very high to ensure divergence.
        )
    )

    return cases


studies: dict[str, list[SimulationConfig]] = {
    # "temporal_estimator_convergence": generate_temporal_convergence_cases(),
    "spatial_estimator_convergence": generate_spatial_convergence_cases(),
    # "hc_estimator_convergence": generate_hc_convergence_cases(),
    "hc_estimator_divergence": generate_hc_divergence_cases(),
}

if __name__ == "__main__":
    fig_dir = dirname / "figures"

    results_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    for study_name, study in studies.items():
        data = []
        # Run the simulations for each study.
        for config in study:
            # Different tim step sizes are required for the different convergence
            # studies.
            if study_name == "temporal_estimator_convergence":
                expected_final_time = float(str(config.case).split("_")[-1])
            elif study_name == "spatial_estimator_convergence":
                # For the spatial convergence study, we use a small fixed time step size
                # to reduce the temporal error.
                expected_final_time = 30.0 * pp.DAY
            else:
                # For the HC convergence and divergence studies, we use a fixed large
                # time step size to make the nonlinear problem challenging to solve.
                expected_final_time = 3000.0 * pp.DAY

            time_manager_params = {
                "schedule": np.array([0.0, expected_final_time]),
                "dt_init": expected_final_time,
                "constant_dt": True,
            }

            run_simulation(
                config,
                time_manager_params=time_manager_params,
                # Important to disable extrapolate_temp_estimator_after_cutting by
                # setting the exponent to 0.0 for the convergence study, as we want to
                # see the effect of time step cutting on the estimators.
                additional_params={"extrapolate_temp_estimator_after_cutting": 0.0},
            )
            data.append(read_solver_stats(config, expected_final_time))
            clean_up_after_simulation(config)

        # Make convergence plots for each study.
        if study_name == "temporal_estimator_convergence":
            fig = plot_convergence(data, time_step_sizes, "time_step_size")
        elif study_name == "spatial_estimator_convergence":
            fig = plot_convergence(
                data, [stats.num_grid_cells for stats in data], "num_grid_cells"
            )
        elif study_name == "hc_estimator_convergence":
            fig = plot_estimators(data[0], combine_disc_est=True)
        elif study_name == "hc_estimator_divergence":
            fig = plot_estimators(
                data[0], combine_disc_est=True, legend_loc="upper right"
            )
        else:
            raise ValueError(f"Unknown study: {study_name}")
        fig.savefig(fig_dir / f"{study_name}.png", dpi=300)


# endregion
