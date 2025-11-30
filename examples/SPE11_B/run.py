r"""Study convergence of solvers on different grid sizes and with different rel.
 perm./cap. pressure models.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton
- Newton
- Newton with Appleyard chopping



Model description:
- Constant CO2 injection in the center.
- No flow boundary condition on the sides and bottom. Homogeneous Dirichlet on top.
- Simulation time: 10 days
- Solid properties:
    - Porosity: SPE11, case A.
    - Permeability: SPE11, case A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.1.
    - CO2:  Residual
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
import shutil
import warnings
from dataclasses import dataclass
from typing import Any, Type

import numpy as np
import porepy as pp
from tpf.derived_models.spe11 import SPE11Mixin, case_B
from tpf.models.adaptive_newton import TwoPhaseFlowANewton
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.models.phase import FluidPhase
from tpf.models.protocol import TPFProtocol
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.viz.solver_statistics import SolverStatisticsANewton, SolverStatisticsHC

# region SETUP

# Disable numba JIT for debugging.
# config.DISABLE_JIT = False

# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow. The latter can appear during estimator
# calculation.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

# endregion


# region MODEL
class InitialConditionsMixin(TPFProtocol):
    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        initial_pressure = np.full(self.g.num_cells, case_B["INITIAL_PRESSURE"])
        initial_saturation = np.full(
            self.g.num_cells, self.params["spe11_initial_saturation"]
        )
        self.equation_system.set_variable_values(
            np.concatenate([initial_pressure, initial_pressure]),
            [self.wetting.p, self.nonwetting.p],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )
        self.equation_system.set_variable_values(
            np.concatenate([initial_saturation, 1 - initial_saturation]),
            [self.wetting.s, self.nonwetting.s],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )


class SPE11HC(
    InitialConditionsMixin,
    SPE11Mixin,
    TwoPhaseFlowAHC,
):  # type: ignore
    ...


class SPE11Newton(
    InitialConditionsMixin,
    SPE11Mixin,
    TwoPhaseFlowANewton,
):  # type: ignore
    ...


# endregion

# region UTILS

default_params = {
    "progressbars": True,
    # Model:
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    # SPE11 parameters:
    "spe11_case": "B",
    "spe11_heterogeneous_cap_pressure": False,
    "spe11_entry_pressure": 100.0,  # [Pa]
    # Nonlinear solver:
    "nl_enforce_physical_saturation": True,
}


init_time_step_params = {
    "schedule": np.array([0.0, 1.0 * pp.DAY]),
    "dt_init": 1.0 * pp.DAY,
    "constant_dt": False,
    "dt_min_max": (1e-1 * pp.DAY, 1.0 * pp.DAY),
    "iter_optimal_range": (9, 12),
    "iter_relax_factors": (0.7, 1.3),
    "recomp_factor": 0.1,
    "recomp_max": 8,
}
default_time_manager_params = {
    "schedule": np.array([0.0, 3000.0 * pp.DAY]),
    "dt_init": 3000.0 * pp.DAY,
    "constant_dt": False,
    "dt_min_max": (1 * pp.DAY, 3000.0 * pp.DAY),
    "iter_optimal_range": (9, 12),
    "iter_relax_factors": (0.7, 1.3),
    "recomp_factor": 0.1,
    "recomp_max": 10,
}


@dataclass(unsafe_hash=True)
class SimulationConfig:
    """Class to store all the simulation parameters that can vary."""

    folder_name: pathlib.Path
    file_name: str
    solver_name: str
    adaptive_error_ratio: float
    refinement_factor: float
    init_s: float
    rp_model_1: dict[str, Any]
    rp_model_2: dict[str, Any]
    cp_model_1: dict[str, Any]
    cp_model_2: dict[str, Any]


def setup_solver(
    solver: str, adaptive_error_ratio: float
) -> tuple[Type[SPE11HC] | Type[SPE11Newton], dict[str, Any], dict[str, Any]]:
    """Return a tuple of solver-specific parameters and model class based on the solver
    name.

    Parameters:
        solver: The name of the solver ("AHC", "Newton", or "NewtonAppleyard").
        adaptive_error_ratio: The error ratio used for adaptive parameter settings.

    Returns:
        A tuple ``(model_class, solver_params)``, where ``model_class`` is the model
        class with the correct adaptive solver and ``solver_params`` is a dictionary
        containing solver parameters.

    """
    logger.info(f"solver: {solver}, adaptive error ratio: {adaptive_error_ratio:.2f}.")
    if solver == "HC":
        solver_params = {
            # Homotopy Continuation (HC) parameters:
            "nonlinear_solver_statistics": SolverStatisticsHC,
            "nonlinear_solver": HCSolver,
            "hc_constant_decay": False,
            "hc_lambda_decay": 0.9,
            "hc_decay_min_max": (0.1, 0.95),
            "nl_iter_optimal_range": (7, 10),
            "nl_iter_relax_factors": (0.7, 1.3),
            "hc_decay_recomp_max": 5,
            # Non-adaptive stopping criteria:
            "hc_adaptive": False,
            "hc_max_iterations": 100,
            "hc_lambda_min": 0.01,
            # Nonlinear solver parameters:
            "nl_convergence_tol": 5e-5,
            "nl_divergence_tol": 1e30,
            "max_iterations": 20,
            "nl_appleyard_chopping": False,
        }
        # Update adaptive time stepping parameters for HC.
        time_manager_params = {
            "iter_optimal_range": (30, 80),
            "iter_relax_factors": (0.7, 1.3),
            "iter_max": 100,  # This has to be the same as "hc_max_iterations", but
            # the TimeManager does not know about that.
        }
        model_class = SPE11HC

    elif solver == "AHC":
        solver_params: dict[str, Any] = {
            # Homotopy Continuation (HC) parameters:
            "nonlinear_solver_statistics": SolverStatisticsHC,
            "nonlinear_solver": HCSolver,
            "hc_max_iterations": 30,
            "hc_constant_decay": False,
            "hc_lambda_decay": 0.9,
            "hc_decay_min_max": (0.1, 0.99),
            "nl_iter_optimal_range": (4, 7),
            "nl_iter_relax_factors": (0.7, 1.3),
            "hc_decay_recomp_max": 5,
            # Adaptivity:
            "hc_adaptive": True,
            "hc_error_ratio": adaptive_error_ratio,  # adaptive error for homotopy
            "nl_error_ratio": 0.1,
            "hc_nl_convergence_tol": 1e-4,
            # Nonlinear solver parameters:
            "nl_convergence_tol": 1e-5,
            "nl_divergence_tol": 1e30,
            "max_iterations": 20,
            "nl_appleyard_chopping": False,
        }
        time_manager_params = {}

        model_class = SPE11HC

    elif solver == "Newton":
        solver_params = {
            # Newton solver parameters:
            "nonlinear_solver_statistics": SolverStatisticsANewton,
            "nonlinear_solver": pp.NewtonSolver,
            # Adaptivity:
            "nl_adaptive": True,
            "nl_error_ratio": adaptive_error_ratio,
            "nl_adaptive_convergence_tol": 1e-4,
            # Further parameters:
            "nl_convergence_tol": 1e-5,
            "nl_divergence_tol": 1e30,
            "nl_appleyard_chopping": False,
            "max_iterations": 20,
        }
        time_manager_params = {}
        model_class = SPE11Newton
    elif solver == "NewtonAppleyard":
        solver_params = {
            # Newton solver params with Appleyard chopping:
            "nonlinear_solver_statistics": SolverStatisticsANewton,
            "nonlinear_solver": pp.NewtonSolver,
            "nl_adaptive": True,
            "nl_error_ratio": adaptive_error_ratio,
            "nl_adaptive_convergence_tol": 1e-4,
            "nl_convergence_tol": 1e-5,
            "nl_divergence_tol": 1e30,
            "nl_appleyard_chopping": True,
            "max_iterations": 50,
        }
        time_manager_params = {}
        model_class = SPE11Newton
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return model_class, solver_params, time_manager_params


def run_simulation(config: SimulationConfig) -> None:
    """Run simulation for a single configuration."""
    logger.info(
        f"solver: {config.solver_name}, "
        f"adaptive error ratio: {config.adaptive_error_ratio:.2f}, "
        f"refinement factor: {config.refinement_factor:.2f}, "
        f"initial saturation: {config.init_s}, "
        f"RP model 1: {config.rp_model_1}, "
        f"RP model 2: {config.rp_model_2}, "
        f"CP model 1: {config.cp_model_1}, "
        f"CP model 2: {config.cp_model_2}."
    )

    model_class, solver_params, time_manager_params = setup_solver(
        config.solver_name, config.adaptive_error_ratio
    )

    # Build params dictionaries.
    params = default_params | solver_params
    time_manager_params = default_time_manager_params | time_manager_params

    # Newton and Appleyard Newton require only one of each constitutive law.
    if config.solver_name.startswith("Newton"):
        rel_perm_constants = config.rp_model_2
        cap_press_constants = config.cp_model_2
    else:
        rel_perm_constants = {
            "model_1": config.rp_model_1,
            "model_2": config.rp_model_2,
        }
        cap_press_constants = {
            "model_1": config.cp_model_1,
            "model_2": config.cp_model_2,
        }

    params.update(
        {
            "meshing_arguments": {"spe11_refinement_factor": config.refinement_factor},
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": cap_press_constants,
            "spe11_initial_saturation": config.init_s,
        }
    )

    params.update(
        {
            "folder_name": config.folder_name,
            "file_name": config.file_name,
            "solver_statistics_file_name": config.folder_name
            / "solver_statistics.json",
            "time_manager": pp.TimeManager(**time_manager_params),
        }
    )

    try:
        shutil.rmtree(config.folder_name)
        config.folder_name.mkdir(parents=True)
    except Exception:
        pass

    try:
        model = model_class(params)
        # # "Heat up" the model to not have temporal discretization error super large only
        # # due to starting from zero.
        # pp.run_time_dependent_model(model=model, params=params)

        # # Actually run the simulation.
        params.update({"time_manager": pp.TimeManager(**time_manager_params)})
        pp.run_time_dependent_model(model=model, params=params)

    except Exception as e:
        config.folder_name.mkdir(parents=True, exist_ok=True)
        with (config.folder_name / "failure.txt").open("w") as f:
            f.write(str(e))
        logger.error(f"Run failed with error: {e}.")


# endregion

# region RUN
solvers_and_ratios: list[tuple[str, float]] = [
    ("AHC", 0.1),
    ("HC", 0.1),
    ("Newton", 0.1),
    ("NewtonAppleyard", 0.1),
]
refinement_factors: list[float] = [20, 5, 0.5]  # , 0.5]

rp_models: dict[str, Any] = {
    "linear": {"model": "linear", "limit": True},
    "Brooks-Corey": {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 1.0,
        "eta": 2.0,
    },  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1
    "Corey_power_2": {"model": "Corey", "limit": True, "power": 2},
    "Corey_power_3": {"model": "Corey", "limit": True, "power": 3},
}

cp_models: dict[str, Any] = {
    "None": {"model": None},
    "linear": {"model": "linear", "linear_param": 3.0},
    "Brooks-Corey": {"model": "Brooks-Corey", "n_b": 2.0},
}


def generate_configs() -> list[SimulationConfig]:
    """Generate all simulation configurations."""
    configs = []
    # Varying rel. perm. models at init_s = 0.8 and init_s = 0.9.
    for init_s in [0.8, 0.9]:
        for rp_model_name, rp_model in rp_models.items():
            if rp_model_name == "linear":
                continue
            for solver_name, adaptive_error_ratio in solvers_and_ratios:
                folder_name = (
                    dirname
                    / f"{solver_name}_{adaptive_error_ratio:.3f}"
                    / "varying_rp"
                    / f"init_s_{init_s}"
                    / rp_model_name
                )
                configs.append(
                    SimulationConfig(
                        file_name=rp_model_name,
                        folder_name=folder_name,
                        solver_name=solver_name,
                        adaptive_error_ratio=adaptive_error_ratio,
                        refinement_factor=refinement_factors[0],
                        init_s=init_s,
                        rp_model_1=rp_models["linear"],
                        rp_model_2=rp_model,
                        cp_model_1=cp_models["None"],
                        cp_model_2=cp_models["Brooks-Corey"],
                    )
                )

    # Varying refinement factors at init_s = 0.8 and init_s = 0.9.
    for init_s in [0.8, 0.9]:
        for refinement_factor in refinement_factors:
            # Highest resolution at init_s = 0.9 takes too long, mostly due to the fact
            # that adaptive Newton only makes sense for small-sized updates to produce
            # physical solutions. But setting ``hc_nl_convergence_tol`` low makes AHC
            # require a lot of time steps.
            for solver_name, adaptive_error_ratio in solvers_and_ratios:
                file_name = f"ref_fac_{refinement_factor:.2f}"
                folder_name = (
                    dirname
                    / f"{solver_name}_{adaptive_error_ratio:.3f}"
                    / "varying_refinement"
                    / f"init_s_{init_s}"
                    / file_name
                )
                configs.append(
                    SimulationConfig(
                        file_name=file_name,
                        folder_name=folder_name,
                        solver_name=solver_name,
                        adaptive_error_ratio=adaptive_error_ratio,
                        refinement_factor=refinement_factor,
                        init_s=init_s,
                        rp_model_1=rp_models["linear"],
                        rp_model_2=rp_models["Brooks-Corey"],
                        cp_model_1=cp_models["None"],
                        cp_model_2=cp_models["Brooks-Corey"],
                    )
                )

    return configs


if __name__ == "__main__":
    configs = generate_configs()
    for config in configs:
        run_simulation(config)

# endregion
