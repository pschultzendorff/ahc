r"""Study convergence of solvers with different rel. perm./cap. pressure models and
 different initial saturations.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton
- Newton
- Newton with Appleyard chopping


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

import itertools
import logging
import os
import pathlib
import shutil
import warnings
from dataclasses import dataclass
from typing import Any, Type

import numpy as np
import porepy as pp
from tpf.derived_models.spe10 import INITIAL_PRESSURE, SPE10Mixin
from tpf.models.adaptive_newton import TwoPhaseFlowANewton
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.models.protocol import TPFProtocol
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.utils.constants_and_typing import FEET
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
logger.setLevel(logging.INFO)

dirname: pathlib.Path = pathlib.Path(__file__).parent

# endregion


# region MODEL
class InitialConditionsMixin(TPFProtocol):
    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        initial_pressure = np.full(self.g.num_cells, INITIAL_PRESSURE)
        initial_saturation = np.full(
            self.g.num_cells, self.params["spe10_initial_saturation"]
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


class SPE10HC(
    InitialConditionsMixin,
    SPE10Mixin,
    TwoPhaseFlowAHC,
):  # type: ignore
    ...


class SPE10Newton(
    InitialConditionsMixin,
    SPE10Mixin,
    TwoPhaseFlowANewton,
):  # type: ignore
    ...


# endregion

# region UTILS
spe10_layer: int = 80

default_params: dict[str, Any] = {
    "progressbars": True,
    # Model:
    "formulation": "fractional_flow",
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    "spe10_layer": spe10_layer,
    "spe10_isotropic_perm": True,
    # HC params:
    "nonlinear_solver_statistics": SolverStatisticsHC,
    "nonlinear_solver": HCSolver,
    "hc_max_iterations": 20,
    # HC decay parameters.
    "hc_constant_decay": False,
    "hc_lambda_decay": 0.9,
    "hc_decay_min_max": (0.1, 0.95),
    "nl_iter_optimal_range": (4, 7),
    "nl_iter_relax_factors": (0.7, 1.3),
    "hc_decay_recomp_max": 5,
    # Nonlinear params:
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e30,
    "max_iterations": 20,
}

time_manager_params: dict[str, Any] = {
    "schedule": np.array([0.0, 10.0 * pp.DAY]),
    "dt_init": 10.0 * pp.DAY,
    "constant_dt": False,
    "dt_min_max": (1e-3 * pp.DAY, 10.0 * pp.DAY),
    "iter_optimal_range": (9, 12),
    "iter_relax_factors": (0.7, 1.3),
    "recomp_factor": 0.1,
    "recomp_max": 5,
}


@dataclass(unsafe_hash=True)
class SimulationConfig:
    """Class to store all the simulation parameters that can vary."""

    folder_name: pathlib.Path
    file_name: str
    solver_name: str
    adaptive_error_ratio: float
    cell_size: float
    init_s: float
    rp_model_1: dict[str, Any]
    rp_model_2: dict[str, Any]
    cp_model: dict[str, Any]


def setup_solver(
    solver: str, adaptive_error_ratio: float
) -> tuple[Type[SPE10HC] | Type[SPE10Newton], dict[str, Any]]:
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

    if solver == "AHC":
        solver_params: dict[str, Any] = {
            # Homotopy Continuation (HC) parameters:
            "nonlinear_solver_statistics": SolverStatisticsHC,
            "nonlinear_solver": HCSolver,
            "hc_max_iterations": 20,
            "hc_constant_decay": False,
            "hc_lambda_decay": 0.9,
            "hc_decay_min_max": (0.1, 0.95),
            "nl_iter_optimal_range": (4, 7),
            "nl_iter_relax_factors": (0.7, 1.3),
            "hc_decay_recomp_max": 5,
            # Adaptivity:
            "hc_adaptive": True,
            "hc_error_ratio": adaptive_error_ratio,  # adaptive error for homotopy
            "nl_error_ratio": 0.1,
            "hc_nl_convergence_tol": 1e3,
            # Nonlinear solver parameters:
            "nl_convergence_tol": 1e-5,
            "nl_divergence_tol": 1e30,
            "max_iterations": 20,
            "nl_appleyard_chopping": False,
        }
        model_class = SPE10HC
    elif solver == "Newton":
        solver_params = {
            # Newton solver parameters:
            "nonlinear_solver_statistics": SolverStatisticsANewton,
            "nonlinear_solver": pp.NewtonSolver,
            # Adaptivity:
            "nl_adaptive": True,
            "nl_error_ratio": adaptive_error_ratio,
            "nl_adaptive_convergence_tol": 1e3,
            # Further parameters:
            "nl_convergence_tol": 1e-5,
            "nl_divergence_tol": 1e30,
            "nl_appleyard_chopping": False,
            "max_iterations": 20,
        }
        model_class = SPE10Newton
    elif solver == "NewtonAppleyard":
        solver_params = {
            # Newton solver params with Appleyard chopping:
            "nonlinear_solver_statistics": SolverStatisticsANewton,
            "nonlinear_solver": pp.NewtonSolver,
            "nl_adaptive": True,
            "nl_error_ratio": adaptive_error_ratio,
            "nl_adaptive_convergence_tol": 1e3,
            "nl_convergence_tol": 1e-5,
            "nl_divergence_tol": 1e30,
            "nl_appleyard_chopping": True,
            "max_iterations": 50,
        }
        model_class = SPE10Newton
    else:
        raise ValueError(f"Unknown solver: {solver}")
    return model_class, solver_params


def run_simulation(config: SimulationConfig) -> None:
    """Run simulation for a single configuration."""
    logger.info(
        f"solver: {config.solver_name}, "
        f"adaptive error ratio: {config.adaptive_error_ratio:.2f}, "
        f"cell size: {config.cell_size:.2f}, "
        f"initial saturation: {config.init_s}, "
        f"RP model 1: {config.rp_model_1}, "
        f"RP model 2: {config.rp_model_2}, "
        f"CP model: {config.cp_model}."
    )

    model_class, solver_params = setup_solver(
        config.solver_name, config.adaptive_error_ratio
    )

    # Build params dictionary
    params = default_params.copy()
    params.update(solver_params)

    # Newton and Appleyard Newton require a single rel. perm. model.
    if config.solver_name.startswith("Newton"):
        rel_perm_constants = config.rp_model_2
    else:
        rel_perm_constants = {
            "model_1": config.rp_model_1,
            "model_2": config.rp_model_2,
        }
    params.update(
        {
            "meshing_arguments": {"cell_size": config.cell_size},
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": config.cp_model,
            "spe10_initial_saturation": config.init_s,
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
    except Exception:
        pass
        config.folder_name.mkdir(parents=True)

    try:
        model = model_class(params)
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as e:
        with (config.folder_name / "failure.txt").open("w") as f:
            f.write(str(e))
        logger.error(f"Run failed with error: {e}.")


# endregion

# region RUN
solvers: list[str] = ["AHC", "Newton", "NewtonAppleyard"]
adaptive_error_ratios: list[float] = [0.1, 0.01]

rp_models: dict[str, Any] = {
    "Brooks-Corey": {
        "model": "Brooks-Corey",
        "limit": True,
        "n1": 2,
        "n2": 2,
        "n3": 1,
    },
    "Corey_power_2": {"model": "Corey", "limit": True, "power": 2},
    "Corey_power_3": {"model": "Corey", "limit": True, "power": 3},
}
cp_models: dict[str, Any] = {
    "None": {"model": None},
    "Brooks-Corey": {"model": "Brooks-Corey", "n1": 2, "n2": 2, "n3": 1},
}


def generate_configs() -> list[SimulationConfig]:
    """Generate all simulation configurations."""
    configs = []
    for solver_name, adaptive_error_ratio in itertools.product(
        solvers, adaptive_error_ratios
    ):
        # Newton at an adaptive error ratio of 0.01 causes problems and is skipped.
        if solver_name == "Newton" and adaptive_error_ratio == 0.01:
            continue

        # Varying rel. perm. models at init_s = 0.2 and init_s = 0.3
        for init_s in [0.2, 0.3]:
            for rp_model_name, rp_model in rp_models.items():
                folder_name = (
                    dirname
                    / f"{solver_name}_{adaptive_error_ratio:.2f}"
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
                        cell_size=600 * FEET / 30,
                        init_s=init_s,
                        rp_model_1={"model": "linear", "limit": True},
                        rp_model_2=rp_model,
                        cp_model={"model": None},
                    )
                )

        # Varying init_s for the Brooks-Corey model.
        for init_s in list(np.linspace(0.2, 0.3, 5)[1:-1]):
            file_name = f"init_s_{init_s:.2f}"
            folder_name = (
                dirname
                / f"{solver_name}_{adaptive_error_ratio:.2f}"
                / "varying_init_s"
                / file_name
            )
            configs.append(
                SimulationConfig(
                    file_name=file_name,
                    folder_name=folder_name,
                    solver_name=solver_name,
                    adaptive_error_ratio=adaptive_error_ratio,
                    cell_size=600 * FEET / 30,
                    init_s=init_s,
                    rp_model_1={"model": "linear", "limit": False},
                    rp_model_2={
                        "model": "Brooks-Corey",
                        "limit": False,
                        "n1": 2,
                        "n2": 2,
                        "n3": 1,
                    },
                    cp_model={"model": None},
                )
            )
    return configs


if __name__ == "__main__":
    configs = generate_configs()
    for config in configs:
        run_simulation(config)

# endregion
