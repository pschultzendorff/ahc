r"""Run the simulation once with linear and once with nonlinear constitutive laws to
 compare the difference.


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

import logging
import os
import pathlib
import shutil
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import porepy as pp
from tpf.derived_models.spe10 import INITIAL_PRESSURE, SPE10Mixin
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.models.protocol import TPFProtocol
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.utils.constants_and_typing import FEET
from tpf.viz.solver_statistics import SolverStatisticsHC

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

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

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
    "spe10_quarter_domain": False,
    "spe10_layer": spe10_layer,
    "spe10_isotropic_perm": True,
    # Nonlinear solver:
    "nl_enforce_physical_saturation": True,
}

time_manager_params: dict[str, Any] = {
    "schedule": np.array([0.0, 30.0 * pp.DAY]),
    "dt_init": 30.0 * pp.DAY,
    "constant_dt": True,
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
    cp_model_1: dict[str, Any]
    cp_model_2: dict[str, Any]


def setup_solver(adaptive_error_ratio: float) -> tuple[type[SPE10HC], dict[str, Any]]:
    """Return a tuple of solver-specific parameters and model class based on the solver
    name.

    Parameters:
        adaptive_error_ratio: The adaptive error ratio to be used in the homotopy
            continuation solver.

    Returns:
        A tuple ``(model_class, solver_params)``, where ``model_class`` is the model
        class with the correct adaptive solver and ``solver_params`` is a dictionary
        containing solver parameters.

    """

    solver_params: dict[str, Any] = {
        # Homotopy Continuation (HC) parameters:
        "nonlinear_solver_statistics": SolverStatisticsHC,
        "nonlinear_solver": HCSolver,
        "hc_max_iterations": 30,
        "hc_constant_decay": False,
        "hc_lambda_decay": 0.9,
        "hc_decay_min_max": (0.1, 0.9),
        "nl_iter_optimal_range": (4, 7),
        "nl_iter_relax_factors": (0.7, 1.3),
        "hc_decay_recomp_max": 5,
        # Adaptivity:
        "hc_adaptive": True,
        "hc_error_ratio": adaptive_error_ratio,  # adaptive error for homotopy
        "nl_error_ratio": 0.1,
        "hc_nl_convergence_tol": 1e-3,
        # Nonlinear solver parameters:
        "nl_convergence_tol": 1e-5,
        "nl_divergence_tol": 1e30,
        "max_iterations": 20,
        "nl_appleyard_chopping": False,
    }
    model_class = SPE10HC
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
        f"CP model: {config.cp_model_2}."
    )

    model_class, solver_params = setup_solver(config.adaptive_error_ratio)

    # Build params dictionary
    params = default_params.copy()
    params.update(solver_params)

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
            "meshing_arguments": {"cell_size": config.cell_size},
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": cap_press_constants,
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
        config.folder_name.mkdir(parents=True)
    except Exception:
        pass

    try:
        model = model_class(params)
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as e:
        with (config.folder_name / "failure.txt").open("w") as f:
            f.write(str(e))
        logger.error(f"Run failed with error: {e}.")


# endregion

# region RUN


if __name__ == "__main__":
    rp_models: dict[str, Any] = {
        "linear": {
            "model": "linear",
            "limit": True,
        },
        "Brooks-Corey": {
            "model": "Brooks-Corey-Mualem",
            "limit": True,
            "n_b": 1.0,
            "eta": 2.0,
        },  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1
    }
    cp_models: dict[str, Any] = {
        "None": {
            "model": None,
        },
        "linear_30": {
            "model": "linear",
            "entry_pressure": 30 * pp.PASCAL,
            "linear_param": 3.0,
        },
    }

    init_s: float = 0.3

    # Linear
    config = SimulationConfig(
        file_name=f"linear_{init_s}",
        folder_name=dirname / f"linear_{init_s}",
        solver_name="AHC",  # Disregarded
        adaptive_error_ratio=0.005,
        cell_size=600 * FEET / 30,
        init_s=0.3,
        rp_model_1=rp_models["linear"],
        rp_model_2=rp_models["linear"],
        cp_model_1=cp_models["None"],
        cp_model_2=cp_models["None"],
    )
    # run_simulation(config)

    # Nonlinear
    config = SimulationConfig(
        file_name=f"nonlinear_{init_s}",
        folder_name=dirname / f"nonlinear_{init_s}",
        solver_name="AHC",  # Disregarded
        adaptive_error_ratio=0.005,
        cell_size=600 * FEET / 30,
        init_s=0.3,
        rp_model_1=rp_models["linear"],
        rp_model_2=rp_models["Brooks-Corey"],
        cp_model_1=cp_models["None"],
        cp_model_2=cp_models["linear_30"],
    )
    # run_simulation(config)

    # Nonlinear
    config = SimulationConfig(
        file_name=f"nonlinear_stop_early_{init_s}",
        folder_name=dirname / f"nonlinear_stop_early_{init_s}",
        solver_name="AHC",  # Disregarded
        adaptive_error_ratio=1.0,  # Stop early
        cell_size=600 * FEET / 30,
        init_s=0.2,
        rp_model_1=rp_models["linear"],
        rp_model_2=rp_models["Brooks-Corey"],
        cp_model_1=cp_models["None"],
        cp_model_2=cp_models["linear_30"],
    )
    run_simulation(config)


# endregion
