r"""Run the SPE11 A example with small time steps for plotting.


Model description:
- Constant CO2 injection in the center.
- No flow boundary condition on the sides and bottom. Homogeneous Dirichlet on top.
- Simulation time: 10 days
- Solid properties:
    - Porosity: SPE11, case A.
    - Permeability: SPE11, case A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.1.
    - CO2: From the NIST database, taken at 20°C and atmospheric pressure. Residual
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
from typing import Any

import numpy as np
import porepy as pp
from tpf.derived_models.spe11 import SPE11Mixin, case_B
from tpf.models.flow_and_transport import TwoPhaseFlow
from tpf.models.protocol import TPFProtocol
from tpf.viz.solver_statistics import SolverStatisticsANewton

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


class SPE11Newton(
    InitialConditionsMixin,
    SPE11Mixin,
    TwoPhaseFlow,
):  # type: ignore
    ...


# endregion

# region UTILS
default_params: dict[str, Any] = {
    "progressbars": True,
    # Model:
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    # SPE11 parameters:
    "spe11_case": "B",
    "spe11_heterogeneous_cap_pressure": False,
    "spe11_entry_pressure": 500.0,  # [Pa]
    # Nonlinear solver:
    "nl_enforce_physical_saturation": True,
}

time_manager_params: dict[str, Any] = {
    "schedule": np.array([0.0, 3000.0 * pp.DAY]),
    "dt_init": 1 * pp.DAY,
    "constant_dt": True,
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


def setup_solver() -> tuple[type[SPE11Newton], dict[str, Any]]:
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

    solver_params = {
        # Newton solver params with Appleyard chopping:
        "nonlinear_solver_statistics": SolverStatisticsANewton,
        "nonlinear_solver": pp.NewtonSolver,
        "nl_adaptive": True,
        "nl_error_ratio": 0.1,
        "nl_adaptive_convergence_tol": 1e-2,
        "nl_convergence_tol": 1e-5,
        "nl_divergence_tol": 1e30,
        "nl_appleyard_chopping": True,
        "max_iterations": 50,
    }
    model_class = SPE11Newton
    return model_class, solver_params


def run_simulation(config: SimulationConfig) -> None:
    """Run simulation for a single configuration."""
    logger.info(
        f"solver: {config.solver_name}, "
        f"adaptive error ratio: {config.adaptive_error_ratio:.2f}, "
        f"refinement factor: {config.refinement_factor:.2f}, "
        f"initial saturation: {config.init_s}, "
        f"RP model 1: {config.rp_model_1}, "
        f"RP model 2: {config.rp_model_2}, "
        f"CP model: {config.cp_model_2}."
    )

    model_class, solver_params = setup_solver()

    # Build params dictionary
    params = default_params.copy()
    params.update(solver_params)

    rel_perm_constants = config.rp_model_2
    cap_press_constants = config.cp_model_2

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

    model = model_class(params)
    pp.run_time_dependent_model(model=model, params=params)


# endregion

# region RUN


if __name__ == "__main__":
    rp_model: dict[str, Any] = {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 1.0,
        "eta": 2.0,
    }  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1

    cp_model: dict[str, Any] = {
        "model": "Brooks-Corey",
        "n_b": 2.0,
    }
    config = SimulationConfig(
        file_name="plotting",
        folder_name=dirname / "plotting",
        solver_name="NewtonAppleyard",
        adaptive_error_ratio=0.0,  # Disregarded
        refinement_factor=1.0,
        init_s=0.8,
        rp_model_1=rp_model,
        rp_model_2=rp_model,
        cp_model_1=cp_model,
        cp_model_2=cp_model,
    )

    run_simulation(config)
# endregion
