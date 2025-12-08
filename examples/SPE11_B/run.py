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
import sys
import warnings
from typing import Any, Type

import numpy as np
import porepy as pp
from tpf.derived_models.spe11 import SPE11Mixin, case_B
from tpf.models.adaptive_newton import TwoPhaseFlowANewton
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.models.protocol import TPFProtocol

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig, clean_up_after_simulation, setup_params

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

default_solver_params = {
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


def setup_model(solver: str) -> Type[SPE11HC] | Type[SPE11Newton]:
    """Return a model class based on the solver name.

    Parameters:
        solver: The name of the solver ("AHC", "HC", "Newton", or "NewtonAppleyard").

    Returns:
        The model class with the correct adaptive solver.

    """
    if solver in ["HC", "AHC"]:
        return SPE11HC
    elif solver in ["Newton", "NewtonAppleyard"]:
        return SPE11Newton
    else:
        raise ValueError(f"Unknown solver: {solver}")


def run_simulation(
    config: SimulationConfig,
    solver_params: dict | None = None,
    time_manager_params: dict | None = None,
) -> None:
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

    model_class = setup_model(config.solver_name)
    updated_solver_params, updated_time_manager_params = setup_params(
        config.solver_name, config.adaptive_error_ratio
    )

    # Build params dictionaries.
    if solver_params is None:
        solver_params = default_solver_params | updated_solver_params
    if time_manager_params is None:
        time_manager_params = default_time_manager_params | updated_time_manager_params

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

    solver_params.update(
        {
            "meshing_arguments": {"spe11_refinement_factor": config.refinement_factor},
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": cap_press_constants,
            "spe11_initial_saturation": config.init_s,
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
        model = model_class(solver_params)
        pp.run_time_dependent_model(model=model, params=solver_params)

    except Exception as e:
        config.folder_name.mkdir(parents=True, exist_ok=True)
        with (config.folder_name / "failure.txt").open("w") as f:
            f.write(str(e))
        logger.error(f"Run failed with error: {e}.")


# endregion

# region RUN
solvers_and_ratios: list[tuple[str, float]] = [
    ("AHC", 0.01),
    # ("HC", 0.1),
    # ("Newton", 0.1),
    # ("NewtonAppleyard", 0.1),
]
refinement_factors: list[float] = [10, 3, 0.5]  # , 0.5]

rp_models: dict[str, Any] = {
    "linear": {"model": "linear", "limit": True},
    "Brooks-Corey_nb_4": {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 4.0,
        "eta": 2.0,
    },  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1
    "Brooks-Corey_nb_2": {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 2.0,
        "eta": 2.0,
    },  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1
    "Corey_power_2": {"model": "Corey", "limit": True, "power": 2},
    "Corey_power_3": {"model": "Corey", "limit": True, "power": 3},
}

cp_models: dict[str, Any] = {
    "None": {"model": None},
    "linear": {
        "model": "linear",
        "linear_param": 5.0,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
    "Brooks-Corey_nb_4": {
        "model": "Brooks-Corey",
        "n_b": 4.0,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
    "Brooks-Corey_nb_2": {
        "model": "Brooks-Corey",
        "n_b": 2.0,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
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
                        cp_model_2=cp_models["Brooks-Corey_nb_4"],
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
                        rp_model_2=rp_models["Brooks-Corey_nb_4"],
                        cp_model_1=cp_models["None"],
                        cp_model_2=cp_models["Brooks-Corey_nb_4"],
                    )
                )

    return configs


if __name__ == "__main__":
    configs = generate_configs()
    for config in configs:
        run_simulation(config)
        clean_up_after_simulation(config)

# endregion
