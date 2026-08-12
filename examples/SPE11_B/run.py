"""Study convergence of solvers on different grid sizes and with different rel.
 perm./cap. pressure models.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton
- Newton
- Newton with Appleyard chopping


Model description:
- Constant CO2 injection in the center.
- No flow boundary condition on the sides and bottom. Homogeneous Dirichlet on top.
- Simulation time: 3000 days
- Solid properties:
    - Porosity: SPE11, case B.
    - Permeability: SPE11, case B.
- Fluid properties:
    - Water: ``pp.fluid_values.water``. Residual saturation is 0.15.
    - CO2: Reservoir conditions (350 bar, 70°C). Residual saturation is 0.1.
- Initial values:
    - Pressure: 30 MPa (reservoir pressure).
    - Saturation: Varying between 0.8 and 0.9.
- Rel. perm. models (model_2; model_1 is always linear for HC/AHC):
    - Brooks-Corey-Mualem with n_b=2, eta=2.
    - Brooks-Corey-Mualem with n_b=4, eta=2.
    - Corey with power 2.
    - Corey with power 3.
- Capillary pressure model (model_2; model_1 is always None for HC/AHC):
    - Brooks-Corey with n_b=4.

"""

import logging
import os
import pathlib
import shutil
import sys
import warnings
from typing import Any

import numpy as np
import porepy as pp
from ahc.derived_models.spe11 import SPE11Mixin, case_B
from ahc.models.adaptive_newton import TwoPhaseFlowANewton
from ahc.models.homotopy_continuation import TwoPhaseFlowHC
from ahc.models.protocol import TPFProtocol

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig, clean_up_after_simulation, setup_porepy_params

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

# Directories for results.
dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()
results_dir = dirname / "results"

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
    TwoPhaseFlowHC,
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
    "spe11_entry_pressure": 30.0,  # [Pa]
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


def setup_porepy_model(solver: str) -> type[SPE11HC] | type[SPE11Newton]:
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
    **kwargs,
) -> None:
    """Run simulation for a single configuration."""
    logger.info(
        f"solver: {config.solver_name}, "
        f"HC tolerance: {config.hc_tol:.2f}, "
        f"NL tolerance: {config.nl_tol:.2f}, "
        f"cell size: {config.cell_size:.2f}, \n"
        f"initial saturation: {config.init_s}, "
        f"RP model 1: {config.rp_model_1}, "
        f"RP model 2: {config.rp_model_2}, \n"
        f"CP model: {config.cp_model_2}."
    )

    model_class = setup_porepy_model(config.solver_name)
    updated_solver_params, updated_time_manager_params = setup_porepy_params(
        config, **kwargs
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

    folder_name = config.folder_name()
    solver_params.update(
        {
            # Meshing and model:
            "meshing_arguments": {"spe11_refinement_factor": config.refinement_factor},
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": cap_press_constants,
            "spe11_initial_saturation": config.init_s,
            "spe11_entry_pressure": config.spe11_entry_pressure,
            # Output:
            "folder_name": folder_name,
            "file_name": config.case.name
            if isinstance(config.case, pathlib.Path)
            else config.case,
            "solver_statistics_file_name": folder_name / "solver_statistics.json",
            "time_manager": pp.TimeManager(**time_manager_params),
        }
    )

    # Remove previous runs.
    shutil.rmtree(folder_name, ignore_errors=True)
    folder_name.mkdir(parents=True)

    try:
        model = model_class(solver_params)
        pp.run_time_dependent_model(model=model, params=solver_params)
    except Exception as e:
        logger.error(f"Run failed with error: {e}.")

    # Save number of grid cells to a file.
    with (folder_name / "num_grid_cells.txt").open("w") as f:
        f.write(str(model.g.num_cells))


# endregion

# region SIMULATIONS
solvers_and_tols: list[tuple[str, float, float]] = [
    ("AHC", 0.1, 0.1),
    ("AHC", 0.01, 0.1),
    ("HC", 0.01, 1e-3),
    ("HC", 0.01, 1e-5),
    ("Newton", 0.0, 0.1),
    ("NewtonAppleyard", 0.0, 0.1),
]
refinement_factors: list[float] = [10, 5, 1]

LINEAR_RP_MODEL = {"model": "linear", "limit": True}
rp_models: dict[str, Any] = {
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

ZERO_CP_MODEL = {"model": None}
cp_models: dict[str, Any] = {
    "Brooks-Corey_nb_4": {
        "model": "Brooks-Corey",
        "n_b": 4.0,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
}

ZERO_BUOYANCY_MODEL = {"gravity_acceleration": 0.0}


def generate_viscous_varying_rp_cases(init_s: float) -> list[SimulationConfig]:
    """Generate simulation configurations for viscous-dominated flow with varying
    relative permeability models, Brooks-Corey capillary pressure, and prescribed
    initial saturation.

    """
    cases = []
    for rp_model_name, rp_model in rp_models.items():
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous",
                    study="varying_rp",
                    case=pathlib.Path(f"init_s_{init_s}") / rp_model_name,
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=init_s,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_model,
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_models["Brooks-Corey_nb_4"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                )
            )

    return cases


def generate_viscous_varying_ref_factor(init_s: float) -> list[SimulationConfig]:
    """Generate simulation configurations for viscous-dominated flow with varying
    grid refinement factor, linear capillary pressure, and prescribed initial
    saturation.

    """
    cases = []
    for ref_factor in refinement_factors:
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous",
                    study="varying_refinement",
                    case=pathlib.Path(f"init_s_{init_s}") / f"ref_fac_{ref_factor:.2f}",
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=init_s,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_models["Brooks-Corey_nb_4"],
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_models["Brooks-Corey_nb_4"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                )
            )

    return cases


studies: dict[str, list[SimulationConfig]] = {
    "viscous_varying_rp_init_s_08": generate_viscous_varying_rp_cases(init_s=0.8),
    "viscous_varying_rp_init_s_09": generate_viscous_varying_rp_cases(init_s=0.9),
    "viscous_varying_ref_factor_init_s_08": generate_viscous_varying_ref_factor(
        init_s=0.8
    ),
    "viscous_varying_ref_factor_init_s_09": generate_viscous_varying_ref_factor(
        init_s=0.9
    ),
}

# endregion

if __name__ == "__main__":
    results_dir.mkdir(exist_ok=True)
    for study in studies.values():
        for config in study:
            run_simulation(config)
            clean_up_after_simulation(config)
