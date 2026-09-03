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

import copy
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
from ahc.models.analytics import ErrorEstimateAnalyticsMixin
from ahc.models.homotopy_continuation import TwoPhaseFlowHC
from ahc.models.protocol import TPFProtocol
from ahc.utils.compare import ComparisonMixin, save_comparison_stats
from ahc.viz.iteration_exporting import IterationExportingMixin

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig, setup_porepy_params

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
    ErrorEstimateAnalyticsMixin,
    ComparisonMixin,
    InitialConditionsMixin,
    SPE11Mixin,
    TwoPhaseFlowHC,
):  # type: ignore
    ...


class SPE11Newton(
    ErrorEstimateAnalyticsMixin,
    ComparisonMixin,
    InitialConditionsMixin,
    SPE11Mixin,
    TwoPhaseFlowANewton,
):  # type: ignore
    ...


# endregion

# region UTILS

# Set solver and time manager parameters specific to all SPE11 simulations.
default_params = {
    "progressbars": True,
    # Model:
    "material_constants": {},
    "grid_type": "simplex",
    # SPE11 parameters:
    "spe11_case": "B",
    "spe11_heterogeneous_cap_pressure": False,
    # Error estimator:
    "disable_spatial_est": False,
}


default_time_manager_params = {
    "schedule": np.array([0.0, 3000.0 * pp.DAY]),
    "dt_init": 3000.0 * pp.DAY,
    "constant_dt": False,
    "dt_min_max": (1 * pp.DAY, 3000.0 * pp.DAY),
}


def setup_porepy_model(
    config: SimulationConfig, **kwargs
) -> type[SPE11HC] | type[SPE11Newton]:
    """Return a model class based on the solver name.

    Parameters:
        config: The simulation configuration specifying the solver type.
        **kwargs: Additional keyword arguments. Currently, only ``iteration_exporting``
            is supported, which adds the ``IterationExportingMixin`` to the model class.

    Returns:
        The model class with the correct adaptive solver.

    """
    if config.solver_name in ["HC", "AHC", "ReferenceSolution"]:
        model_class = SPE11HC
    elif config.solver_name in ["Newton", "NewtonAppleyard"]:
        model_class = SPE11Newton
    else:
        raise ValueError(f"Unknown solver: {config.solver_name}")

    if kwargs.get("iteration_exporting", False):
        model_class = type(
            f"{model_class.__name__}WithIterationExporting",
            (IterationExportingMixin, model_class),
            {},
        )

    return model_class


def run_simulation(
    config: SimulationConfig,
    params: dict | None = None,
    time_manager_params: dict | None = None,
    **kwargs,
) -> None:
    """Run simulation for a single configuration.

    Parameters:
        config: The simulation configuration.
        params: Optional dictionary of model and solver parameters. If None, default
            parameters are used and updated with the parameters from the config.
        time_manager_params: Optional dictionary of time manager parameters. If None,
            default parameters are used and updated with the parameters from the config.
        **kwargs: Additional keyword arguments to pass to the model setup and parameter
            setup functions.

    """
    logger.info(
        f"solver: {config.solver_name}, "
        f"HC tolerance: {config.hc_tol:.2f}, "
        f"NL tolerance: {config.nl_tol:.2f}, "
        f"cell size: {config.spe10_cell_size:.2f}, \n"
        f"initial saturation: {config.init_s}, "
        f"RP model 1: {config.rp_model_1}, "
        f"RP model 2: {config.rp_model_2}, \n"
        f"CP model: {config.cp_model_2}."
        f"Buoyancy: {config.buoyancy_constants_2}"
    )

    model_kwargs = kwargs.get("model_kwargs", {})
    # This triple parameter dict construction is ugly, but we avoid adding every
    # parameter we ever want to change to SimulationConfig or setup_porepy_params.
    additional_params = kwargs.get("additional_params", {})
    additional_time_manager_params = kwargs.get("additional_time_manager_params", {})

    model_class = setup_porepy_model(config, **model_kwargs)
    updated_params, updated_time_manager_params = setup_porepy_params(config)

    # Build porepy params dictionaries.
    if params is None:
        params = copy.deepcopy(default_params) | updated_params | additional_params
    if time_manager_params is None:
        time_manager_params = (
            copy.deepcopy(default_time_manager_params)
            | updated_time_manager_params
            | additional_time_manager_params
        )

    folder_name = config.folder_name()

    # Update SPE11 specific params.
    # solver_params is not None at this point. Ignore pylance complaining.
    params.update(  # type: ignore
        {
            # Meshing and model:
            "meshing_arguments": {
                "spe11_refinement_factor": config.spe11_refinement_factor
            },
            "spe11_initial_saturation": config.init_s,
            "spe11_entry_pressure": config.spe11_entry_pressure,
        }
    )

    # Add TimeManager to params.
    params.update(  # type: ignore
        {
            # All required parameters for TimeManager are included, ignore pylance
            # complaining.
            "time_manager": pp.TimeManager(**time_manager_params),  # type: ignore
        }
    )

    # Remove previous runs.
    shutil.rmtree(folder_name, ignore_errors=True)
    folder_name.mkdir(parents=True)

    is_reference_solution = config.solver_name == "ReferenceSolution"

    try:
        model = model_class(params)
        pp.run_time_dependent_model(model=model, params=params)

        if is_reference_solution:
            model.save_solution()

    # It is okay to catch general exceptions because we recognize failed simulations
    # in plotting.py.
    except Exception as exception:
        logger.error(f"Run failed with exception: {exception}.")
        raise exception

    # Save comparison stats if a reference solution exists. If not, skip this step.

    # The reference solution was saved in the parallel path for the
    # ReferenceSolution solver instead of the current solver.
    reference_solution_file = (
        pathlib.Path(
            *(
                "ReferenceSolution_0.010_1.00e-02" if p == config.solver_specs() else p
                for p in folder_name.parts
            )
        )
        / "solution.npy"
    )

    if not is_reference_solution and reference_solution_file.exists():
        reference_solution = np.load(reference_solution_file)
        absolute_stats, relative_stats = model.compare_with_reference(
            reference_solution
        )
        save_comparison_stats(
            absolute_stats, folder_name / "absolute_comparison_stats.json"
        )
        save_comparison_stats(
            relative_stats, folder_name / "relative_comparison_stats.json"
        )

    # Save number of grid cells to a file.
    with (folder_name / "num_grid_cells.txt").open("w") as f:
        f.write(str(model.g.num_cells))


# endregion

# region SIMULATIONS
solvers_and_tols: list[tuple[str, float, float]] = [
    # ("ReferenceSolution", 0.01, 0.01),
    # ("AHC", 0.01, 0.01),
    # ("AHC", 0.1, 0.1),
    # ("AHC", 0.1, 0.01),
    # ("AHC", 0.01, 0.01),
    # ("HC", 0.05, 1e-3),
    # ("HC", 0.01, 1e-3),
    # ("HC", 0.01, 1e-5),
    # ("Newton", 0.0, 0.1),
    ("NewtonAppleyard", 0.0, 0.1),
]
refinement_factors: list[float] = [10, 5, 1]
SPE11_ENTRY_PRESSURE: float = 1.0  # [Pa]


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
    "Corey_power_2": {"model": "Corey", "power": 2, "limit": True},
    "Corey_power_3": {"model": "Corey", "power": 3, "limit": True},
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
                    spe11_refinement_factor=refinement_factors[2],
                    spe11_entry_pressure=SPE11_ENTRY_PRESSURE,
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
                    spe11_refinement_factor=refinement_factors[2],
                    spe11_entry_pressure=SPE11_ENTRY_PRESSURE,
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
            # clean_up_after_simulation(config)
