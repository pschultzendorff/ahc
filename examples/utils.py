import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import seaborn as sns
from ahc.derived_models.spe10 import water
from ahc.numerics.nonlinear.hc_solver import HCSolver
from ahc.numerics.nonlinear.newton import ModifiedNewtonSolver
from ahc.utils.compare import ComparisonStats
from ahc.utils.constants_and_typing import FEET
from ahc.viz.solver_statistics import SolverStatisticsANewton, SolverStatisticsHC
from matplotlib.ticker import (
    FuncFormatter,
    LogFormatter,
    LogLocator,
    MaxNLocator,
)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

# Setup logging.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

sns.set_theme("paper")
sns.set_style("whitegrid")


# region RUNNING
@dataclass(unsafe_hash=True)
class SimulationConfig:
    """Class to store all simulation parameters for one simulation."""

    # Metadata for the simulation.
    results_dir: pathlib.Path
    """Main directory for simulation results."""
    regime: str
    """Flow regime, e.g., viscous, capillary, or gravity dominated."""
    study: str
    """Study name, e.g., `varying_water_density.` for flow regimes with gravity."""
    case: str | pathlib.Path
    """Case name, e.g., `water_density_1000` for the `varying_water_density`
    experiment.

    Note: May also be a path to support further ordering of cases, e.g.,
        `varying_rp/init_s_0.2/Brooks_Corey_nb_4` for the `varying_rp` experiment with
        initial saturation 0.2 and Brooks-Corey model with n_b=4.

    """

    # Solver parameters.
    solver_name: str
    """Nonlinear solver name, i.e., "AHC", "HC", "Newton", or "NewtonAppleyard"."""
    hc_tol: float
    r"""Solver convergence parameter for the outer HC loop (if required).
    
    - AHC: Adaptive error ratio :math:`\gamma_{\mathrm{cont}}`
    - HC: Minimum :math:`\beta`

    """
    nl_tol: float
    r"""Solver convergence parameter for the inner Newton loop.

    - AHC: Adaptive error ratio :math:`\gamma_{\mathrm{lin}}`
    - Newton: Adaptive error ratio :math:`\gamma_{\mathrm{lin}}`
    - HC: Absolute and relative tolerance for the inner Newton loop.
    
    """

    # Model parameters.
    init_s: float
    rp_model_1: dict[str, Any]
    """Relative permeability model for the auxiliary problem in the HC."""
    rp_model_2: dict[str, Any]
    """Relative permeability model for the target problem for all solvers."""
    cp_model_1: dict[str, Any]
    """Capillary pressure model for the auxiliary problem in the HC."""
    cp_model_2: dict[str, Any]
    """Capillary pressure model for the target problem for all solvers."""
    buoyancy_constants_1: dict[str, Any]
    """Buyancy constants for the auxiliary problem in the HC."""
    buoyancy_constants_2: dict[str, Any]
    """Buyancy constants for the target problem for all solvers."""

    solver_name_postfix: str = ""
    """Optional postfix to the solver name for when generating folder names, e.g., for
    different parameter values. Default is an empty string.

    """

    # SPE10 specific model parameters.
    spe10_cell_size: float = 600 * FEET / 30  # Default cell size.
    """Cell size parameter to be passed to gmsh for the SPE10 model."""
    spe10_layer: int = 55
    """SPE10 layer number. Possible values are 0 to 84. Default is 55."""
    spe10_case: str = "five_spot"
    """SPE10 case name. Possible values are "five_spot" and "gravity_segregation".
    Default is "five_spot". 

    """
    spe10_water_density: float = water["density"]  # kg/m^3
    r"""Water density for the SPE10 model. Default is :math:`998.2\,\mathrm{kg/m^3}`."""

    # SPE11 specific model parameters.
    spe11_refinement_factor: float = 1.0
    """Grid refinement factor for the SPE11 model. Default is 1.0, which means no
    refinement. Values larger than 1.0 mean a coarser grid and values smaller than 1.0
    mean a finer grid. 

    See https://github.com/Simulation-Benchmarks/11thSPE-CSP/tree/main/geometries.

    """
    spe11_entry_pressure: float = 30 * pp.PASCAL
    r"""Homogenized entry pressure for the SPE11 model. Default is
    :math:`30\,\mathrm{Pa}`."""

    def solver_specs(self) -> str:
        """Get the solver specifications as a string."""
        postfix_with_underscore = (
            f"_{self.solver_name_postfix}" if self.solver_name_postfix else ""
        )
        solver_name_with_params = (
            f"{self.solver_name}{postfix_with_underscore}"
            f"_{self.hc_tol:.3f}_{self.nl_tol:.2e}"
        )
        return solver_name_with_params

    def folder_name(self) -> pathlib.Path:
        """Create a folder path from metadata and solver parameters.

        Returns:
            name: Path to the folder for the simulation results.

        """
        return (
            self.results_dir
            / self.solver_specs()
            / self.regime
            / self.study
            / self.case
        )


def setup_porepy_params(
    config: SimulationConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Setup model, solver, and time manager parameters in PorePy format.

    Note: Many of the solver and time stepping parameters for all benchmarks are
    hardcoded in this function.

    Parameters:
        config: The simulation configuration specifying nonlinear solver type and
            tolerances.

    Returns:
        params: A dictionary containing all parameters for the model and solver, WITHOUT
            an initialized ``time manager``. The dictionary will be passed to the model
            class for initialization and to ``run_model``.
        time_manager_params: A dictionary containing parameters for the time manager.

    """
    solver_name = config.solver_name
    hc_tol = config.hc_tol
    nl_tol = config.nl_tol
    logger.info(
        f"solver: {solver_name}, HC tolerance: {hc_tol:.2f}, NL tolerance: {nl_tol:.2f}."
        " Generating solver and time manager parameters."
    )

    # 1st: Setup nonlinear solver parameters.
    # Params shared between all solvers.
    params = {
        # NOTE The first two are only needed for adaptive solvers. For the nonadaptive
        # HC solver, it won't have any effect.
        "adaptive_threshold_rel": 10.0,
        # When cutting time steps, project the temporal estimator onto the
        # original time step length. This avoids strengthening the adaptive
        # criteria. The temporal convergence is assumed to be sublinear in the
        # time step length.
        "extrapolate_temp_estimator_after_cutting": 0.75,
        # NOTE Now come parameters valid for all solvers.
        "nl_divergence_tol": 1e30,
        "nl_max_iterations": 30,
        "nl_enforce_physical_saturation": True,
    }

    # Solver specific params.
    match solver_name:
        case "HC":
            params.update(
                {
                    # Metadata:
                    "nonlinear_solver_statistics": SolverStatisticsHC,
                    "nonlinear_solver": HCSolver,
                    # HC parameters:
                    "hc_constant_decay": False,
                    "hc_max_iterations": 50,
                    "hc_lambda_decay": 0.9,
                    "hc_decay_min_max": (0.1, 0.95),
                    "nl_iter_optimal_range": (6, 9),
                    "nl_iter_relax_factors": (0.7, 1.3),
                    "hc_decay_recomp_max": 5,
                    # Non-adaptive stopping criteria for HC:
                    "hc_adaptive": False,
                    "hc_lambda_min": hc_tol,  # Minimum :math:`\beta` for HC.
                    # Non-adaptive stopping and other Newton solver parameters:
                    "nl_convergence_tol_abs": nl_tol,  # Absolute tolerance for inner loop.
                    "nl_convergence_tol_rel": nl_tol,  # Relative tolerance for inner loop.
                    "nl_max_iterations": 30,
                    "nl_appleyard_chopping": False,
                    "nl_enforce_physical_saturation": True,
                }
            )

        case "AHC":
            params.update(
                {
                    # Metadata:
                    "nonlinear_solver_statistics": SolverStatisticsHC,
                    "nonlinear_solver": HCSolver,
                    # HC parameters:
                    "hc_max_iterations": 50,
                    "hc_constant_decay": False,
                    "hc_lambda_decay": 0.9,
                    "hc_decay_min_max": (0.1, 0.95),
                    "nl_iter_optimal_range": (6, 9),
                    "nl_iter_relax_factors": (0.7, 1.3),
                    "hc_decay_recomp_max": 5,
                    # Adaptive stopping criteria for HC and Newton:
                    "hc_adaptive": True,
                    "hc_error_ratio": hc_tol,  # Adaptive error ratio for outer loop.
                    "nl_error_ratio": nl_tol,  # Adaptive error ratio for inner loop.
                    # Non-adaptive stopping and other Newton solver parameters:
                    "nl_convergence_tol_abs": 1e-5,
                    "nl_convergence_tol_rel": 1e-5,
                    "nl_max_iterations": 30,
                    "nl_appleyard_chopping": False,
                    "nl_enforce_physical_saturation": True,
                }
            )

        case "ReferenceSolution":
            # The reference solution is computed with the AHC solver.
            params.update(
                {
                    # Metadata:
                    "nonlinear_solver_statistics": SolverStatisticsHC,
                    "nonlinear_solver": HCSolver,
                    # HC parameters:
                    "hc_max_iterations": 50,
                    "hc_constant_decay": False,
                    "hc_lambda_decay": 0.9,
                    "hc_decay_min_max": (0.1, 0.95),
                    "nl_iter_optimal_range": (6, 9),
                    "nl_iter_relax_factors": (0.7, 1.3),
                    "hc_decay_recomp_max": 5,
                    # Adaptive stopping criteria for HC and Newton:
                    "hc_adaptive": True,
                    "hc_error_ratio": hc_tol,  # Adaptive error ratio for outer loop.
                    "nl_error_ratio": nl_tol,  # Adaptive error ratio for inner loop.
                    # Non-adaptive stopping and other Newton solver parameters:
                    "nl_convergence_tol_abs": 1e-5,
                    "nl_convergence_tol_rel": 1e-5,
                    "nl_max_iterations": 30,
                    "nl_appleyard_chopping": False,
                    "nl_enforce_physical_saturation": True,
                    "reference_solution": True,  # Flag to indicate that a reference
                    # solution shall be computed.
                }
            )

        case "Newton":
            params.update(
                {
                    # Metadata:
                    "nonlinear_solver_statistics": SolverStatisticsANewton,
                    "nonlinear_solver": ModifiedNewtonSolver,
                    # Adaptive stopping criteria for Newton:
                    "nl_adaptive": True,
                    "nl_error_ratio": nl_tol,  # Adaptive error ratio for Newton.
                    # Non-adaptive stopping parameters and other solver parameters:
                    "nl_convergence_tol_abs": 1e-5,
                    "nl_convergence_tol_rel": 1e-5,
                    "nl_appleyard_chopping": False,
                    "nl_enforce_physical_saturation": True,
                }
            )
        case "NewtonAppleyard":
            params.update(
                {
                    # Metadata:
                    "nonlinear_solver_statistics": SolverStatisticsANewton,
                    "nonlinear_solver": ModifiedNewtonSolver,
                    # Adaptive stopping criteria for Newton:
                    "nl_adaptive": True,
                    "nl_error_ratio": nl_tol,  # Adaptive error ratio for Newton.
                    # Non-adaptive stopping parameters and other solver parameters:
                    "nl_convergence_tol_abs": 1e-5,
                    "nl_convergence_tol_rel": 1e-5,
                    "nl_appleyard_chopping": True,
                }
            )
        case _:
            raise ValueError(f"Unknown solver: {solver_name}")

    # 2nd: Sepcify constitutive laws.
    # Newton and Appleyard Newton require only one of each constitutive law.
    if config.solver_name.startswith("Newton"):
        rel_perm_constants = config.rp_model_2
        cap_press_constants = config.cp_model_2
        buoyancy_constants = config.buoyancy_constants_2
    else:
        rel_perm_constants = {
            "model_1": config.rp_model_1,
            "model_2": config.rp_model_2,
        }
        cap_press_constants = {
            "model_1": config.cp_model_1,
            "model_2": config.cp_model_2,
        }
        buoyancy_constants = {
            "model_1": config.buoyancy_constants_1,
            "model_2": config.buoyancy_constants_2,
        }
    params.update(
        {
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": cap_press_constants,
            "buoyancy_constants": buoyancy_constants,
        }
    )

    # 3rd: Add metadata
    folder_name = config.folder_name()
    params.update(
        {
            "folder_name": folder_name,
            "file_name": config.case.name
            if isinstance(config.case, pathlib.Path)
            else config.case,
            "solver_statistics_file_name": folder_name / "solver_statistics.json",
        }
    )

    # 4th: Set the time manager parameters.
    time_manager_params = {
        "recomp_factor": 0.1,
        "recomp_max": 5,
    }
    if solver_name.endswith("HC"):
        # Update adaptive time stepping parameters for HC.
        time_manager_params.update(
            {
                "iter_optimal_range": (
                    15,
                    30,
                ),  # Higher than values for Newton, because typically more HC outer
                # iterations are required. Default value is (4, 7).
                "iter_relax_factors": (0.7, 1.3),
                "iter_max": 50,  # This should be the same as "hc_max_iterations", which
                # the TimeManager does not know about.
            }
        )
    else:
        # Update adaptive time stepping parameters for Newton.
        time_manager_params.update(
            {
                "iter_optimal_range": (
                    8,
                    20,
                ),  # Default value is (4, 7).
                "iter_relax_factors": (0.7, 1.3),  # Default value.
                "iter_max": 30,  # This should be the same as "nl_max_iterations", which
                # the TimeManager does not know about.
            }
        )

    return params, time_manager_params


def clean_up_after_simulation(config: SimulationConfig) -> None:
    """Delete simulation results not needed for the analysis, i.e., everything but
        statistic files.

    Parameters:
        config: The simulation configuration containing the folder name.

    """
    for pattern in ["*.vtu", "*.pvd"]:
        for file in config.folder_name().glob(pattern):
            file.unlink(missing_ok=True)
    (config.folder_name() / "times.json").unlink(missing_ok=True)


# endregion


# region PLOTTING


@dataclass
class SolverStats:
    """Class to store statistics read from a 'solver_statistics.json' file."""

    # IMPLEMENTATION NOTE The type of the list element will vary depending on the solver
    # type, as the HC solvers have nested lists for the inner loops We do not specify
    # this here.
    discrete_times: list = field(default_factory=list)
    time_step_convergence: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=bool)
    )
    time_step_sizes: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    time_step_nl_iters: list = field(default_factory=list)

    spat_estimator: list = field(default_factory=list)
    temp_estimator: list = field(default_factory=list)
    hc_estimator: list = field(default_factory=list)
    lin_estimator: list = field(default_factory=list)
    energy_norm: list = field(default_factory=list)

    lambdas: list = field(default_factory=list)
    converged: bool = True
    final_time: float = 0.0

    num_grid_cells: int = 1


def _flatten_nested_list(xx: list[list]) -> list:
    return [x for sublist in xx for x in sublist]


def _parse_hc_steps(
    time_step: dict[str, Any],
) -> tuple[
    list[int],
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    """Helper function to parse HC iterations within a time step and return the relevant
    statistics."""
    hc_steps = list(time_step.values())[:-5]

    num_nl_iterations = [s["num_iteration"] for s in hc_steps]
    spat_estimator = [s["spatial_est"] for s in hc_steps]
    temp_estimator = [s["temp_est"] for s in hc_steps]
    hc_estimator = [s["hc_est"] for s in hc_steps]
    lin_estimator = [s["lin_est"] for s in hc_steps]
    energy_norm = [s["global_energy_norm"] for s in hc_steps]

    return (
        num_nl_iterations,
        spat_estimator,
        temp_estimator,
        hc_estimator,
        lin_estimator,
        energy_norm,
    )


def _parse_newton_steps(
    time_step: dict[str, Any],
) -> tuple[int, list[float], list[float], list[float], list[float]]:
    """Helper function to parse Newton iterations within a time step and return the
    relevant statistics.

    """
    return (
        time_step["num_iteration"],
        time_step["spatial_est"],
        time_step["temp_est"],
        time_step["lin_est"],
        time_step["global_energy_norm"],
    )


def read_solver_stats(
    config: SimulationConfig,
    expected_final_time: float,
) -> SolverStats:
    """Read solver stats from a `.json` file.


    Parameters:
        config: _description_
        expected_final_time: _description_

    Raises:
        ValueError: _description_

    Returns:
        _description_

    """
    with (config.folder_name() / "solver_statistics.json").open() as f:
        data: dict[str, Any] = json.load(f)

    time_steps = list(data.values())
    stats = SolverStats()

    # Read number of grid cells, before possibly returning empty statistics (when
    # failed).
    try:
        stats.num_grid_cells = int(
            (config.folder_name() / "num_grid_cells.txt").read_text().strip()
        )
    except FileNotFoundError:
        return stats

    # Check if the simulation reached the final time.
    # NOTE In theory, the last time step could reach the final time but not converge,
    # with the simulation failing afterwards because time step size couldn't be reduced
    # further. This is rather unlikely, so we assume that the last time step converged
    # if the final time was reached. The better solution would be to store the
    # convergence status of each time step in nonlinear_solver_statistics.
    stats.final_time = time_steps[-1]["current time"]
    if not np.isclose(stats.final_time, expected_final_time):
        stats.converged = False

    # Skip the zeroth time step t=0.
    for time_step in time_steps[1:]:
        if (
            config.solver_name.endswith("HC")
            or config.solver_name == "ReferenceSolution"
        ):
            (
                num_nl_iterations,
                spat_estimator,
                temp_estimator,
                hc_estimator,
                lin_estimator,
                energy_norm,
            ) = _parse_hc_steps(time_step)

            stats.hc_estimator.append(hc_estimator)
            stats.lambdas.append(time_step["hc_lambdas"])
            stats.energy_norm.append(energy_norm)
        elif config.solver_name.startswith("Newton"):
            (
                num_nl_iterations,
                spat_estimator,
                temp_estimator,
                lin_estimator,
                energy_norm,
            ) = _parse_newton_steps(time_step)
        else:
            raise ValueError(f"Unknown solver: {config.solver_name}")

        # Append data to the statistics object.
        stats.discrete_times.append(time_step["current time"])
        stats.time_step_nl_iters.append(num_nl_iterations)
        stats.spat_estimator.append(spat_estimator)
        stats.temp_estimator.append(temp_estimator)
        stats.lin_estimator.append(lin_estimator)

    # Find failed time steps by checking whether the discrete time value is smaller than
    # the previous time step. The zeroth discrete time (t=0) is not saved in
    # stats.discrete_times. Convergence of the last time step is given by convergence of
    # the full simulation.
    num_time_steps = len(stats.discrete_times)
    stats.time_step_convergence = np.empty(num_time_steps, dtype=bool)
    stats.time_step_sizes = np.empty(num_time_steps, dtype=float)

    # Some simulations may have empty solver statistics , e.g., the reference solution
    # if it failed.
    # FIXME 2026-09-07: hc_solver was fixed to run after_hc_failure() for a failed
    # reference solution. Now, num_time_steps should always be >= 1. Currently, the
    # check is kept for legacy reasons.
    if num_time_steps >= 1:
        last_converged_time: float = 0.0

        for step, (current_time, next_time) in enumerate(
            zip(stats.discrete_times[:-1], stats.discrete_times[1:])
        ):
            stats.time_step_convergence[step] = current_time < next_time
            stats.time_step_sizes[step] = current_time - last_converged_time
            if stats.time_step_convergence[step]:
                last_converged_time = current_time

        # The last time step's convergence is determined by the overall convergence.
        stats.time_step_convergence[-1] = stats.converged
        stats.time_step_sizes[-1] = stats.final_time - last_converged_time

    return stats


def calc_relative_est(stats: SolverStats) -> dict[str, float]:
    """Calculate relative error estimators at the end of the simulation."""
    # Determine the solver by number of nested loops.
    solver_type = "Newton" if isinstance(stats.energy_norm[-1][-1], float) else "HC"

    energy_norm = (
        stats.energy_norm[-1][-1]
        if solver_type == "Newton"
        else stats.energy_norm[-1][-1][-1]
    )
    result = {}
    for est_name in ["total", "lin", "spat", "temp", "hc"]:
        if est_name == "total":
            if solver_type == "Newton":
                result["total"] = (
                    stats.lin_estimator[-1][-1]
                    + stats.spat_estimator[-1][-1]
                    + stats.temp_estimator[-1][-1]
                ) / energy_norm
            else:
                result["total"] = (
                    stats.hc_estimator[-1][-1][-1]
                    + stats.lin_estimator[-1][-1][-1]
                    + stats.spat_estimator[-1][-1][-1]
                    + stats.temp_estimator[-1][-1][-1]
                ) / energy_norm
        elif est_name == "hc":
            if solver_type == "HC":
                result[est_name] = stats.hc_estimator[-1][-1][-1] / energy_norm
        else:
            if solver_type == "Newton":
                result[est_name] = (
                    getattr(stats, est_name + "_estimator")[-1][-1] / energy_norm
                )
            else:
                result[est_name] = (
                    getattr(stats, est_name + "_estimator")[-1][-1][-1] / energy_norm
                )

    return result


def plot_nl_iterations(
    data: dict[tuple[str, str], SolverStats],
    varying_param_name: str,
    title: str | None = None,
    **kwargs,
):
    """Plot a heatmap of nonlinear iterations for different solvers and parameter values
    from one study.

    Parameters:
        data: Dictionary mapping solver specs and varying parameter values (as a tuple
            of strings) to simulation statistics.
        varying_param_name: Name of the parameter that varies between the
            configurations. This will be the title of the x-axis.
        title: Optional title for the plot.

    """
    # Loop through data and transform into an array that stores solver_specs,
    # parameter_value, and the statistics of interest.
    data_dtype = np.dtype(
        [
            # Make the strings long enough to avoid any issues.
            ("solver_specs", "U200"),
            ("parameter_value", "U100"),
            ("nl_iterations", "i8"),
            ("annotation", "U100"),
            ("converged", "?"),
            ("final_time", "float32"),
            ("final_time_step_size", "float32"),
        ]
    )
    data_as_array = np.zeros(len(data), dtype=data_dtype)

    for i, ((solver_specs, parameter_value), stats) in enumerate(data.items()):
        # Transform the solver specs into annotations. Unique annotations for each
        # combination of solver name and specs.
        solver_specs_list: list[str] = solver_specs.split("_")
        solver_name = solver_specs_list[0]
        solver_name_postfix = solver_specs_list[1] if len(solver_specs_list) > 3 else ""
        match solver_name:
            case "HC":
                data_as_array[i]["solver_specs"] = (
                    f"{solver_name}{solver_name_postfix}\n"
                    rf"$\beta_{{\min}} = {solver_specs_list[-2]}$"
                    "\n"
                    rf"$\epsilon_\mathrm{{Newton}} = {solver_specs_list[-1]}$"
                )
            case "AHC":
                data_as_array[i]["solver_specs"] = (
                    f"{solver_name}{solver_name_postfix}\n"
                    rf"$\gamma_\mathrm{{HC}} = {solver_specs_list[-2]}$"
                    "\n"
                    rf"$\gamma_\mathrm{{lin}} = {solver_specs_list[-1]}$"
                )
            case "Newton" | "NewtonAppleyard":
                data_as_array[i]["solver_specs"] = (
                    f"{solver_name}{solver_name_postfix}\n"
                    rf"$\gamma_\mathrm{{lin}} = {solver_specs_list[-1]}$"
                )
            case _:
                raise ValueError(f"Unknown solver: {solver_name}")

        data_as_array[i]["parameter_value"] = parameter_value

        # Now, read the stats of the case.
        data_as_array[i]["converged"] = stats.converged
        if not stats.converged:
            data_as_array[i]["final_time"] = stats.final_time
            data_as_array[i]["final_time_step_size"] = stats.time_step_sizes[-1]
            # Leave the other statistics empty if the solver did not converge.

        tot_nl_iterations = (
            sum(stats.time_step_nl_iters)
            if solver_name.startswith("Newton")
            else sum(_flatten_nested_list(stats.time_step_nl_iters))
        )
        data_as_array[i]["nl_iterations"] = tot_nl_iterations

        # Create annotations for the heatmap entries.
        # For HC and AHC, these include #nl_iters, #hc_iters, final beta value, and
        # #time_steps.
        if solver_name.endswith("HC"):
            tot_hc_iters = len(_flatten_nested_list(stats.time_step_nl_iters))
            # At the last lambda, the Newton solver was not run anymore. Check
            # solver_statistics.
            final_lambda = stats.lambdas[-1][-2]
            data_as_array[i]["annotation"] = (
                f"{tot_nl_iterations}/{tot_hc_iters}/{final_lambda:.4f}\n"
                f"({len(stats.discrete_times)}/{np.sum(np.logical_not(stats.time_step_convergence))})"
            )
        # For Newton, these include only #nl_iters and #time_steps.
        else:
            data_as_array[i]["annotation"] = (
                f"{tot_nl_iterations}\n"
                f"({len(stats.discrete_times)}/{np.sum(np.logical_not(stats.time_step_convergence))})"
            )

    # Create ticks for x- and y-axes.
    x_ticks = np.unique(data_as_array["parameter_value"])
    y_ticks = np.unique(data_as_array["solver_specs"])

    # Sort indices first by solver_specs, then by parameter_value.
    idx = np.lexsort((data_as_array["parameter_value"], data_as_array["solver_specs"]))
    # Apply sorting to data.
    data_as_array = data_as_array[idx]

    # Reshape each data column into an array of shape=(len(x_ticks), len(y_ticks)) for
    # the heatmap. Due to the previous sorting, the column value will appear at the x, y
    # value corresponding to the solver_specs and parameter_value.
    grids = {
        col: data_as_array[col].reshape(len(y_ticks), len(x_ticks))
        # stats_as_array.dtype.names will not be None. Ignore pylance.
        for col in data_as_array.dtype.names  # type: ignore
        if col not in ("solver_specs", "parameter_value")
    }

    # Now, we can finally create the heatmap figure.
    fig_height = len(y_ticks)
    fig_width = len(x_ticks) * 2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Number of total nonlinear iterations corresponds to shade of blue. Failed time
    # steps are marked red.
    cmap = matplotlib.colormaps["Blues"]
    cmap.set_bad(color="red")

    sns.heatmap(
        grids["nl_iterations"],
        mask=np.logical_not(grids["converged"]),
        annot=grids["annotation"],
        fmt="s",
        cmap=cmap,
        cbar=True,
        # Keep colorbar width approximately constant independent of height of figure.
        cbar_kws={
            "label": "Number of cumulative nonlinear iterations",
            "aspect": fig_height / fig_width * 20 / (8 / 5),
        },
        # It's okay to use numpy arrays here. Ignore pylance.
        xticklabels=x_ticks,  # type: ignore
        yticklabels=y_ticks,  # type: ignore
        linewidths=0.8,
        ax=ax,
    )

    # Annotate failed simulations with the final time reached.
    for i, j in np.argwhere(np.logical_not(grids["converged"])):
        ax.text(
            j + 0.5,
            i + 0.5,
            rf"$\Delta t = {grids['final_time_step_size'][i, j] / 86400:.1f}\,\mathrm{{d}}$"
            + "\n"
            rf"$t={grids['final_time'][i, j] / 86400:.1f}\,\mathrm{{d}}$"
            + "\n"
            + grids["annotation"][i, j],
            ha="center",
            va="center",
            fontsize=10,
            color="black",
        )

    # Set labels and title.
    ax.set_xlabel(varying_param_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Solver & adaptive error ratio", fontsize=12, fontweight="bold")
    ax.set_title(
        title
        or r"# cumulative NL iters/#HC iters/final $\beta$"
        + "\n"
        + "(#total time steps/#failed time steps) \n",
        # + f"by solver and {varying_param_name}",
        fontsize=14,
        fontweight="bold",
    )

    if kwargs.get("tight_layout", True):
        fig.tight_layout()

    # Long x-tick labels may be rotated by 45 degrees to fit the figure.
    if kwargs.get("rotate_x_labels", False):
        ax.tick_params(axis="x", labelrotation=45)
        # Ensure rotated x tick labels are not cut off at the bottom.
        fig.subplots_adjust(
            bottom=max(
                0.1,
                max(
                    (len(label.get_text()) for label in ax.get_xticklabels()),
                    default=0,
                )
                * 0.01,
            )
        )
        plt.draw()
        fig.tight_layout(pad=1.5)

    return fig


def plot_estimators(
    stats: SolverStats,
    title: str | None = None,
    combine_disc_est: bool = False,
    **kwargs,
) -> plt.Figure:
    """Create a plot showing the evolution of different error estimators over time.

    Returns:
        A matplotlib figure with the plotted estimators.

    """
    # Check if HC estimator is present.
    uses_hc: bool = len(stats.hc_estimator) > 0

    fig, ax = plt.subplots(figsize=(8, 6))

    if uses_hc:
        # Create a secondary y-axis for lambdas.
        ax2 = ax.twinx()

    tot_nl_iterations: int = 0
    tot_nl_iterations_fine: int = 0
    # Plot spatial estimator
    for i, (time, spat_est, temp_est, lin_est) in enumerate(
        zip(
            stats.discrete_times,
            stats.spat_estimator,
            stats.temp_estimator,
            stats.lin_estimator,
        )
    ):
        if i > 30:
            break
        if uses_hc:
            for j, lin_est_i in enumerate(lin_est):
                # Plot NL est for each HC iteration.
                ax.plot(
                    range(
                        tot_nl_iterations_fine, tot_nl_iterations_fine + len(lin_est_i)
                    ),
                    lin_est_i,
                    "v-",
                    color="orange",
                    markersize=kwargs.get("marker_size", 4),
                    fillstyle="none",
                    markerfacecolor="orange",
                    label=r"$\eta_\mathrm{lin}$" if i == 0 else "",
                )
                # Plot betas on the second y-axis
                ax2.plot(
                    range(
                        tot_nl_iterations_fine, tot_nl_iterations_fine + len(lin_est_i)
                    ),
                    [stats.lambdas[i][j]]
                    * len(lin_est_i),  # Same lambda for each HC step.
                    linestyle="-",
                    color="black",
                    marker="s",
                    markersize=kwargs.get("marker_size", 4),
                    alpha=0.7,
                    label=r"$\beta$" if i == 0 else "",
                )
                tot_nl_iterations_fine += len(lin_est_i)

            hc_est_flat = _flatten_nested_list(stats.hc_estimator[i])
            spat_est_flat = _flatten_nested_list(spat_est)
            temp_est_flat = _flatten_nested_list(temp_est)
        else:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(lin_est)),
                lin_est,
                "o-",
                color="orange",
                markersize=kwargs.get("marker_size", 4),
                fillstyle="none",
                markerfacecolor="orange",
                label=r"$\eta_\mathrm{lin}$" if i == 0 else "",
            )

            spat_est_flat = spat_est
            temp_est_flat = temp_est

        if uses_hc:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(hc_est_flat)),
                hc_est_flat,
                "^-",
                markersize=kwargs.get("marker_size", 4),
                color="blue",
                fillstyle="none",
                markerfacecolor="blue",
                label=r"$\eta_\mathrm{HC}$" if i == 0 else "",
            )

        # Plot spatial and temporal estimators for each time step.
        if combine_disc_est:
            # Combine spatial and temporal estimators.
            disc_est_flat = np.array(spat_est_flat) + np.array(temp_est_flat)
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(disc_est_flat)),
                disc_est_flat,
                "o-",
                color="red",
                markersize=kwargs.get("marker_size", 4),
                fillstyle="none",
                markerfacecolor="red",
                label=r"$\eta_\mathrm{disc}$" if i == 0 else "",
            )
        else:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(spat_est_flat)),
                spat_est_flat,
                "bo-",
                markersize=kwargs.get("marker_size", 4),
                fillstyle="none",
                markerfacecolor="blue",
                label=r"$\eta_{sp}$" if i == 0 else "",
            )
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(temp_est_flat)),
                temp_est_flat,
                "rv-",
                markersize=kwargs.get("marker_size", 4),
                fillstyle="none",
                label=r"$\eta_{\mathrm{temp}}$" if i == 0 else "",
            )

        # Update number of nl iterations.
        tot_nl_iterations += len(spat_est_flat)
        # Plot a dotted grey vertical line to separate time steps.

        if i < len(stats.discrete_times) - 1:
            ax.axvline(
                x=tot_nl_iterations - 0.5,
                color="grey",
                linestyle="--",
                linewidth=2.0,
                alpha=0.5,
            )

    # Format axes, labels, and title.
    ax.set_yscale("log")

    # On the x-axis use integer ticks only with sensible density.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10, prune=None))
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_xlabel("Cumulative nonlinear iteration", fontsize=14, fontweight="bold")
    ax.set_ylabel("Error estimators", fontsize=14, fontweight="bold")

    if uses_hc:
        ax2.set_yscale("log")
        ax2.set_ylim(
            0.8 * min(min(hc_step) for hc_step in stats.lambdas), 1.1
        )  # Set y-limits for better visibility of beta values.])
        ax2.tick_params(axis="y", labelsize=12, labelcolor="black")
        ax2.set_ylabel(
            r"$\beta$ values",
            fontsize=14,
            fontweight="bold",
            color="black",
        )
        ax2.grid(False)

    if title is None:
        title = "Error Estimators Evolution"
    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add legend
    # Get handles and labels from primary axis
    handles, labels = ax.get_legend_handles_labels()

    # If using HC estimator, also get handles and labels from secondary axis
    if uses_hc:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles.extend(handles2)
        labels.extend(labels2)

    # Remove duplicates while preserving order
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc=kwargs.get("legend_loc", "lower left"),
        ncol=2,
        prop={"size": 14, "weight": "bold"},
    )

    fig.tight_layout()
    return fig


def plot_convergence(
    stats: list[SolverStats],
    parameters: list[float],
    parameter_name: str,
) -> plt.Figure:
    """Plot spatial or temporal error estimator convergence.

    Returns:
        A matplotlib figure with the plotted estimators.

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    final_estimators = []

    for stat in stats:
        uses_hc: bool = len(stat.hc_estimator) > 0

        if parameter_name == "num_grid_cells":
            if uses_hc:
                final_estimators.append(stat.spat_estimator[-1][-1][-1])
            else:
                final_estimators.append(stat.spat_estimator[-1][-1])

        elif parameter_name == "time_step_size":
            if uses_hc:
                final_estimators.append(stat.temp_estimator[-1][-1][-1])
            else:
                final_estimators.append(stat.temp_estimator[-1][-1])

    # Sort by params to make the plot and reference lines clean.
    params_np = np.asarray(parameters, dtype=float)
    est_np = np.asarray(final_estimators, dtype=float)

    ax.loglog(params_np, est_np, "o-", markersize=4, linewidth=2, fillstyle="none")

    # # Format axes, labels, and title.

    # Major ticks at decades.
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_locator(LogLocator(base=10))

    ax.xaxis.set_major_formatter(LogFormatter(base=10))
    ax.yaxis.set_major_formatter(LogFormatter(base=10))

    # Minor ticks at all log subdivisions (2–9)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=list(range(2, 10))))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=list(range(2, 10))))

    def selective_minor_formatter(val, pos):
        if val <= 0:
            return ""
        exp = np.log10(val)
        k = np.floor(exp)
        mantissa = val / 10**k
        if np.isclose(mantissa, 2.0) or np.isclose(mantissa, 5.0):
            return f"{val:g}"
        return ""

    ax.xaxis.set_minor_formatter(FuncFormatter(selective_minor_formatter))
    ax.yaxis.set_minor_formatter(FuncFormatter(selective_minor_formatter))

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    if parameter_name == "num_grid_cells":
        x_label = "Number of grid cells"
        y_label = r"$\eta_{\mathrm{spat}}$"
        title = "Convergence of Spatial Error Estimator"

        inset_loc = "lower left"
        slope = -1.0  # Spatial estimator decreases with higher cell count.

    elif parameter_name == "time_step_size":
        x_label = "Time step size ($s$)"
        y_label = r"$\eta_{\mathrm{temp}}$"
        title = "Convergence of Temporal Error Estimator"

        inset_loc = "lower right"
        slope = 1.0  # Temporal estimator increases with larger time step.

    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    ax_ins = inset_axes(ax, width="30%", height="30%", loc=inset_loc, borderpad=2)

    # Ensure slopes from the original axes are preserved in the inset.
    ax_ins.set_aspect("equal", adjustable="box")

    x_ref = np.array([1, 10])
    y_linear = x_ref ** (slope * 1)  # Slope of -+1 (Linear)
    y_quadratic = x_ref ** (slope * 2)  # Slope of -+2 (Quadratic)

    ax_ins.loglog(x_ref, y_linear, color="gray", linestyle="--", lw=2)
    ax_ins.loglog(x_ref, y_quadratic, color="gray", linestyle=":", lw=2)

    # Clean up the inset: remove ticks to keep it purely as a visual slope guide.
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    ax_ins.set_xticklabels([])
    ax_ins.set_yticklabels([])

    # Get location for text on reference lines.
    x_mid = np.sqrt(x_ref[0] * x_ref[1])
    y_lin_mid = x_mid ** (slope * 1)
    y_quad_mid = x_mid ** (slope * 2)

    ax_ins.text(
        x_mid, y_lin_mid, r"$\mathcal{O}(x^1)$", fontsize=14, ha="center", va="bottom"
    )
    ax_ins.text(
        x_mid, y_quad_mid, r"$\mathcal{O}(x^2)$", fontsize=14, ha="center", va="bottom"
    )

    fig.tight_layout()
    return fig


def read_comparison_stats(
    config: SimulationConfig,
    solver_stats: SolverStats,
    relative_comparison: bool = True,
) -> ComparisonStats:
    """Read statistics from a comparison simulation.

    Note: Comparison statistics only make sense if the simulation converged in a single
        time step, i.e., if the simulation and the reference solution solved the same
        problem.

    Note: If no comparison statistics are found, an empty ComparisonStats object is
        returned.

    Parameters:
        config: The simulation configuration.
        solver_stats: The solver statistics of the simulation.
        relative_comparison: Whether to read relative comparison statistics (True) or
            absolute comparison statistics (False).

    Returns:
        A ComparisonStats object containing the comparison statistics.

    """
    filename = (
        "relative_comparison_stats.json"
        if relative_comparison
        else "absolute_comparison_stats.json"
    )

    if solver_stats.converged and len(solver_stats.discrete_times) == 1:
        try:
            with (config.folder_name() / filename).open() as f:
                data: dict[str, Any] = json.load(f)
                return ComparisonStats(**data)
        except FileNotFoundError:
            return ComparisonStats()
    else:
        return ComparisonStats()


def tabulate_comparison_stats(
    data: dict[tuple[str, str], ComparisonStats], varying_param_name: str, **kwargs
) -> list[str]:
    """Tabulate comparison statistics for different solvers and parameter values from one study.

    Parameters:
        data: Dictionary mapping solver specs and varying parameter values (as a tuple
            of strings) to comparison statistics.
        varying_param_name: Name of the parameter that varies between the
            configurations. This will be the title of the x-axis.

    """
    # Same construction as in plot_nl_iterations, but we only tabulate the data and do
    # not reshape the columns.
    data_dtype = np.dtype(
        [
            # Make the strings long enough to avoid any issues.
            ("solver_specs", "U200"),
            ("parameter_value", "U100"),
            ("pressure_diff_norm", "float32"),
            ("saturation_diff_norm", "float32"),
            ("total_flux_diff_norm", "float32"),
            ("wetting_flux_diff_norm", "float32"),
            ("flow_residual_norm", "float32"),
            ("transport_residual_norm", "float32"),
        ]
    )
    data_as_array = np.zeros(len(data), dtype=data_dtype)

    for i, ((solver_specs, parameter_value), stats) in enumerate(data.items()):
        data_as_array[i]["solver_specs"] = solver_specs
        data_as_array[i]["parameter_value"] = parameter_value

        # Now, read the stats of the case.
        data_as_array[i]["pressure_diff_norm"] = stats.pressure_diff_norm
        data_as_array[i]["saturation_diff_norm"] = stats.saturation_diff_norm
        data_as_array[i]["total_flux_diff_norm"] = stats.total_flux_diff_norm
        data_as_array[i]["wetting_flux_diff_norm"] = stats.wetting_flux_diff_norm
        data_as_array[i]["flow_residual_norm"] = stats.flow_residual_norm
        data_as_array[i]["transport_residual_norm"] = stats.transport_residual_norm

    # Sort indices first by parameter_value, then by solver_specs.
    idx = np.lexsort(
        (
            data_as_array["solver_specs"],
            data_as_array["parameter_value"],
        )
    )
    # Apply sorting to data.
    data_as_array = data_as_array[idx]

    # Tabulate the data.
    table_lines = []
    header = (
        "Solver Name",
        "hc_tol",
        "nl_tol",
        varying_param_name,
        "pressure_diff_norm",
        "saturation_diff_norm",
        "total_flux_diff_norm",
        "wetting_flux_diff_norm",
        "flow_residual_norm",
        "transport_residual_norm",
    )
    table_lines.append(header)

    for row in data_as_array:
        solver_specs_list: list[str] = row["solver_specs"].split("_")
        solver_name_postfix = solver_specs_list[1] if len(solver_specs_list) > 3 else ""
        solver_name = solver_specs_list[0] + solver_name_postfix
        hc_tol, nl_tol = solver_specs_list[-2], solver_specs_list[-1]

        table_lines.append(
            (
                solver_name,
                hc_tol,
                nl_tol,
                row["parameter_value"],
                row["pressure_diff_norm"],
                row["saturation_diff_norm"],
                row["total_flux_diff_norm"],
                row["wetting_flux_diff_norm"],
                row["flow_residual_norm"],
                row["transport_residual_norm"],
            )
        )

    return table_lines
