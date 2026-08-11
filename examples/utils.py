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
    cell_size: float = 600 * FEET / 30  # Default cell size.
    """Cell size parameter to be passed to gmsh for the SPE10 model."""
    spe10_layer: int = 0
    """SPE10 layer number. Possible values are 0 to 84. Default is 0."""
    spe10_case: str = "five_spot"
    """SPE10 case name. Possible values are "five_spot" and "gravity_separation".
    Default is "five_spot". 

    """
    spe10_water_density: float = water["density"]  # kg/m^3
    r"""Water density for the SPE10 model. Default is :math:`998.2\,\mathrm{kg/m^3}`."""

    # SPE11 specific model parameters.
    refinement_factor: float = 1.0
    """Grid refinement factor for the SPE11 model. Default is 1.0, which means no
    refinement. Values larger than 1.0 mean a coarser grid and values smaller than 1.0
    mean a finer grid. 

    See https://github.com/Simulation-Benchmarks/11thSPE-CSP/tree/main/geometries.

    """
    spe11_entry_pressure: float = 30 * pp.PASCAL
    r"""Homogenized entry pressure for the SPE11 model. Default is
    :math:`30\,\mathrm{Pa}`."""

    def folder_name(self) -> pathlib.Path:
        """Create a folder path from metadata and solver parameters.

        Returns:
            name: Path to the folder for the simulation results.

        """
        solver_name_with_params = (
            f"{self.solver_name}_{self.solver_name_postfix}_{self.hc_tol}_{self.nl_tol}"
        )
        return self.results_dir / solver_name_with_params / self.regime / self.study


def setup_porepy_params(
    config: SimulationConfig, **kwargs
) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Setup solver and time manager parameters in PorePy format.

    Note: Many of the solver and time stepping parameters for all benchmarks are
    hardcoded in this function.

    Parameters:
        config: The simulation configuration specifying nonlinear solver type and
            tolerances.
        kwargs: Additional keyword arguments, e.g., for extrapolation time error
            estimator after time step cutting.

    Returns:
        A tuple ``(solver_params, time_manager_params)``, where ``solver_params``
        contains parameters for the nonlinear solver, and ``time_manager_params``
        contains parameters for the time manager.

    """
    solver_name = config.solver_name
    hc_tol = config.hc_tol
    nl_tol = config.nl_tol
    logger.info(
        f"solver: {solver_name}, HC tolerance: {hc_tol:.2f}, NL tolerance: {nl_tol:.2f}."
        " Generating solver and time manager parameters."
    )

    match solver_name:
        case "HC":
            solver_params = {
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
                "nl_divergence_tol": 1e30,
                "max_iterations": 30,
                "nl_appleyard_chopping": False,
            }

        case "AHC":
            solver_params = {
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
                "adaptive_threshold_rel": 1e2,
                "extrapolate_temp_estimator_after_cutting": kwargs.get(
                    "extrapolate_temp_estimator_after_cutting", True
                ),
                # Non-adaptive stopping and other Newton solver parameters:
                "nl_convergence_tol_abs": 1e-5,
                "nl_convergence_tol_rel": 1e-5,
                "nl_divergence_tol": 1e30,
                "max_iterations": 30,
                "nl_appleyard_chopping": False,
            }

        case "Newton":
            solver_params = {
                # Metadata:
                "nonlinear_solver_statistics": SolverStatisticsANewton,
                "nonlinear_solver": ModifiedNewtonSolver,
                # Adaptive stopping criteria for Newton:
                "nl_adaptive": True,
                "nl_error_ratio": nl_tol,  # Adaptive error ratio for Newton.
                "adaptive_threshold_rel": 1e2,
                "extrapolate_temp_estimator_after_cutting": kwargs.get(
                    "extrapolate_temp_estimator_after_cutting", True
                ),
                # Non-adaptive stopping parameters and other solver parameters:
                "nl_convergence_tol_abs": 1e-5,
                "nl_convergence_tol_rel": 1e-5,
                "nl_divergence_tol": 1e30,
                "nl_appleyard_chopping": False,
                "max_iterations": 30,
            }
        case "NewtonAppleyard":
            solver_params = {
                # Metadata:
                "nonlinear_solver_statistics": SolverStatisticsANewton,
                "nonlinear_solver": ModifiedNewtonSolver,
                # Adaptive stopping criteria for Newton:
                "nl_adaptive": True,
                "nl_error_ratio": nl_tol,  # Adaptive error ratio for Newton.
                "adaptive_threshold_rel": 1e2,
                "extrapolate_temp_estimator_after_cutting": kwargs.get(
                    "extrapolate_temp_estimator_after_cutting", True
                ),
                # Non-adaptive stopping parameters and other solver parameters:
                "nl_convergence_tol_abs": 1e-5,
                "nl_convergence_tol_rel": 1e-5,
                "nl_divergence_tol": 1e30,
                "nl_appleyard_chopping": True,
                "max_iterations": 30,
            }
        case _:
            raise ValueError(f"Unknown solver: {solver_name}")

    if solver_name.endswith("HC"):
        # Update adaptive time stepping parameters for HC.
        time_manager_params = {
            "iter_optimal_range": (
                15,
                30,
            ),  # Default value is (4, 7), which is used for the Newton solvers.
            "iter_relax_factors": (0.7, 1.3),
            "iter_max": 50,  # This should be the same as "hc_max_iterations", which the
            # TimeManager does not know about.
        }
    else:
        # Update adaptive time stepping parameters for Newton.
        time_manager_params = {
            "iter_optimal_range": (
                8,
                20,
            ),  # Default value is (4, 7).
            "iter_relax_factors": (0.7, 1.3),  # Default value.
            "iter_max": 30,  # This should be the same as "max_iterations", which the
            # TimeManager does not know about.
        }

    return solver_params, time_manager_params


def clean_up_after_simulation(config: SimulationConfig) -> None:
    """Remove large files that are not needed for the analysis.

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
class SimulationStatistics:
    # Type of list element vary depending on the solver type. We do not specify this
    # here.
    discrete_times: list = field(default_factory=list)
    timestep_nl_iters: list = field(default_factory=list)

    spat_estimator: list = field(default_factory=list)
    temp_estimator: list = field(default_factory=list)
    hc_estimator: list = field(default_factory=list)
    lin_estimator: list = field(default_factory=list)
    energy_norm: list = field(default_factory=list)

    lambdas: list = field(default_factory=list)
    converged: bool = False
    final_time: float = 0.0

    num_grid_cells: int = 1


def flatten(xx: list[list]) -> list:
    return [x for sublist in xx for x in sublist]


# TODO Unify the naming of the error estimators for Newton and HC.
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
    spat_estimator = [s["spatial_error_estimates"] for s in hc_steps]
    temp_estimator = [s["temporal_error_estimates"] for s in hc_steps]
    hc_estimator = [s["hc_error_estimates"] for s in hc_steps]
    lin_estimator = [s["linearization_error_estimates"] for s in hc_steps]
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
) -> tuple[int, list[float], list[float], list[float]]:
    """Helper function to parse Newton iterations within a time step and return the
    relevant statistics.

    """
    return (
        time_step["num_iteration"],
        time_step["spatial_est"],
        time_step["temp_est"],
        time_step["lin_est"],
        # NOTE Forgot to implement saving save global energy norm for adaptive
        # Newton, hence this will be empty.
        # time_step["global_energy_norm"]
    )


def read_data(
    config: SimulationConfig,
    expected_final_time: float,
) -> SimulationStatistics:
    with (config.folder_name() / "solver_statistics.json").open() as f:
        data: dict[str, Any] = json.load(f)

    time_steps = list(data.values())
    stats = SimulationStatistics()

    # Read number of grid cells, before possibly returning empty statistics (when
    # failed).
    try:
        stats.num_grid_cells = int(
            (config.folder_name() / "num_grid_cells.txt").read_text().strip()
        )
    except FileNotFoundError:
        return stats

    # Check if the simulation reached the final time.
    stats.final_time = time_steps[-1]["current time"]
    if not np.isclose(stats.final_time, expected_final_time):
        stats.converged = False
        return stats

    for time_step in time_steps[1:]:
        if config.solver_name.endswith("HC"):
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
            num_nl_iterations, spat_estimator, temp_estimator, lin_estimator = (
                _parse_newton_steps(time_step)
            )
        else:
            raise ValueError(f"Unknown solver: {config.solver_name}")

        # Append data to the statistics object.
        stats.discrete_times.append(time_step["current time"])
        stats.timestep_nl_iters.append(num_nl_iterations)
        stats.spat_estimator.append(spat_estimator)
        stats.temp_estimator.append(temp_estimator)
        stats.lin_estimator.append(lin_estimator)

    # Reset discrete time values at failed time steps. Failed time steps are recognized
    # by the fact that the time step was retried with a shorter time step size.
    for i in range(len(stats.discrete_times) - 2, -1, -1):
        stats.discrete_times[i] = min(
            stats.discrete_times[i], stats.discrete_times[i + 1]
        )

    return stats


def calc_relative_error(stats: SimulationStatistics) -> dict[str, float]:
    """Calculate relative errors at the end of the simulation."""
    # Determine the solver by number of nested loops.
    solver_type = "Newton" if isinstance(stats.energy_norm[-1][-1], float) else "HC"

    energy_norm = (
        stats.energy_norm[-1][-1]
        if solver_type == "Newton"
        else stats.energy_norm[-1][-1][-1]
    )
    result = {}
    for error_name in ["total", "lin", "spat", "temp", "hc"]:
        if error_name == "total":
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
        elif error_name == "hc":
            if solver_type == "HC":
                result[error_name] = stats.hc_estimator[-1][-1][-1] / energy_norm
        else:
            if solver_type == "Newton":
                result[error_name] = (
                    getattr(stats, error_name + "_estimator")[-1][-1] / energy_norm
                )
            else:
                result[error_name] = (
                    getattr(stats, error_name + "_estimator")[-1][-1][-1] / energy_norm
                )

    return result


def plot_nl_iterations(
    data: dict[str, SimulationStatistics],
    varying_param_name: str,
    title: str | None = None,
    **kwargs,
):
    """Create a heatmap showing nonlinear iterations for different solvers and parameter values.

    Parameters:
        data: Dictionary mapping simulation configurations to simulation statistics.
        varying_param_name: Name of the parameter that varies between the configurations.
        title: Optional title for the plot.
    """
    # Extract solvers and parameter values from case names
    cases = list(data.keys())
    solvers = sorted(set("\n".join(case.split("_")[:2]) for case in cases))
    solvers = []
    for case in cases:
        solver_name, adaptive_error_ratio_str, varying_param = case.split("_")
        if solver_name == "HC":
            solvers.append(solver_name)
        elif solver_name.startswith("AHC"):
            solvers.append(
                f"{solver_name}\n"
                + rf"$\gamma_\mathrm{{HC}} = {adaptive_error_ratio_str}$"
                + "\n"
                # Hardcode adaptive stopping criterion for corrector loop.
                + r"$\gamma_\mathrm{lin} = 0.1$"
            )
        elif solver_name.startswith("Newton"):
            solvers.append(
                f"{solver_name}\n"
                + rf"$\gamma_\mathrm{{lin}} = {adaptive_error_ratio_str}$"
            )
    solvers = sorted(set(solvers))

    x_ticks = sorted(
        set(" ".join(case.split("_")[2:]) for case in cases),
        key=lambda x: float(x) if x.replace(".", "", 1).isdigit() else x,
    )

    # Transform data to arrays for nl iterations, annotations, and convergence status.
    nl_iterations = np.empty((len(solvers), len(x_ticks)))
    annotations = np.empty((len(solvers), len(x_ticks)), dtype="<U25")
    converged = np.empty((len(solvers), len(x_ticks)), dtype=bool)
    final_times = np.empty((len(solvers), len(x_ticks)))

    for case, stat in data.items():
        solver_name, adaptive_error_ratio_str, varying_param = case.split("_")

        adaptive_error_ratio = float(adaptive_error_ratio_str)

        if solver_name == "HC":
            # Standard HC solver does not have an adaptive error ratio.
            i = solvers.index(solver_name)
        elif solver_name == "AHC":
            i = solvers.index(
                f"{solver_name}\n"
                + rf"$\gamma_\mathrm{{HC}} = {adaptive_error_ratio}$"
                + "\n"
                # Hardcode adaptive stopping criterion for corrector loop.
                + r"$\gamma_\mathrm{lin} = 0.1$"
            )
        elif solver_name.startswith("Newton"):
            i = solvers.index(
                f"{solver_name}\n"
                + rf"$\gamma_\mathrm{{lin}} = {adaptive_error_ratio}$"
            )

        j = x_ticks.index(varying_param)

        # Do not read statistics if the solver did not converge.
        converged[i, j] = stat.converged
        if not stat.converged:
            final_times[i, j] = stat.final_time
            continue

        tot_nl_iters = (
            sum(stat.timestep_nl_iters)
            if solver_name.startswith("Newton")
            else sum(flatten(stat.timestep_nl_iters))
        )
        nl_iterations[i, j] = tot_nl_iters

        # For HC and AHC, show nl iters, hc iters, final beta, and time steps.
        if solver_name in ["HC", "AHC"]:
            tot_hc_iters = len(flatten(stat.timestep_nl_iters))
            final_lambda = stat.lambdas[-1][-1]
            annotations[i, j] = (
                f"{tot_nl_iters}/{tot_hc_iters}/{final_lambda:.4f}\n"
                + f"({len(stat.discrete_times)})"
            )
        # For Newton, show only nl iters and time steps.
        else:
            annotations[i, j] = f"{tot_nl_iters}\n({len(stat.discrete_times)})"

    # Create heatmap figure.
    if kwargs.get("extended_figure_height", False):
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Failed time steps are marked in red.
    cmap = matplotlib.colormaps["Blues"]
    cmap.set_bad(color="red")

    sns.heatmap(
        nl_iterations,
        mask=np.logical_not(converged),
        annot=annotations,
        fmt="s",
        cmap=cmap,
        cbar=True,
        cbar_kws={"label": "Number of cumulative nonlinear iterations"},
        xticklabels=x_ticks,
        yticklabels=solvers,
        linewidths=0.8,
        ax=ax,
    )

    # Annotate failed simulations
    for i in range(converged.shape[0]):
        for j in range(converged.shape[1]):
            if not converged[i, j]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    r"Reached min. $\Delta t$"
                    + "\n"
                    # i and j are switched in the flattened data.
                    + f"at t={final_times[i, j] / 86400:.1f} d",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    # Set labels and title
    ax.set_xlabel(varying_param_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Solver & adaptive error ratio", fontsize=12, fontweight="bold")
    ax.set_title(
        title
        or r"# cumulative NL iters/#HC iters/final $\beta$" + "\n" + "(#time steps)\n",
        # + f"by solver and {varying_param_name}",
        fontsize=14,
        fontweight="bold",
    )

    if kwargs.get("tight_layout", True):
        fig.tight_layout()
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
    # Use tight_layout with padding to prevent clipping of rotated labels.
    try:
        fig.tight_layout(pad=1.5)
    except Exception:
        pass

    return fig


def plot_estimators(
    stats: SimulationStatistics,
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

            hc_est_flat = flatten(stats.hc_estimator[i])
            spat_est_flat = flatten(spat_est)
            temp_est_flat = flatten(temp_est)
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
    stats: list[SimulationStatistics],
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
    elif parameter_name == "time_step_size":
        x_label = "Time step size ($s$)"
        y_label = r"$\eta_{\mathrm{temp}}$"
        title = "Convergence of Temporal Error Estimator"

    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add linear and quadratic reference lines on log-log scale.
    if parameter_name == "num_grid_cells":
        inset_loc = "lower left"
        slope = -1.0  # Spatial estimator decreases with higher cell count.
    elif parameter_name == "time_step_size":
        inset_loc = "lower right"
        slope = 1.0  # Temporal estimator increases with larger time step.

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
