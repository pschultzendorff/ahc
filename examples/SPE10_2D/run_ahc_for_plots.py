r"""Run the simulation with small uniform time steps for fancy plotting.


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

import json
import logging
import os
import pathlib
import shutil
import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
import seaborn as sns
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

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

sns.set_theme("paper")
sns.set_style("whitegrid")

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

    Returns:
        A tuple ``(model_class, solver_params)``, where ``model_class`` is the model
        class with the correct adaptive solver and ``solver_params`` is a dictionary
        containing solver parameters.

    """

    solver_params: dict[str, Any] = {
        # Homotopy Continuation (HC) parameters:
        "nonlinear_solver_statistics": SolverStatisticsHC,
        "nonlinear_solver": HCSolver,
        "hc_max_iterations": 50,
        "hc_constant_decay": False,
        "hc_lambda_decay": 0.9,
        "hc_decay_min_max": (0.1, 0.95),
        "nl_iter_optimal_range": (4, 7),
        "nl_iter_relax_factors": (0.7, 1.3),
        "hc_decay_recomp_max": 5,
        # Adaptivity:
        "hc_adaptive": True,
        "hc_error_ratio": adaptive_error_ratio,  # adaptive error rate for homotopy
        "nl_error_ratio": 0.1,
        "hc_nl_convergence_tol": 1e5,
        # Nonlinear solver parameters:
        "nl_convergence_tol": 1e-5,
        "nl_divergence_tol": 1e30,
        "max_iterations": 50,
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

    model = model_class(params)
    pp.run_time_dependent_model(model=model, params=params)


# endregion


# region UTILS
@dataclass
class SimulationStatistics:
    # Type of list element vary depending on the solver type. We do not specify this
    # here.
    time_steps: list = field(default_factory=list)
    timestep_nl_iters: list = field(default_factory=list)
    spat_estimator: list = field(default_factory=list)
    temp_estimator: list = field(default_factory=list)
    hc_estimator: list = field(default_factory=list)
    lin_estimator: list = field(default_factory=list)
    lambdas: list = field(default_factory=list)


def flatten(xx: list[list]) -> list:
    return [x for sublist in xx for x in sublist]


def read_data(config: SimulationConfig) -> SimulationStatistics:
    # Check if the solver failed at some time step and return empty statistics.
    if "failure" in [f.stem for f in config.folder_name.iterdir()]:
        # raise ValueError("Simulation did not complete.")
        return SimulationStatistics()

    # If not we can read the data.
    with open(config.folder_name / "solver_statistics.json", "r") as f:
        data: dict[str, Any] = json.load(f)

    statistics = SimulationStatistics()

    for time_step in list(data.values())[1:]:
        # Treat different solvers.
        if config.solver_name.startswith("AHC") or config.solver_name.startswith("HC"):
            # Create lists for the outer loop.
            num_nl_iterations = []
            spat_estimator = []
            temp_estimator = []
            hc_estimator = []
            lin_estimator = []
            # The last 5 entries are not hc steps.
            for hc_step in list(time_step.values())[:-5]:
                # Read iterations per time step.
                num_nl_iterations.append(hc_step["num_iteration"])
                spat_estimator.append(hc_step["spatial_error_estimates"])
                temp_estimator.append(hc_step["temporal_error_estimates"])
                hc_estimator.append(hc_step["hc_error_estimates"])
                lin_estimator.append(hc_step["linearization_error_estimates"])
            statistics.hc_estimator.append(hc_estimator)
            statistics.lambdas.append(time_step["hc_lambdas"])
        elif config.solver_name.startswith("Newton"):
            if config.solver_name == "NewtonAppleyard":
                pass
            # Read iterations per time step.
            num_nl_iterations = time_step["num_iteration"]
            spat_estimator = time_step["spatial_est"]
            temp_estimator = time_step["temp_est"]
            lin_estimator = time_step["lin_est"]

        # Append data to the statistics object.
        statistics.time_steps.append(time_step["current time"])
        statistics.timestep_nl_iters.append(num_nl_iterations)
        statistics.spat_estimator.append(spat_estimator)
        statistics.temp_estimator.append(temp_estimator)
        statistics.lin_estimator.append(lin_estimator)

    # Treat failed time steps.
    time_steps_copy: list[float] = statistics.time_steps.copy()
    for i, (ts, next_ts) in enumerate(
        zip(statistics.time_steps[:-1], statistics.time_steps[1:])
    ):
        # TODO Fix this!
        # QUESTION What is wrong?
        if ts > next_ts:
            time_steps_copy[i] = next_ts
    statistics.time_steps = time_steps_copy

    return statistics


def plot_estimators(
    statistics: SimulationStatistics,
    title: str | None = None,
    combine_disc_est: bool = False,
) -> plt.Figure:
    """Create a plot showing the evolution of different error estimators over time.

    Returns:
        A matplotlib figure with the plotted estimators.

    """
    # Check if HC estimator is present.
    uses_hc: bool = len(statistics.hc_estimator) > 0

    fig, ax = plt.subplots(figsize=(8, 6))

    if uses_hc:
        # Create a secondary y-axis for lambdas.
        ax2 = ax.twinx()

    tot_nl_iterations: int = 0
    tot_nl_iterations_fine: int = 0
    # Plot spatial estimator
    for i, (time, spat_est, temp_est, lin_est) in enumerate(
        zip(
            statistics.time_steps,
            statistics.spat_estimator,
            statistics.temp_estimator,
            statistics.lin_estimator,
        )
    ):
        if uses_hc:
            for j, lin_est_i in enumerate(lin_est):
                # Plot NL est for each HC iteration.
                ax.plot(
                    range(
                        tot_nl_iterations_fine, tot_nl_iterations_fine + len(lin_est_i)
                    ),
                    lin_est_i,
                    "v-",
                    color="green",
                    markersize=4,
                    fillstyle="none",
                    markerfacecolor="green",
                    label=r"$\eta_\mathrm{lin}$" if i == 0 else "",
                )
                # Plot betas on the second y-axis
                ax2.plot(
                    range(
                        tot_nl_iterations_fine, tot_nl_iterations_fine + len(lin_est_i)
                    ),
                    [statistics.lambdas[i][j]]
                    * len(lin_est_i),  # Same lambda for each HC step.
                    linestyle="-",
                    color="black",
                    marker="s",
                    markersize=4,
                    alpha=0.7,
                    label=r"$\beta$" if i == 0 else "",
                )
                tot_nl_iterations_fine += len(lin_est_i)

            hc_est_flat = flatten(statistics.hc_estimator[i])
            spat_est_flat = flatten(spat_est)
            temp_est_flat = flatten(temp_est)
        else:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(lin_est)),
                lin_est,
                "o-",
                color="green",
                markersize=4,
                fillstyle="none",
                markerfacecolor="green",
                label=r"$\eta_\mathrm{lin}$" if i == 0 else "",
            )

            spat_est_flat = spat_est
            temp_est_flat = temp_est

        if uses_hc:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(hc_est_flat)),
                hc_est_flat,
                "^-",
                markersize=4,
                color="orange",
                fillstyle="none",
                markerfacecolor="orange",
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
                color="goldenrod",
                markersize=4,
                fillstyle="none",
                markerfacecolor="goldenrod",
                label=r"$\eta_\mathrm{discr}$" if i == 0 else "",
            )
        else:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(spat_est_flat)),
                spat_est_flat,
                "bo-",
                markersize=4,
                fillstyle="none",
                markerfacecolor="blue",
                label=r"$\eta_{sp}$" if i == 0 else "",
            )
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(temp_est_flat)),
                temp_est_flat,
                "rv-",
                markersize=4,
                fillstyle="none",
                label=r"$\eta_{\mathrm{temp}}$" if i == 0 else "",
            )

        # Update number of nl iterations.
        tot_nl_iterations += len(spat_est_flat)
        # Plot a dotted grey vertical line to separate time steps.

        if i < len(statistics.time_steps) - 1:
            ax.axvline(
                x=tot_nl_iterations - 0.5,
                color="grey",
                linestyle="--",
                linewidth=2.0,
                alpha=0.5,
            )

    # Set y scale to log
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(ax.get_ylim())

    # Add labels and title
    # Set ticks to integers only
    ax.set_xticks(range(0, tot_nl_iterations + 1, max(1, tot_nl_iterations // 10)))
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel(
        "Nonlinear iteration",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Error estimate",
        fontsize=14,
        fontweight="bold",
    )

    if uses_hc:
        ax2.tick_params(axis="y", labelcolor="black")
        ax2.set_ylabel(r"$\beta$ values", color="black", fontsize=14, fontweight="bold")
        # Disable grid for secondary axis.
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
        loc="lower left",
        ncol=2,
        prop={"size": 14, "weight": "bold"},
    )

    fig.tight_layout()
    return fig


# endregion


# region RUN

if __name__ == "__main__":
    rp_model_1 = {"model": "linear", "limit": False}
    cp_model_1 = {
        "model": None,
    }
    rp_model_2 = {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 1.0,
        "eta": 2.0,
    }  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1

    cp_model_2: dict[str, Any] = {
        "model": "Brooks-Corey",
        "n_b": 2.0,
        "entry_pressure": 30 * pp.PASCAL,
    }
    config = SimulationConfig(
        file_name="ahc_for_plots",
        folder_name=dirname / "ahc_for_plots",
        solver_name="AHC",
        adaptive_error_ratio=0.00000001,
        cell_size=600 * FEET / 30,
        init_s=0.3,
        rp_model_1=rp_model_1,
        rp_model_2=rp_model_2,
        cp_model_1=cp_model_1,
        cp_model_2=cp_model_2,
    )

    # run_simulation(config)

    statistics = read_data(config)
    fig = plot_estimators(statistics, combine_disc_est=True)
    fig.savefig(dirname / "convergence_estimators.png")

# endregion
