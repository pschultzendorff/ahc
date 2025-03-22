import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from ahc import SimulationConfig, generate_configs
from matplotlib.ticker import ScalarFormatter

dirname: pathlib.Path = pathlib.Path(__file__).parent


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


def flatten(xx: list[list]) -> list:
    return [x for sublist in xx for x in sublist]


def read_data(config: SimulationConfig) -> SimulationStatistics:
    # Check if the solver failed at some point.
    if "failure" in [f.stem for f in config.folder_name.iterdir()]:
        raise ValueError("Simulation did not complete.")

    # If not we can read the data.
    with open(config.folder_name / "solver_statistics.json", "r") as f:
        data: dict[str, Any] = json.load(f)

    statistics = SimulationStatistics()

    for time_step in list(data.values())[1:]:
        # Treat different solvers.
        if config.solver_name.startswith("AHC"):
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
        elif config.solver_name.startswith("Newton"):
            if config.solver_name == "NewtonAppleyard":
                pass
            # Read iterations per time step.
            num_nl_iterations = time_step["num_iteration"]
            spat_estimator = time_step["spatial_est"]
            temp_estimator = time_step["temp_est"]
            lin_estimator = time_step["linearization_est"]

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
        if ts > next_ts:
            time_steps_copy[i] = next_ts
    statistics.time_steps = time_steps_copy

    return statistics


def plot_nl_iterations(
    data: dict[str, SimulationStatistics],
    varying_param_name: str,
    title: str | None = None,
):
    """Create a heatmap showing nonlinear iterations for different solvers and parameter
    values.

    Args:
        data: Dictionary mapping simulation configurations to simulation statistics.
        varying_param: Name of the parameter that varies between the configurations.

    """
    # Get unique solvers and parameter values.
    cases: list[str] = list(data.keys())
    solvers = list(set("_".join(case.split("_")[:2]) for case in cases))
    x_ticks = list(set("_".join(case.split("_")[2:]) for case in cases))
    solvers.sort()
    x_ticks.sort()

    # Create a matrix for the heatmap.
    matrix = np.empty((len(solvers), len(x_ticks)))

    # Fill the matrix with iteration counts.
    for case, statistic in data.items():
        solver_name, adaptive_error_ratio_str, *varying_param = case.split("_")
        varying_param = "_".join(varying_param)
        adaptive_error_ratio = float(adaptive_error_ratio_str)

        i = solvers.index(f"{solver_name}_{adaptive_error_ratio}")
        j = x_ticks.index(varying_param)
        if solver_name.startswith("AHC"):
            tot_nl_iters = sum(flatten(statistic.timestep_nl_iters))
        elif solver_name.startswith("Newton"):
            tot_nl_iters = sum(statistic.timestep_nl_iters)
        matrix[i, j] = tot_nl_iters

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define a colormap
    cmap = plt.cm.YlOrRd  # type: ignore
    norm = colors.Normalize(vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))  # type: ignore

    # Create custom colormap with white for zeros.
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    im = ax.imshow(masked_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Number of nonlinear iterations")

    # Configure axes
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_yticks(np.arange(len(solvers)))
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(solvers)

    # Add grid
    ax.set_xticks(np.arange(-0.5, len(x_ticks), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(solvers), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)

    # Add parameter value inside each cell
    for i in range(len(solvers)):
        for j in range(len(x_ticks)):
            if matrix[i, j] != 0:
                text = str(int(matrix[i, j]))
            else:
                text = "Diverged"
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="black",
                )

    # Format x-axis if the parameter values are float numbers
    if isinstance(x_ticks[0], float):
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add labels and title
    ax.set_xlabel(varying_param_name)
    ax.set_ylabel("Solver")
    if title is None:
        title = f"Nonlinear iterations by solver and {varying_param_name}"
    ax.set_title(title)

    fig.tight_layout()
    return fig


def plot_estimators(
    statistics: SimulationStatistics,
    title: str | None = None,
) -> plt.Figure:
    """Create a plot showing the evolution of different error estimators over time.

    Returns:
        A matplotlib figure with the plotted estimators.

    """
    # Check if HC estimator is present.
    uses_hc: bool = len(statistics.hc_estimator) > 0

    fig, ax = plt.subplots(figsize=(10, 6))

    tot_nl_iterations: int = 0
    ts_nl_iterations: int = 0
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
            # Plot for each HC iteration.
            hc_est = statistics.hc_estimator[i]
            for hc_est_i, lin_est_i in zip(hc_est, lin_est):
                ax.plot(
                    range(ts_nl_iterations, ts_nl_iterations + len(hc_est_i)),
                    hc_est_i,
                    "g^-",
                    markersize=4,
                    fillstyle="none",
                    label=r"$\eta_{hc}$" if i == 0 else "",
                )
                ax.plot(
                    range(ts_nl_iterations, ts_nl_iterations + len(lin_est_i)),
                    lin_est_i,
                    "mo-",
                    markersize=4,
                    fillstyle="none",
                    label=r"$\eta_{lin}$" if i == 0 else "",
                )
                ts_nl_iterations += len(lin_est)

            spat_est_flat = flatten(spat_est)
            temp_est_flat = flatten(temp_est)

        else:
            ax.plot(
                range(ts_nl_iterations, ts_nl_iterations + len(lin_est)),
                lin_est,
                "mo-",
                markersize=4,
                fillstyle="none",
                label=r"$\eta_{lin}$" if i == 0 else "",
            )

            ts_nl_iterations += len(lin_est)

            spat_est_flat = spat_est
            temp_est_flat = temp_est

        # Plot spatial and temporal estimators for the full time step.
        ax.plot(
            range(tot_nl_iterations, tot_nl_iterations + len(spat_est_flat)),
            spat_est_flat,
            "bo-",
            markersize=4,
            fillstyle="none",
            label=r"$\eta_{sp}$" if i == 0 else "",
        )
        ax.plot(
            range(tot_nl_iterations, tot_nl_iterations + len(spat_est_flat)),
            temp_est_flat,
            "rv-",
            markersize=4,
            fillstyle="none",
            label=r"$\eta_{temp}$" if i == 0 else "",
        )

        # Update number of nl iterations.
        tot_nl_iterations += len(spat_est_flat)

    # Set y scale to log
    ax.set_yscale("log")

    # Add labels and title
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Error estimate (log scale)")
    if title is None:
        title = "Error Estimators Evolution"
    ax.set_title(title)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best")

    fig.tight_layout()
    return fig


# endregion

if __name__ == "__main__":
    configs = generate_configs()
    configs_varying_rp_init_s_08 = configs[::10] + configs[1::10] + configs[2::10]
    configs_varying_rp_init_s_09 = configs[3::10] + configs[4::10] + configs[5::10]
    configs_varying_ref_init_s08 = configs[::10] + configs[6::10] + configs[7::10]
    configs_varying_ref_init_s09 = configs[3::10] + configs[8::10] + configs[9::10]
    data = {}
    for config in configs_varying_rp_init_s_08:
        if config.rp_model_2["model"] == "Brooks-Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}"
        else:
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}_{config.rp_model_2['power']}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig1 = plot_nl_iterations(
        data,
        "Rel. perm. model",
        title=r"NL iterations by solver and rel. perm. model. $s_{init} = 0.8$",
    )
    data = {}
    for config in configs_varying_rp_init_s_09:
        if config.rp_model_2["model"] == "Brooks-Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}"
        else:
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}_{config.rp_model_2['power']}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig2 = plot_nl_iterations(
        data,
        "Rel. perm. model",
        title=r"NL iterations by solver and rel. perm. model. $s_{init} = 0.9$",
    )
    data = {}
    for config in configs_varying_ref_init_s08:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig3 = plot_nl_iterations(
        data,
        "grid refinement",
        title=r"NL iterations by solver and grid refinement. Brooks-Corey rel. perm., $s_{init} = 0.8$",
    )
    data = {}
    for config in configs_varying_ref_init_s09:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig4 = plot_nl_iterations(
        data,
        "grid refinement",
        title=r"NL iterations by solver and grid refinement. Brooks-Corey rel. perm., $s_{init} = 0.9$",
    )

    fig1.savefig(dirname / "nl_iters_rp_model_s_init_08.png")
    fig2.savefig(dirname / "nl_iters_rp_model_s_init_09.png")
    fig3.savefig(dirname / "nl_iters_ref_s_init_08.png")
    fig4.savefig(dirname / "nl_iters_ref_s_init_09.png")
    # fig4 = plot_estimators(data["NewtonAppleyard_0.1_0.2"])
    # fig4.savefig(dirname / "estimators_newton_appleyard.png")
    # fig5 = plot_estimators(data["AHC_0.1_0.2"])
    # fig5.savefig(dirname / "estimators_ahc.png")
