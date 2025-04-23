import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ahc import SimulationConfig, generate_configs

dirname: pathlib.Path = pathlib.Path(__file__).parent
sns.set_theme()


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
    """Create a heatmap showing nonlinear iterations for different solvers and parameter values.

    Args:
        data: Dictionary mapping simulation configurations to simulation statistics.
        varying_param_name: Name of the parameter that varies between the configurations.
        title: Optional title for the plot.
    """
    # Extract solvers and parameter values from case names
    cases = list(data.keys())
    solvers = sorted(set("\n".join(case.split("_")[:2]) for case in cases))
    x_ticks = sorted(
        set(" ".join(case.split("_")[2:]) for case in cases),
        key=lambda x: (x.isdigit(), x),
    )

    # Transform data to two arrays for iterations and annotations.
    iterations = np.empty((len(solvers), len(x_ticks)))
    annotations = np.empty((len(solvers), len(x_ticks)), dtype="<U25")

    for case, stat in data.items():
        solver_name, adaptive_error_ratio_str, varying_param = case.split("_")

        adaptive_error_ratio = float(adaptive_error_ratio_str)

        i = solvers.index(f"{solver_name}\n{adaptive_error_ratio}")
        j = x_ticks.index(varying_param)

        tot_nl_iters = (
            sum(flatten(stat.timestep_nl_iters))
            if solver_name.startswith("AHC")
            else sum(stat.timestep_nl_iters)
        )

        iterations[i, j] = tot_nl_iters
        annotations[i, j] = f"{tot_nl_iters}\n({len(stat.time_steps)})"

    mask = iterations == 0

    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.heatmap(
        iterations,
        mask=mask,
        annot=annotations,
        fmt="s",
        cmap="Blues",
        cbar=True,
        cbar_kws={"label": "Number of nonlinear iterations"},
        xticklabels=x_ticks,
        yticklabels=solvers,
        linewidths=0.8,
        ax=ax,
    )

    # Annotate failed simulations
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    r"Reached min. $\Delta t$",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    # Set labels and title
    ax.set_xlabel(varying_param_name, fontsize=12, fontweight="bold")
    ax.set_ylabel("Solver & adaptive error ratio", fontsize=12, fontweight="bold")
    ax.set_title(
        title or f"#NL iterations (#time steps) by solver and {varying_param_name}",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
    return fig


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
            # Plot NL est for each HC iteration.
            for lin_est_i in lin_est:
                ax.plot(
                    range(
                        tot_nl_iterations_fine, tot_nl_iterations_fine + len(lin_est_i)
                    ),
                    lin_est_i,
                    "mo-",
                    markersize=4,
                    fillstyle="none",
                    markerfacecolor="none",
                    label=r"$\eta_{lin}$" if i == 0 else "",
                )
                tot_nl_iterations_fine += len(lin_est_i)

            hc_est_flat = flatten(statistics.hc_estimator[i])
            spat_est_flat = flatten(spat_est)
            temp_est_flat = flatten(temp_est)

        else:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(lin_est)),
                lin_est,
                "mo-",
                markersize=4,
                fillstyle="none",
                markerfacecolor="none",
                label=r"$\eta_{lin}$" if i == 0 else "",
            )

            spat_est_flat = spat_est
            temp_est_flat = temp_est

        if uses_hc:
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(hc_est_flat)),
                hc_est_flat,
                "g^-",
                markersize=4,
                fillstyle="none",
                markerfacecolor="green",
                label=r"$\eta_{hc}$" if i == 0 else "",
            )

        # Plot spatial and temporal estimators for the full time step.
        if combine_disc_est:
            # Combine spatial and temporal estimators.
            disc_est_flat = np.array(spat_est_flat) + np.array(temp_est_flat)
            ax.plot(
                range(tot_nl_iterations, tot_nl_iterations + len(disc_est_flat)),
                disc_est_flat,
                "bo-",
                markersize=4,
                fillstyle="none",
                markerfacecolor="blue",
                label=r"$\eta_{disc}$" if i == 0 else "",
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
                label=r"$\eta_{temp}$" if i == 0 else "",
            )

        # Update number of nl iterations.
        tot_nl_iterations += len(spat_est_flat)

    # Set y scale to log
    ax.set_yscale("log")

    # Add labels and title
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel(
        "Nonlinear iteration",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Error estimate (log scale)",
        fontsize=14,
        fontweight="bold",
    )
    if title is None:
        title = "Error Estimators Evolution"
    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
    )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="best",
        ncol=2,
        prop={"size": 14, "weight": "bold"},
    )

    fig.tight_layout()
    return fig


# endregion

if __name__ == "__main__":
    configs = generate_configs()
    configs_varying_rp_init_s_08 = configs[:12]
    configs_varying_rp_init_s_09 = configs[12:24]
    configs_varying_ref_init_s_08 = configs[24:36]
    configs_varying_ref_init_s_09 = configs[:4] + configs[36:]
    data = {}
    for config in configs_varying_rp_init_s_08:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        else:
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig1 = plot_nl_iterations(
        data,
        "rel. perm. model",
    )
    data = {}
    for config in configs_varying_rp_init_s_09:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        else:
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig2 = plot_nl_iterations(
        data,
        "rel. perm. model",
    )
    data = {}
    for config in configs_varying_ref_init_s_08:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig3 = plot_nl_iterations(
        data,
        "refinement factor",
    )
    data = {}
    for config in configs_varying_ref_init_s_09:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        try:
            statistics = read_data(config)
        except ValueError:
            statistics = SimulationStatistics()
        data[key] = statistics
    fig4 = plot_nl_iterations(
        data,
        "refinement factor",
    )

    fig1.savefig(dirname / "nl_iters_rp_model_s_init_08.png")
    fig2.savefig(dirname / "nl_iters_rp_model_s_init_09.png")
    fig3.savefig(dirname / "nl_iters_ref_fac_s_init_08.png")
    fig4.savefig(dirname / "nl_iters_ref_fac_s_init_09.png")
    # fig4 = plot_estimators(data["NewtonAppleyard_0.1_0.2"])
    # fig4.savefig(dirname / "estimators_newton_appleyard.png")
    # fig5 = plot_estimators(data["AHC_0.1_0.2"], combine_disc_est=True)
    # fig5.savefig(dirname / "estimators_ahc_0.1.png")
    # fig5 = plot_estimators(data["AHC_0.005_0.2"], combine_disc_est=True)
    # fig5.savefig(dirname / "estimators_ahc_0.005.png")
