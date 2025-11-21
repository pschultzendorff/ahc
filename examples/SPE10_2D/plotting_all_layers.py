import json
import math
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from run_all_layers import SimulationConfig, generate_configs

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

sns.set_theme("paper")
sns.set_style("whitegrid")


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
    converged: bool = True
    final_time: float = 0.0


def flatten(xx: list[list]) -> list:
    return [x for sublist in xx for x in sublist]


def read_data(config: SimulationConfig) -> SimulationStatistics:
    # If not we can read the data.
    with open(config.folder_name / "solver_statistics.json", "r") as f:
        data: dict[str, Any] = json.load(f)

    # Check if the solver failed at some time step and return reduced statistics.
    if "failure" in [f.stem for f in config.folder_name.iterdir()]:
        final_time = list(data.values())[-1]["current time"]
        return SimulationStatistics(converged=False, final_time=final_time)

    # Else, read all estimators etc.
    stats = SimulationStatistics()

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
            stats.hc_estimator.append(hc_estimator)
            stats.lambdas.append(time_step["hc_lambdas"])
        elif config.solver_name.startswith("Newton"):
            if config.solver_name == "NewtonAppleyard":
                pass
            # Read iterations per time step.
            num_nl_iterations = time_step["num_iteration"]
            spat_estimator = time_step["spatial_est"]
            temp_estimator = time_step["temp_est"]
            lin_estimator = time_step["lin_est"]

        # Append data to the statistics object.
        stats.time_steps.append(time_step["current time"])
        stats.timestep_nl_iters.append(num_nl_iterations)
        stats.spat_estimator.append(spat_estimator)
        stats.temp_estimator.append(temp_estimator)
        stats.lin_estimator.append(lin_estimator)

    return stats


def plot_statistics(
    data: dict[str, SimulationStatistics],
):
    # Collect num_time_steps and num_iterations per solver and layer.
    num_iterations = defaultdict(lambda: defaultdict(float))
    num_time_steps = defaultdict(lambda: defaultdict(float))

    for case, stats in data.items():
        solver, adaptive_error_ratio_str, layer = case.split("_")

        if solver == "HC":
            solver_label = solver
        elif solver == "AHC":
            solver_label = (
                rf"{solver} $\gamma_\mathrm{{HC}} = {adaptive_error_ratio_str}$"
            )
        elif solver.startswith("Newton"):
            solver_label = (
                rf"{solver} $\gamma_\mathrm{{lin}} = {adaptive_error_ratio_str}$"
            )

        iters = (
            sum(stats.timestep_nl_iters)
            if solver.startswith("Newton")
            else sum(flatten(stats.timestep_nl_iters))
        )
        num_iterations[solver_label][layer] = iters
        num_time_steps[solver_label][layer] = len(stats.time_steps)

    # Ensure solvers and layers are sorted correctly.
    solver_labels = list(sorted(num_iterations.keys()))
    num_iterations = [
        list(dict(sorted(v.items(), key=lambda item: int(item[0]))).values())
        for _, v in sorted(num_iterations.items())
    ]
    num_time_steps = [
        list(dict(sorted(v.items(), key=lambda item: int(item[0]))).values())
        for _, v in sorted(num_time_steps.items())
    ]

    fig_list = []

    num_layers = len(num_iterations[0])
    layer_indices = range(1, num_layers + 1)
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
    ]
    linestyles = ["-", "--", "-.", ":"]

    for data_list in [num_iterations, num_time_steps]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, solver_data in enumerate(data_list):
            ax.plot(
                layer_indices,
                solver_data,
                label=solver_labels[i],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2,
            )

        # Mark upper and lower layers.
        ax.axvline(x=35.5, color="black", linestyle=":", linewidth=2)
        ax.set_xlim(1, num_layers)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # y_text_pos = 0.5 * ylim[1]  # Vertically centered.
        ax.text(
            0.175,
            0.55,
            "upper layers",
            fontsize=16,
            fontweight="bold",
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.675,
            0.55,
            "lower layers",
            fontsize=16,
            fontweight="bold",
            ha="center",
            transform=ax.transAxes,
        )

        ax.set_yscale("log")
        # Ignore Pylance complaining about `ticker` not being in `matplotlib`.
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter())  # type: ignore
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=12)
        ax.set_xlabel("SPE10 layer", fontsize=16, fontweight="bold")

        if data_list is num_iterations:
            ax.set_ylabel("# nonlinear iterations", fontsize=16, fontweight="bold")
        else:
            ax.set_ylabel("# time steps", fontsize=16, fontweight="bold")

        ax.legend(loc="lower left", framealpha=0.7, fontsize=14)

        fig.tight_layout()
        fig_list.append(fig)

    fig1, fig2 = fig_list
    return fig1, fig2


# endregion

if __name__ == "__main__":
    configs = generate_configs()
    data = {}
    for config in configs:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.spe10_layer}"
        data[key] = read_data(config)
    fig1, fig2 = plot_statistics(
        data,
    )

    fig1.savefig(dirname / "num_iterations_per_layer.png", dpi=300)
    fig2.savefig(dirname / "num_time_steps_per_layer.png", dpi=300)
