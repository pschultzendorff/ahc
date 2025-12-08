import pathlib
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
from run_all_layers import generate_configs

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationStatistics, calc_relative_error, flatten, read_data

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()


# region UTILS


def plot_statistics(
    data: dict[str, SimulationStatistics],
):
    # Collect num_time_steps and num_iterations per solver and layer.
    num_iterations = defaultdict(lambda: defaultdict(float))
    num_time_steps = defaultdict(lambda: defaultdict(float))
    relative_errors = defaultdict(lambda: defaultdict(float))

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

        # Forgot to save global energy norm for adaptive Newton.
        if solver.startswith("AHC"):
            relative_errors[solver_label][layer] = calc_relative_error(stats)

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
    relative_errors = [
        list(dict(sorted(v.items(), key=lambda item: int(item[0]))).values())
        for _, v in sorted(relative_errors.items())
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

    for data_list in [num_iterations, num_time_steps, relative_errors]:
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
        elif data_list is num_time_steps:
            ax.set_ylabel("# time steps", fontsize=16, fontweight="bold")
        else:
            ax.set_ylabel(r"$\eta_{\mathrm{tot}}$", fontsize=16, fontweight="bold")

        if data_list is not relative_errors:
            ax.legend(loc="lower left", framealpha=0.7, fontsize=14)

        fig.tight_layout()
        fig_list.append(fig)

    fig1, fig2, fig3 = fig_list
    return fig1, fig2, fig3


# endregion

if __name__ == "__main__":
    configs = generate_configs()
    data = {}
    for config in configs:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.spe10_layer}"
        data[key] = read_data(config)
    fig1, fig2, fig3 = plot_statistics(
        data,
    )

    fig1.savefig(dirname / "num_iterations_per_layer.png", dpi=300)
    fig2.savefig(dirname / "num_time_steps_per_layer.png", dpi=300)
    fig3.savefig(dirname / "relative_error_per_layer.png", dpi=300)
