import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter, LogLocator, NullFormatter
from run import default_time_manager_params
from run_all_layers import generate_configs

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationStatistics, calc_relative_error, flatten, read_data

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()


# region UTILS


def plot_statistics(
    data: dict[str, SimulationStatistics],
):
    # Collect num_time_steps and num_iterations per solver and layer.
    num_iterations = defaultdict(lambda: defaultdict(float))  # type: ignore
    num_time_steps = defaultdict(lambda: defaultdict(float))  # type: ignore
    relative_errors = {}

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

        # Forgot to save global energy norm for adaptive Newton, thus we use AHC to
        # calculate the relative errors.
        if solver.startswith("AHC"):
            relative_errors[layer] = calc_relative_error(stats)

    # Ensure solvers and layers are sorted correctly.
    solver_labels = list(sorted(num_iterations.keys()))
    error_keys = list(sorted(relative_errors["0"].keys()))
    error_keys.remove("spat")

    # Turn into sorted list (solvers) of sorted list (layers) of values.
    # NOTE A little hacky. turning the sorted into a dict is not expected to maintain
    # ordering.
    num_iterations = [  # type: ignore
        [value for __, value in sorted(v.items(), key=lambda item: int(item[0]))]
        for _, v in sorted(num_iterations.items())
    ]
    num_time_steps = [  # type: ignore
        [value for __, value in sorted(v.items(), key=lambda item: int(item[0]))]
        for _, v in sorted(num_time_steps.items())
    ]

    # Turn into sorted list (errors) of sorted list (layers) of values.
    relative_errors = [  # type: ignore
        [
            value
            for __, value in sorted(
                {k: v[error_key] for k, v in relative_errors.items()}.items(),
                key=lambda item: int(item[0]),
            )
        ]
        for error_key in error_keys
    ]

    fig_list = []

    num_layers = len(num_iterations[0])
    layer_indices = range(1, num_layers + 1)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:cyan"]
    linestyles = ["-", "--", "-.", ":", (0, (1, 3))]

    for data_list in [num_iterations, num_time_steps, relative_errors]:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot either for each error type (relative_errors), or for each solver
        # (num_iterations and num_time_steps)
        if data_list is relative_errors:
            for i, error_type in enumerate(data_list):  # type: ignore
                error_key = error_keys[i]
                label = (
                    rf"$\hat{{\eta}}_\mathrm{{{error_key.upper()}}}$"
                    if error_key == "hc"
                    else rf"$\hat{{\eta}}_\mathrm{{{error_key}}}$"
                )
                ax.plot(
                    layer_indices,
                    error_type,
                    label=label,
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2,
                )
        else:
            for i, solver_data in enumerate(data_list):  # type: ignore
                # Hide the graph at failed timesteps.
                solver_data = np.asarray(solver_data, dtype=np.float32)
                failed = solver_data == 0
                solver_data[failed] = np.nan

                ax.plot(
                    layer_indices,
                    solver_data,
                    label=solver_labels[i],
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2,
                )

                # Interpolate y-position for failed points.
                y_interp = np.interp(
                    np.asarray(layer_indices)[failed],
                    np.asarray(layer_indices)[~failed],
                    solver_data[~failed],
                )
                # Mark failed runs explicitly.
                ax.plot(
                    np.asarray(layer_indices)[failed],
                    y_interp,
                    linestyle="none",
                    marker="o",
                    markersize=20,
                    markerfacecolor="none",
                    markeredgecolor=colors[i % len(colors)],
                    markeredgewidth=2.0,
                    color=colors[i % len(colors)],
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

        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatter(base=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.tick_params(axis="both", which="major", labelsize=14)

        ax.set_xlabel("SPE10 layer", fontsize=16, fontweight="bold")

        if data_list is num_iterations:
            ax.set_ylabel(
                "# cumulative nonlinear iterations", fontsize=16, fontweight="bold"
            )
        elif data_list is num_time_steps:
            ax.set_ylabel("# time steps", fontsize=16, fontweight="bold")
        else:
            ax.set_ylabel("Relative error estimator", fontsize=16, fontweight="bold")

        ax.legend(loc="lower left", framealpha=0.7, fontsize=14)

        fig.tight_layout()
        fig_list.append(fig)

    fig1, fig2, fig3 = fig_list
    return fig1, fig2, fig3


# endregion

if __name__ == "__main__":
    EXPECTED_FINAL_TIME = default_time_manager_params["schedule"][-1]  # type: ignore

    configs = generate_configs()
    data = {}
    for config in configs:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.spe10_layer}"
        data[key] = read_data(config, EXPECTED_FINAL_TIME)
    fig1, fig2, fig3 = plot_statistics(
        data,
    )

    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig1.savefig(fig_dir / "num_iterations_per_layer.png", dpi=300)
    fig2.savefig(fig_dir / "num_time_steps_per_layer.png", dpi=300)
    fig3.savefig(fig_dir / "relative_error_per_layer.png", dpi=300)
