import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter, LogLocator, NullFormatter
from run import default_time_manager_params
from run_all_layers import generate_cases

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import (
    SolverStats,
    _flatten_nested_list,
    calc_relative_est,
    read_solver_stats,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()


# region UTILS


def _mark_failed_runs(ax, x: np.ndarray, y: np.ndarray, color: str):
    """Mark failed runs on a plot by interpolating the y-position for failed points and
    plotting them as empty circles.

    """
    failed = y == 0

    if not np.any(failed):
        return

    y = y.astype(float)
    y[failed] = np.nan

    y_interp = np.interp(
        x[failed],
        x[~failed],
        y[~failed],
    )

    ax.plot(
        x[failed],
        y_interp,
        linestyle="none",
        marker="o",
        markersize=20,
        markerfacecolor="none",
        markeredgecolor=color,
        markeredgewidth=2,
    )


def plot_statistics(
    data: dict[str, SolverStats],
):
    # Collect num_time_steps and num_iterations per solver and layer.
    num_iterations: dict[str, dict[str, float]] = defaultdict(dict)
    num_time_steps: dict[str, dict[str, int]] = defaultdict(dict)
    relative_errors: dict[str, dict[str, float]] = {}

    for case, stats in data.items():
        solver, hc_tol, nl_tol, layer = case.split("_")

        if solver == "HC":
            solver_label = solver
        elif solver == "AHC":
            solver_label = rf"{solver} $\gamma_\mathrm{{HC}} = {hc_tol}$"
        elif solver.startswith("Newton"):
            solver_label = rf"{solver} $\gamma_\mathrm{{lin}} = {nl_tol}$"

        num_iterations[solver_label][layer] = (
            sum(stats.timestep_nl_iters)
            if solver.startswith("Newton")
            else sum(_flatten_nested_list(stats.timestep_nl_iters))
        )
        num_time_steps[solver_label][layer] = len(stats.discrete_times)

        # Forgot to save global energy norm for adaptive Newton, thus we use AHC to
        # calculate the relative errors.
        if solver.startswith("AHC"):
            relative_errors[layer] = calc_relative_est(stats)

    # Turn into sorted arrays. Sorting in the first dimension is solvers/error_key, in
    # the second dimension layers.
    solver_labels = sorted(num_iterations)
    error_keys = sorted(relative_errors["0"])
    # Ignore spatial error estimators.
    error_keys.remove("spat")

    layers = sorted(relative_errors, key=int)

    num_iterations_array = np.asarray(  # type: ignore
        [
            [num_iterations[solver][layer] for layer in layers]
            for solver in solver_labels
        ]
    )
    num_time_steps_array = np.asarray(  # type: ignore
        [
            [num_time_steps[solver][layer] for layer in layers]
            for solver in solver_labels
        ]
    )

    # NOTE relative_errors is sorted the other way around, i.e., first layers then
    # error_keys.
    relative_errors_array = np.asarray(  # type: ignore
        [
            [relative_errors[layer][error_key] for layer in layers]
            for error_key in error_keys
        ]
    )

    fig_list = []

    num_layers = len(layers)
    layer_idx = np.arange(1, num_layers + 1)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:cyan"]
    linestyles = ["-", "--", "-.", ":", (0, (1, 3))]

    datasets = [
        (num_iterations_array, solver_labels, "# cumulative nonlinear iterations"),
        (num_time_steps_array, solver_labels, "# time steps"),
        (
            relative_errors_array,
            [
                rf"$\hat{{\eta}}_\mathrm{{{error_key.upper()}}}$"
                if error_key == "hc"
                else rf"$\hat{{\eta}}_\mathrm{{{error_key}}}$"
                for error_key in error_keys
            ],
            "Relative error estimator",
        ),
    ]

    for dataset, labels, y_label in datasets:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot for all solvers/error_keys. The x-axis is the layer number.
        for i, data_over_layers in enumerate(dataset):  # type: ignore
            ax.plot(
                layer_idx,
                data_over_layers,
                label=labels[i],
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2,
            )
            # NOTE Marking failed runs based on where the errors are zero doesn't make
            # too much sense, but we don't need that plot anyways.
            _mark_failed_runs(ax, layer_idx, data_over_layers, colors[i % len(colors)])

        # Mark upper and lower layers.
        ax.axvline(x=35.5, color="black", linestyle=":", linewidth=2)
        ax.set_xlim(1, num_layers)
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
        ax.set_ylabel(y_label, fontsize=16, fontweight="bold")
        ax.legend(loc="lower left", framealpha=0.7, fontsize=14)

        fig.tight_layout()
        fig_list.append(fig)

    fig1, fig2, fig3 = fig_list
    return fig1, fig2, fig3


# endregion

if __name__ == "__main__":
    EXPECTED_FINAL_TIME = default_time_manager_params["schedule"][-1]  # type: ignore

    configs = generate_cases()
    data = {}
    for config in configs:
        key = (
            f"{config.solver_name}_{config.hc_tol}_{config.nl_tol}_{config.spe10_layer}"
        )
        data[key] = read_solver_stats(config, EXPECTED_FINAL_TIME)
    fig1, fig2, fig3 = plot_statistics(
        data,
    )

    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig1.savefig(fig_dir / "num_iterations_per_layer.png", dpi=300)
    fig2.savefig(fig_dir / "num_time_steps_per_layer.png", dpi=300)
    fig3.savefig(fig_dir / "relative_error_per_layer.png", dpi=300)
