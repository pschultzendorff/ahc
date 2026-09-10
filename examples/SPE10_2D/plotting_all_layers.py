import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter, LogLocator, NullFormatter
from run import default_time_manager_params
from run_all_layers import generate_all_layers

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import (
    SimulationConfig,
    SolverStats,
    _flatten_nested_list,
    calc_relative_est,
    read_solver_stats,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = default_time_manager_params["schedule"][-1]

# region UTILS


def _mark_failed_runs(ax, x: np.ndarray, y: np.ndarray, failed: np.ndarray, color: str):
    """Mark failed runs on a plot by interpolating the y-position for failed points and
    plotting them as empty circles.

    """

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
) -> list[plt.Figure]:
    data_dtype = np.dtype(
        [
            # Make the strings long enough to avoid any issues.
            ("solver_label", "U200"),
            ("layer", np.int32),
            ("nl_iterations", np.int32),
            ("num_time_steps", np.int32),
            ("converged", np.bool_),
            ("relative_errors", object),
        ]
    )
    data_as_array = np.zeros(len(data), dtype=data_dtype)

    for i, ((solver_specs, layer), stats) in enumerate(data.items()):
        solver_specs_list: list[str] = solver_specs.split("_")
        solver_name = solver_specs_list[0]
        solver_name_postfix = solver_specs_list[1] if len(solver_specs_list) > 3 else ""
        hc_tol = solver_specs_list[-2]
        nl_tol = solver_specs_list[-1]

        match solver_name:
            case "HC":
                solver_label = rf"{solver_name}{solver_name_postfix} $\beta_{{\min}} = {hc_tol}$ $\epsilon_\mathrm{{Newton}} = {nl_tol}$"
            case "AHC":
                solver_label = rf"{solver_name}{solver_name_postfix} $\gamma_\mathrm{{HC}} = {hc_tol}$ $\gamma_\mathrm{{lin}} = {nl_tol}$"
            case "Newton" | "NewtonAppleyard":
                solver_label = rf"{solver_name}{solver_name_postfix} $\gamma_\mathrm{{lin}} = {nl_tol}$"
            case _:
                raise ValueError(f"Unknown solver: {solver_name}")

        data_as_array[i]["solver_label"] = solver_label
        data_as_array[i]["layer"] = int(layer)
        data_as_array[i]["nl_iterations"] = (
            sum(stats.time_step_nl_iters)
            if solver_name.startswith("Newton")
            else sum(_flatten_nested_list(stats.time_step_nl_iters))
        )
        data_as_array[i]["num_time_steps"] = len(stats.discrete_times)
        data_as_array[i]["converged"] = stats.converged

        # Forgot to save global energy norm for adaptive Newton, thus we use AHC to
        # calculate the relative errors.
        if solver_name.startswith("AHC"):
            errors = calc_relative_est(stats)
            # Ignore spatial error estimators.
            errors.pop("spat")
            data_as_array[i]["relative_errors"] = errors

    # Create ticks for x- and y-axes.
    layers = np.unique(data_as_array["layer"]) + 1
    solver_labels = np.unique(data_as_array["solver_label"])

    # Sort indices first by solver_specs, then by layer number.
    idx = np.lexsort((data_as_array["layer"], data_as_array["solver_label"]))
    # Apply sorting to data.
    data_as_array = data_as_array[idx]

    # Reshape each data column into an array of shape=(len(solver_labels), len(layers)) for
    # the heatmap. Due to the previous sorting, the column value will appear at the x, y
    # value corresponding to the solver_label and layer.
    grids = {
        col: data_as_array[col].reshape(len(solver_labels), len(layers))
        # stats_as_array.dtype.names will not be None. Ignore pylance.
        for col in data_as_array.dtype.names  # type: ignore
        if col not in ("solver_label", "layer")
    }

    fig_list = []

    # Hardcode line styles and colors for now.
    solver_styles = {
        r"AHC $\gamma_\mathrm{HC} = 0.01$ $\gamma_\mathrm{lin} = 0.01$": (
            "#1f77b4",
            "-",
        ),
        r"AHC $\gamma_\mathrm{HC} = 0.1$ $\gamma_\mathrm{lin} = 0.01$": (
            "#4c9ed9",
            "--",
        ),
        r"AHC $\gamma_\mathrm{HC} = 0.1$ $\gamma_\mathrm{lin} = 0.1$": (
            "#9ecae1",
            "-.",
        ),
        r"HC $\beta_{\min} = 0.01$ $\epsilon_\mathrm{Newton} = 1e-05$": (
            "#ff7f0e",
            "-",
        ),
        r"HC $\beta_{\min} = 0.01$ $\epsilon_\mathrm{Newton} = 0.001$": (
            "#ffbb78",
            "--",
        ),
        r"HC $\beta_{\min} = 0.05$ $\epsilon_\mathrm{Newton} = 0.001$": (
            "#fdd0a2",
            "-.",
        ),
        r"Newton $\gamma_\mathrm{lin} = 0.1$": ("#2ca02c", ":"),
        r"NewtonAppleyard $\gamma_\mathrm{lin} = 0.1$": ("#d62728", (0, (1, 2))),
    }

    datasets = [
        (
            "nl_iterations",
            "# cumulative nonlinear iterations",
        ),
        ("num_time_steps", "# time steps"),
        # (
        #     data_as_array["relative_errors"],
        #     [
        #         rf"$\hat{{\eta}}_\mathrm{{{error_key.upper()}}}$"
        #         if error_key == "hc"
        #         else rf"$\hat{{\eta}}_\mathrm{{{error_key}}}$"
        #         for error_key in error_keys
        #     ],
        #     "Relative error estimator",
        # ),
    ]

    for col, y_label in datasets:
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot for all solver_labels. The x-axis is the layer number.
        for i, solver_label in enumerate(solver_labels):
            color, linestyle = solver_styles[solver_label]

            converged = grids["converged"][i]
            # Mask the y_data for failed runs to avoid plotting them. The failed runs
            # will be marked with empty circles in the _mark_failed_runs function.
            y_data = np.asarray(grids[col][i], dtype=np.float32)
            y_data[~converged] = np.nan

            ax.plot(
                layers,
                y_data,
                label=solver_label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            # NOTE Marking failed runs based on where the errors are zero doesn't make
            # too much sense, but we don't need that plot anyways.
            _mark_failed_runs(ax, layers, grids[col][i], ~converged, color)

        # Mark upper and lower layers.
        ax.axvline(x=35.5, color="black", linestyle=":", linewidth=2)
        ax.set_xlim(1, len(layers) + 1)
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
        ax.legend(loc="upper left", framealpha=0.7, fontsize=14)

        fig.tight_layout()
        fig_list.append(fig)

    return fig_list


# endregion


def _key_varying_layer(
    config: SimulationConfig, solver_stats: SolverStats
) -> tuple[str, str]:
    return config.solver_specs(), str(config.spe10_layer)


if __name__ == "__main__":
    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    study = generate_all_layers()
    data_solver = {}
    for config in study:
        if config.solver_name == "ReferenceSolution":
            # Reference solution stats are not of interest.
            continue

        solver_stats = read_solver_stats(config, EXPECTED_FINAL_TIME)
        key = _key_varying_layer(config, solver_stats)
        data_solver[key] = solver_stats

    fig1, fig2 = plot_statistics(data_solver)
    # fig1, fig2, fig3 = plot_statistics(data_solver)

    fig1.savefig(fig_dir / "num_iterations_per_layer.png", dpi=300)
    fig2.savefig(fig_dir / "num_time_steps_per_layer.png", dpi=300)
    # fig3.savefig(fig_dir / "relative_error_per_layer.png", dpi=300)
