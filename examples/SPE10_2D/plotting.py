import csv
import json
import pathlib
import sys
from collections.abc import Callable

from ahc.utils.compare import ComparisonStats
from matplotlib import pyplot as plt
from run import (
    default_time_manager_params,
    studies,
)

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import (
    SimulationConfig,
    SolverStats,
    calc_relative_est,
    plot_nl_iterations,
    read_comparison_stats,
    read_solver_stats,
    tabulate_comparison_stats,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = default_time_manager_params["schedule"][-1]  # type: ignore

REL_ESTS: dict[str, str] = {}


def analyze_study(
    cases: list[SimulationConfig],
    key_func: Callable[[SimulationConfig, SolverStats], tuple[str, str]],
    varying_param_name: str,
    figure_save_path: pathlib.Path,
    table_save_path: pathlib.Path,
    **kwargs,
) -> None:
    """Read all cases from a study, plot solver statistics, tabulate comparison
    statistics, and save the results.

    Raises:
        ValueError: If any of the reference solutions did not converge in a single time
            step. In that case, the comparison statistics are not meaningful and cannot
            be tabulated.

    """
    # To allow for easy sorting of the data by solver specs and parameter values in
    # plot_nl_iterations, we use tuples of strings as keys. The first string is the
    # solver name plus all relevant specs, while the second string is the parameter
    # value. See the various helper functions below for details.
    data_solver: dict[tuple[str, str], SolverStats] = {}
    data_comparison: dict[tuple[str, str], ComparisonStats] = {}

    # Store the parameter_values (second element of the key) for which the reference
    # solution did not converge. The comparison statistics for these cases will be set
    # to default values to indicate that they are not meaningful.
    failed_reference_solutions: list[str] = []

    # Read solver and comparison statistics for all cases.
    for config in cases:
        solver_stats = read_solver_stats(config, EXPECTED_FINAL_TIME)
        key = key_func(config, solver_stats)

        # There are several cases to consider regarding the reference solution and the
        # comparison statistics.
        # - The reference solution exists but the solver did not converge in a single
        #   time step. This is handled by read_comparison_stats.
        # - The reference solver failed with an error at some point and no solution was
        #   saved. This is handled by run, which does not save any comparison stats.
        #   Read comparison stats will return a default ComparisonStats object in this
        #   case.
        # - The reference solver converged in more than one time step. This is handled
        #   below.
        if config.solver_name == "ReferenceSolution":
            if not solver_stats.converged or len(solver_stats.discrete_times) != 1:
                failed_reference_solutions.append(key[1])

        else:
            # For all other solvers, we store the solver and comparison stats.
            # In no comparison
            comparison_stats = read_comparison_stats(config, solver_stats)
            data_solver[key] = solver_stats
            data_comparison[key] = comparison_stats

        # Calculate relative estimator values from the finest AHC solution.
        if (
            config.solver_name == "AHC"
            and config.hc_tol == 0.01
            and config.nl_tol == 0.01
        ):
            if solver_stats.converged:
                REL_ESTS[f"{config.folder_name()}_{key}"] = (
                    f"{calc_relative_est(solver_stats)['total']:.2f}"
                )
            else:
                REL_ESTS[f"{config.folder_name()}_{key}"] = "not converged"

    for parameter_value in failed_reference_solutions:
        # Set the comparison statistics to default values to indicate that they are not
        # meaningful.
        for key in data_comparison:
            if key[1] == parameter_value:
                data_comparison[key] = ComparisonStats()
                break

    # Plot solver statistics and tabulate comparison statistics.
    solver_stats_fig = plot_nl_iterations(data_solver, varying_param_name, **kwargs)
    comparison_stats_table = tabulate_comparison_stats(
        data_comparison, varying_param_name, **kwargs
    )

    solver_stats_fig.savefig(figure_save_path)
    plt.close(solver_stats_fig)

    with table_save_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(comparison_stats_table)


def _key_varying_rp(config: SimulationConfig, stats: SolverStats) -> tuple[str, str]:
    match config.rp_model_2["model"]:
        case "Corey":
            parameter_value = (
                f"{config.rp_model_2['model']} {config.rp_model_2['power']}"
            )
        case "Brooks-Corey-Mualem":
            parameter_value = f"Br.-Corey {config.rp_model_2['n_b']}"
        case _:
            raise ValueError(
                f"Unknown relative permeability model: {config.rp_model_2['model']}"
            )
    return config.solver_specs(), parameter_value


def _key_varying_init_s(
    config: SimulationConfig, stats: SolverStats
) -> tuple[str, str]:
    return config.solver_specs(), str(config.init_s)


def _key_varying_cap(config: SimulationConfig, stats: SolverStats) -> tuple[str, str]:
    parameter_value = f"Br.-C. $nb={config.cp_model_2['n_b']}$\n"
    match config.rp_model_2["model"]:
        case "Brooks-Corey-Mualem":
            parameter_value += f"Br.-C. $nb={config.cp_model_2['n_b']}$"
        case "Corey":
            parameter_value += f"C. $p={config.rp_model_2['power']}$"
        case _:
            raise ValueError(f"Unknown rel. perm. model: {config.rp_model_2['model']}")
    return config.solver_specs(), parameter_value


def _key_varying_entry_pressure(
    config: SimulationConfig, stats: SolverStats
) -> tuple[str, str]:
    return config.solver_specs(), str(config.cp_model_2["entry_pressure"])


def _key_varying_water_density(
    config: SimulationConfig, stats: SolverStats
) -> tuple[str, str]:
    return config.solver_specs(), str(config.spe10_water_density)


if __name__ == "__main__":
    fig_dir = dirname / "figures"
    comparison_dir = dirname / "comparison_stats"
    fig_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)

    for study_name, study in studies.items():
        kwargs = {}
        match study_name:
            case (
                "viscous_varying_rp_init_s_02"
                | "viscous_varying_rp_init_s_02_spat_est_on"
                | "viscous_varying_rp_init_s_03"
                | "viscous_varying_rp_with_spatial_estimators_init_s_02"
            ):
                key_func = _key_varying_rp
                varying_param_name = "Relative permeability model"
            case "viscous_varying_init_s" | "capillary_varying_init_s":
                key_func = _key_varying_init_s
                varying_param_name = r"$s_\mathrm{w}^0$"
            case "capillary_varying_rp" | "buoyancy_varying_rp":
                key_func = _key_varying_cap
                varying_param_name = "Capillary pressure & Relative permeability model"
                kwargs = {
                    "tight_layout": True,
                    "rotate_x_labels": True,
                    "extended_figure_height": True,
                }
            case "capillary_varying_entry_pressure" | "buoyancy_varying_entry_pressure":
                key_func = _key_varying_entry_pressure
                varying_param_name = r"$p_\mathrm{e}$"
            case "buoyancy_varying_density" | "gravity_segregation":
                key_func = _key_varying_water_density
                varying_param_name = r"$\rho_\mathrm{w}$"
            case _:
                raise ValueError(f"Unknown study: {study_name}")

        # FIXME Add cases from viscous rel. perm studies to init_s studies.

        fig = analyze_study(
            study,
            key_func=key_func,
            varying_param_name=varying_param_name,
            figure_save_path=fig_dir / f"nl_iters_{study_name}.png",
            table_save_path=comparison_dir / f"comparison_stats_{study_name}.csv",
            **kwargs,
        )

    with (fig_dir / "relative_errors.txt").open("w") as f:
        json.dump(REL_ESTS, f, indent=2)
