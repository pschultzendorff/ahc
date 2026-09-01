import json
import pathlib
import sys
from collections.abc import Callable

from matplotlib.figure import Figure
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
    read_solver_stats,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = default_time_manager_params["schedule"][-1]  # type: ignore

rel_ests: dict[str, str] = {}


def plot_study(
    cases: list[SimulationConfig],
    key_func: Callable[[SimulationConfig, SolverStats], tuple[str, str]],
    varying_param_name: str,
    **kwargs,
) -> Figure:
    # To allow for easy sorting of the data by solver specs and parameter values in
    # plot_nl_iterations, we use tuples of strings as keys. The first string is the
    # solver name plus all relevant specs, while the second string is the parameter
    # value. See the various helper functions below for details.
    data: dict[tuple[str, str], SolverStats] = {}
    for config in cases:
        if config.solver_name == "ReferenceSolution":
            # The solver statistics of the reference solution are not of interest.
            continue
        stats = read_solver_stats(config, EXPECTED_FINAL_TIME)
        key = key_func(config, stats)
        data[key] = stats

        # Calculate relative estimator values for the finest AHC solution
        if config.solver_name == "AHC" and config.hc_tol == 0.01:
            if stats.converged:
                rel_ests[f"{config.folder_name()}_{key}"] = (
                    f"{calc_relative_est(stats)['total']:.2f}"
                )
            else:
                rel_ests[f"{config.folder_name()}_{key}"] = "not converged"
    return plot_nl_iterations(data, varying_param_name, **kwargs)


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
    fig_dir.mkdir(exist_ok=True)

    for study_name, study in studies.items():
        kwargs = {}
        match study_name:
            case (
                "viscous_varying_rp_init_s_02"
                | "viscous_varying_rp_init_s_02_spat_est_on"
                | "viscous_varying_rp_init_s_03"
                | "buoyancy_varying_rp"
            ):
                key_func = _key_varying_rp
                varying_param_name = "Relative permeability model"
            case "viscous_varying_init_s" | "capillary_varying_init_s":
                key_func = _key_varying_init_s
                varying_param_name = r"$s_\mathrm{w}^0$"
            case "capillary_varying_rp":
                key_func = _key_varying_cap
                varying_param_name = "Capillary pressure & Relative permeability model"
                kwargs = {
                    "tight_layout": True,
                    "rotate_x_labels": True,
                    "extended_figure_height": True,
                }
            case "capillary_varying_entry_pressure":
                key_func = _key_varying_entry_pressure
                varying_param_name = r"$p_\mathrm{e}$"
            case "buoyancy_varying_density" | "gravity_segregation":
                key_func = _key_varying_water_density
                varying_param_name = r"$\rho_\mathrm{w}$"
            case _:
                raise ValueError(f"Unknown study: {study_name}")

        fig = plot_study(
            study,
            key_func=key_func,
            varying_param_name=varying_param_name,
            **kwargs,
        )
        fig.savefig(fig_dir / f"nl_iters_{study_name}.png")

    with (fig_dir / "relative_errors.txt").open("w") as f:
        json.dump(rel_ests, f, indent=2)
