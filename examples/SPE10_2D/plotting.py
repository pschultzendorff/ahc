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
    SimulationStatistics,
    calc_relative_error,
    plot_nl_iterations,
    read_data,
)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = default_time_manager_params["schedule"][-1]  # type: ignore

rel_errors: dict[str, str] = {}


def plot_study(
    cases: list[SimulationConfig],
    key_func: Callable[[SimulationConfig], str],
    varying_param_name: str,
    **kwargs,
) -> Figure:
    data: dict[str, SimulationStatistics] = {}
    for config in cases:
        stats = read_data(config, EXPECTED_FINAL_TIME)
        key = key_func(config)
        data[key] = stats
        if config.solver_name == "AHC" and config.hc_tol == 0.01:
            if stats.converged:
                rel_errors[f"{config.folder_name()}_{key}"] = (
                    f"{calc_relative_error(stats)['total']:.2f}"
                )
            else:
                rel_errors[f"{config.folder_name()}_{key}"] = "not converged"
    return plot_nl_iterations(data, varying_param_name, **kwargs)


def _key_varying_rp(config: SimulationConfig) -> str:
    if config.rp_model_2["model"] == "Corey":
        return f"{config.solver_name}_{config.hc_tol}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
    elif config.rp_model_2["model"] == "Brooks-Corey-Mualem":
        return (
            f"{config.solver_name}_{config.hc_tol}_Br.-Corey {config.rp_model_2['n_b']}"
        )
    else:
        raise ValueError(
            f"Unknown relative permeability model: {config.rp_model_2['model']}"
        )


def _key_varying_init_s(config: SimulationConfig) -> str:
    return f"{config.solver_name}_{config.hc_tol}_{config.init_s}"


def _key_varying_cap(config: SimulationConfig) -> str:
    key = (
        f"{config.solver_name}_{config.hc_tol}_"
        + f"Br.-C. $nb={config.cp_model_2['n_b']}$\n"
    )
    if config.rp_model_2["model"] == "Corey":
        key += f"C. $p={config.rp_model_2['power']}$"
    else:
        key += f"Br.-C. $nb={config.rp_model_2['n_b']}$"
    return key


def _key_varying_entry_pressure(config: SimulationConfig) -> str:
    return f"{config.solver_name}_{config.hc_tol}_{config.cp_model_2['entry_pressure']}"


def _key_varying_water_density(config: SimulationConfig) -> str:
    return f"{config.solver_name}_{config.hc_tol}_{config.spe10_water_density}"


if __name__ == "__main__":
    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    for study_name, study in studies.items():
        kwargs = {}
        match study_name:
            case (
                "viscous_varying_rp_init_s_02"
                | "viscous_varying_rp_init_s_03"
                | "buoyancy_varying_rp"
                | "gravity_separation"
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
            case "buoyancy_varying_density":
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
        json.dump(rel_errors, f, indent=2)
