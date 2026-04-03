import json
import pathlib
import sys

import porepy as pp
from .run import generate_configs

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from ..utils import calc_relative_error, plot_nl_iterations, read_data

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = 3000.0 * pp.DAY

if __name__ == "__main__":
    configs = generate_configs()
    configs_varying_rp_init_s_08 = configs[:20]
    configs_varying_rp_init_s_09 = configs[20:40]
    configs_varying_ref_init_s_08 = configs[40:55]
    configs_varying_ref_init_s_09 = configs[55:]
    rel_errors = {}

    data_1 = {}
    for config in configs_varying_rp_init_s_08:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        elif config.rp_model_2["model"] == "Brooks-Corey-Mualem":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_Br.-Corey {config.rp_model_2['n_b']}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_1[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_1_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_1_{key}"] = "not converged"
    fig1 = plot_nl_iterations(data_1, "Relative permeability model")

    data_2 = {}
    for config in configs_varying_rp_init_s_09:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        elif config.rp_model_2["model"] == "Brooks-Corey-Mualem":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_Br.-Corey {config.rp_model_2['n_b']}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_2[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_2_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_2_{key}"] = "not converged"
    fig2 = plot_nl_iterations(data_2, "Relative permeability model")
    data_3 = {}
    for config in configs_varying_ref_init_s_08:
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{statistics.num_grid_cells}"
        data_3[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_3_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_3_{key}"] = "not converged"
    fig3 = plot_nl_iterations(data_3, "Number of grid cells")

    data_4 = {}
    for config in configs_varying_ref_init_s_09:
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{statistics.num_grid_cells}"
        data_4[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_4_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_4_{key}"] = "not converged"
    fig4 = plot_nl_iterations(data_4, "Number of grid cells")

    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    with (fig_dir / "relative_errors.txt").open("w") as f:
        json.dump(rel_errors, f, indent=2)

    fig1.savefig(fig_dir / "nl_iters_rp_model_s_init_08.png")
    fig2.savefig(fig_dir / "nl_iters_rp_model_s_init_09.png")
    fig3.savefig(fig_dir / "nl_iters_ref_fac_s_init_08.png")
    fig4.savefig(fig_dir / "nl_iters_ref_fac_s_init_09.png")
