import json
import pathlib
import sys

from .run import default_time_manager_params, generate_configs

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from ..utils import calc_relative_error, plot_nl_iterations, read_data

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = default_time_manager_params["schedule"][-1]  # type: ignore

if __name__ == "__main__":
    configs = generate_configs()
    configs_viscous_varying_rp_init_s_02 = configs[:20]
    configs_viscous_varying_rp_init_s_03 = configs[20:40]
    configs_viscous_varying_init_s = configs[5:10] + configs[40:55] + configs[25:30]
    configs_viscous_and_cap_varying_cap_init_s_03 = configs[55:75]
    configs_viscous_and_cap_varying_init_s = configs[75:90] + configs[55:60]
    configs_viscous_and_cap_varying_entry_press = configs[55:60] + configs[90:]
    rel_errors = {}

    data_1 = {}
    for config in configs_viscous_varying_rp_init_s_02:
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
    for config in configs_viscous_varying_rp_init_s_03:
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
    for config in configs_viscous_varying_init_s:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.init_s}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_3[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_3_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_3_{key}"] = "not converged"
    fig3 = plot_nl_iterations(data_3, r"$s_\mathrm{w}^0$")

    data_4 = {}
    for config in configs_viscous_and_cap_varying_cap_init_s_03:
        key = (
            f"{config.solver_name}_{config.adaptive_error_ratio}_"
            + f"Br.-C. $nb={config.cp_model_2['n_b']}$\n"
        )
        if config.rp_model_2["model"] == "Corey":
            key += f"C. $p={config.rp_model_2['power']}$"
        else:
            key += f"Br.-C. $nb={config.rp_model_2['n_b']}$"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_4[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_4_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_4_{key}"] = "not converged"
    fig4 = plot_nl_iterations(
        data_4,
        "Capillary pressure & Relative permeability model",
        tight_layout=True,
        rotate_x_labels=True,
        extended_figure_height=True,
    )

    data_5 = {}
    for config in configs_viscous_and_cap_varying_init_s:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.init_s}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_5[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_5_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_5_{key}"] = "not converged"
    fig5 = plot_nl_iterations(data_5, r"$s_\mathrm{w}^0$")

    data_6 = {}
    for config in configs_viscous_and_cap_varying_entry_press:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.cp_model_2['entry_pressure']}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_6[key] = statistics
        if config.solver_name == "AHC" and config.adaptive_error_ratio == 0.01:
            if statistics.converged:
                rel_errors[f"fig_6_{key}"] = (
                    f"{calc_relative_error(statistics)['total']:.2f}"
                )
            else:
                rel_errors[f"fig_6_{key}"] = "not converged"
    fig6 = plot_nl_iterations(data_6, r"$p_\mathrm{e}$")

    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    with (fig_dir / "relative_errors.txt").open("w") as f:
        json.dump(rel_errors, f, indent=2)

    fig1.savefig(fig_dir / "nl_iters_viscous_varying_rp_init_s_02.png")
    fig2.savefig(fig_dir / "nl_iters_viscous_varying_rp_init_s_03.png")
    fig3.savefig(fig_dir / "nl_iters_viscous_varying_init_s.png")
    fig4.savefig(fig_dir / "nl_iters_viscous_and_cap_varying_cap_init_s_03.png")
    fig5.savefig(fig_dir / "nl_iters_viscous_and_cap_varying_init_s.png")
    fig6.savefig(fig_dir / "nl_iters_viscous_and_cap_varying_entry_press.png")
