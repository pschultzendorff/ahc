import pathlib
import sys

from run import generate_configs

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import plot_nl_iterations, read_data

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()


if __name__ == "__main__":
    configs = generate_configs()
    configs_viscous_varying_rp_init_s_02 = configs[:16]
    configs_viscous_varying_rp_init_s_03 = configs[16:32]
    configs_viscous_varying_init_s = configs[4:8] + configs[32:44] + configs[20:24]
    configs_viscous_and_cap_varying_cap_init_s_03 = configs[44:60]
    configs_viscous_and_cap_varying_init_s = configs[60:72] + configs[44:48]
    configs_viscous_and_cap_varying_entry_press = configs[44:48] + configs[72:]
    data_1 = {}
    for config in configs_viscous_varying_rp_init_s_03:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        elif config.rp_model_2["model"] == "Brooks-Corey-Mualem":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_Br.-Corey {config.rp_model_2['n_b']}"
        data_1[key] = read_data(config)
    fig1 = plot_nl_iterations(
        data_1,
        "rel. perm. model",
    )
    data_2 = {}
    for config in configs_viscous_varying_rp_init_s_03:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        else:
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}"
        data_2[key] = read_data(config)
    fig2 = plot_nl_iterations(
        data_2,
        "rel. perm. model",
    )
    data_3 = {}
    for config in configs_viscous_varying_init_s:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.init_s}"
        statistics = read_data(config)
        data_3[key] = statistics
    fig3 = plot_nl_iterations(
        data_3,
        r"$s_\mathrm{w}^0$",
    )
    data_4 = {}
    for config in configs_viscous_and_cap_varying_cap_init_s_03:
        key = (
            f"{config.solver_name}_{config.adaptive_error_ratio}_"
            + f"{config.cp_model_2['model']} $nb = {config.cp_model_2['n_b']}$"
            + f" & {config.rp_model_2['model']}"
        )
        if config.rp_model_2["model"] == "Corey":
            key += f"$p = {config.rp_model_2['power']}$"
        statistics = read_data(config)
        data_4[key] = statistics
    fig4 = plot_nl_iterations(
        data_4,
        "cap. press. model & rel. perm. model",
    )
    data_5 = {}
    for config in configs_viscous_and_cap_varying_init_s:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.init_s}"
        statistics = read_data(config)
        data_5[key] = statistics
    fig5 = plot_nl_iterations(
        data_5,
        r"$s_\mathrm{w}^0$",
    )
    data_6 = {}
    for config in configs_viscous_and_cap_varying_entry_press:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.cp_model_2['entry_pressure']}"
        statistics = read_data(config)
        data_6[key] = statistics
    fig6 = plot_nl_iterations(
        data_6,
        r"$p_\mathrm{e}$",
    )

    fig1.savefig(dirname / "nl_iters_viscous_varying_rp_init_s_02.png")
    fig2.savefig(dirname / "nl_iters_viscous_varying_rp_init_s_03.png")
    fig3.savefig(dirname / "nl_iters_viscous_varying_init_s.png")
    fig4.savefig(dirname / "nl_iters_viscous_and_cap_varying_cap_init_s_03.png")
    fig5.savefig(dirname / "nl_iters_viscous_and_cap_varying_init_s.png")
    fig6.savefig(dirname / "nl_iters_viscous_and_cap_varying_entry_press.png")
