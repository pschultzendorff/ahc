import pathlib
import sys

import porepy as pp
from run import generate_configs

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import plot_nl_iterations, read_data

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

EXPECTED_FINAL_TIME = 3000.0 * pp.DAY

if __name__ == "__main__":
    configs = generate_configs()
    configs_varying_rp_init_s_08 = configs[:16]
    configs_varying_rp_init_s_09 = configs[16:]
    configs_varying_ref_init_s_08 = configs[32:44]
    configs_varying_ref_init_s_09 = configs[44:]
    data_1 = {}
    for config in configs_varying_rp_init_s_08:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        elif config.rp_model_2["model"] == "Brooks-Corey-Mualem":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_Br.-Corey {config.rp_model_2['n_b']}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_1[key] = statistics
    fig1 = plot_nl_iterations(
        data_1,
        "rel. perm. model",
    )

    data_2 = {}
    for config in configs_varying_rp_init_s_09:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        elif config.rp_model_2["model"] == "Brooks-Corey-Mualem":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_Br.-Corey {config.rp_model_2['n_b']}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_2[key] = statistics
    fig2 = plot_nl_iterations(
        data_2,
        "rel. perm. model",
    )
    data_3 = {}
    for config in configs_varying_ref_init_s_08:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_3[key] = statistics
    fig3 = plot_nl_iterations(
        data_3,
        "refinement factor",
    )

    data_4 = {}
    for config in configs_varying_ref_init_s_09:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        statistics = read_data(config, EXPECTED_FINAL_TIME)
        data_4[key] = statistics
    fig4 = plot_nl_iterations(
        data_4,
        "refinement factor",
    )

    fig_dir = dirname / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig1.savefig(fig_dir / "nl_iters_rp_model_s_init_08.png")
    fig2.savefig(fig_dir / "nl_iters_rp_model_s_init_09.png")
    fig3.savefig(fig_dir / "nl_iters_ref_fac_s_init_08.png")
    fig4.savefig(fig_dir / "nl_iters_ref_fac_s_init_09.png")
