import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from run import generate_configs

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import plot_nl_iterations, plot_spatial_convergence, read_data

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()


if __name__ == "__main__":
    configs = generate_configs()
    configs_varying_rp_init_s_08 = configs[:12]
    configs_varying_rp_init_s_09 = configs[12:24]
    configs_varying_ref_init_s_08 = configs[24:36]
    configs_varying_ref_init_s_09 = configs[36:]
    data_1 = {}
    for config in configs_varying_rp_init_s_08:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        else:
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}"
        statistics = read_data(config)
        data_1[key] = statistics
    fig1 = plot_nl_iterations(
        data_1,
        "rel. perm. model",
    )
    # fig1a = plot_estimators(
    #     data_1["AHC_5e-05_Brooks-Corey-Mualem"],
    #     title="AHC 0.00005 Brooks-Corey-Mualem",
    #     combine_disc_est=True,
    # )

    data_2 = {}
    for config in configs_varying_rp_init_s_09:
        if config.rp_model_2["model"] == "Corey":
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']} {config.rp_model_2['power']}"
        else:
            key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.rp_model_2['model']}"
        statistics = read_data(config)
        data_2[key] = statistics
    fig2 = plot_nl_iterations(
        data_2,
        "rel. perm. model",
    )
    data_3 = {}
    for config in configs_varying_ref_init_s_08:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        statistics = read_data(config)
        data_3[key] = statistics
    fig3 = plot_nl_iterations(
        data_3,
        "refinement factor",
    )
    fig3a = plot_spatial_convergence(
        list(data_3.values())[::4], [5.0, 1.0, 0.5], combine_disc_est=True
    )
    fig3b = plot_spatial_convergence(
        list(data_3.values())[3::4], [5.0, 1.0, 0.5], combine_disc_est=True
    )

    data_4 = {}
    for config in configs_varying_ref_init_s_09:
        key = f"{config.solver_name}_{config.adaptive_error_ratio}_{config.refinement_factor}"
        statistics = read_data(config)
        data_4[key] = statistics
    fig4 = plot_nl_iterations(
        data_4,
        "refinement factor",
    )

    fig1.savefig(dirname / "nl_iters_rp_model_s_init_08.png")
    # fig1a.savefig(dirname / "estimators_rp_model_s_init_08.png")
    fig2.savefig(dirname / "nl_iters_rp_model_s_init_09.png")
    fig3.savefig(dirname / "nl_iters_ref_fac_s_init_08.png")
    fig3a.savefig(dirname / "ahc_spatial_convergence_s_init_08.png")
    fig3b.savefig(dirname / "newtonappleyard_spatial_convergence_s_init_08.png")
    fig4.savefig(dirname / "nl_iters_ref_fac_s_init_09.png")
    # fig5 = plot_estimators(data["AHC_0.1_0.2"], combine_disc_est=True)
    # fig5.savefig(dirname / "estimators_ahc_0.1.png")
    # fig5 = plot_estimators(data["AHC_0.005_0.2"], combine_disc_est=True)
    # fig5.savefig(dirname / "estimators_ahc_0.005.png")
