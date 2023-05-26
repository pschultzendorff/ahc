import logging
import os
import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from tpf_lab.applications.convergence_analysis import ConvergenceAnalysisExtended

# Fix seed for reproducability.
random.seed(0)
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def save_convergence_results(
    analysis: ConvergenceAnalysisExtended,
    results: list,
    level_type: Literal["time", "space"] = "time",
    courant_number: float = 0.0,
    max_iterations: int = 30,
    foldername: str = "results",
) -> None:
    x_axis = "time_step" if level_type == "time" else "cell_diameter"
    ooc = analysis.order_of_convergence(
        analysis.transform_results_to_classical(results), x_axis=x_axis
    )
    logger.info(f"Order of convergence: {ooc}")
    with open(
        os.path.join(foldername, f"order_of_convergence_in_{level_type}.txt"), "w"
    ) as f:
        f.write(f"order of convergence: {ooc}")

    final_errors, cell_diameter_levels, time_step_levels = analysis.final_l2_error(
        results
    )
    xx = time_step_levels if level_type == "time" else cell_diameter_levels
    # Plot final error for each parameter.
    fig = plt.figure()
    plt.plot(
        xx,
        final_errors,
        "xb-",
        label=f"final error ({max_iterations} iter)",
    )
    plt.xlabel(r"$\log_{10}(\Delta t)$")
    plt.ylabel(r"$\log_{10}(\|e\|)$")
    plt.xscale("log")
    plt.yscale("log")
    plt.axvline(x=courant_number, color="b", label=r"$\mathcal{C}$")
    plt.title("Convergence Analysis")
    plt.legend()
    fig.subplots_adjust(left=0.4, bottom=0.2)
    plt.savefig(os.path.join(foldername, f"accuracy_in_{level_type}.png"))

    (
        avg_nonlinear_iterations,
        cell_diameter_levels,
        time_step_levels,
    ) = analysis.average_number_of_iterations(results)
    xx = time_step_levels if level_type == "time" else cell_diameter_levels
    fig = plt.figure()
    plt.plot(
        xx,
        avg_nonlinear_iterations,
        "xb-",
        label=f"average Newton iterations",
    )
    plt.xlabel(r"$\log_{10}(\Delta t)$")
    plt.ylabel(r"$n_{iterations}$")
    plt.xscale("log")
    plt.axvline(x=courant_number, color="b", label=r"$\mathcal{C}$")
    plt.title("Convergence analysis")
    plt.legend()
    fig.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(
        os.path.join(foldername, f"average_newton_iterations_varying_{x_axis}.png")
    )
