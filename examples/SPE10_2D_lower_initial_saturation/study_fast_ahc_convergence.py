import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
from tpf.utils.constants_and_typing import FEET

dirname: pathlib.Path = pathlib.Path(__file__).parent

solver: str = "ahc"

# region VARYING_CELL_SIZES

cell_size: int = int(600 * FEET / 60)
basedir: pathlib.Path = (
    dirname
    / solver
    / "varying_cell_sizes"
    / list((dirname / solver / "varying_cell_sizes").iterdir())[0]
)
file_path: pathlib.Path = basedir / f"cellsz_{cell_size}" / "solver_statistics.json"
with open(file_path, "r") as f:
    history: dict[str, Any] = json.load(f)
    history_list: list[Any] = list(history.values())[1:]

    fig, ax = plt.subplots()

    discretization_est: list[float] = []
    hc_est: list[float] = []
    linearization_est: list[float] = []
    for n, time_step in enumerate(history_list):
        # Skip the last 5 values at each time steps. They are general information, not
        # corresponding to any nonlinear loop.
        for nl_step in list(time_step.values())[:-5]:
            if isinstance(nl_step, int) or isinstance(nl_step, float):
                continue
            discretization_est.extend(nl_step["discretization_error_estimates"])
            hc_est.extend(nl_step["hc_error_estimates"])
            linearization_est.extend(nl_step["linearization_error_estimates"])

    ax.semilogy(discretization_est, label="Discretization estimator")
    ax.semilogy(hc_est, label="HC estimator")
    ax.semilogy(linearization_est, label="Linearization estimator")
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Estimator values")
    ax.legend()
    plt.show()
    fig.savefig("fast_ahc_convergence_cellsz_3.png")

# endregion

# region VARYING_RELATIVE_PERMEABILITIES
basedir = (
    dirname
    / solver
    / "varying_cell_sizes"
    / list((dirname / solver / "varying_rel_perm_models").iterdir())[0]
)
file_path = basedir / f"rp1_linear_rp2_Brooks-Corey" / "solver_statistics.json"
with open(file_path, "r") as f:
    history = json.load(f)
    history_list: list[Any] = list(history.values())[1:]

    fig, ax = plt.subplots()

    discretization_est = []
    hc_est = []
    linearization_est = []
    for n, time_step in enumerate(history_list):
        for nl_step in list(time_step.values())[:-5]:
            if isinstance(nl_step, int) or isinstance(nl_step, float):
                continue
            discretization_est.extend(nl_step["discretization_error_estimates"])
            hc_est.extend(nl_step["hc_error_estimates"])
            linearization_est.extend(nl_step["linearization_error_estimates"])

    ax.semilogy(discretization_est, label="Discretization estimator")
    ax.semilogy(hc_est, label="HC estimator")
    ax.semilogy(linearization_est, label="Linearization estimator")
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Estimator values")
    ax.legend()
    plt.show()
    fig.savefig(dirname / "fast_ahc_convergence_Brooks-Corey.png")
# endregion
