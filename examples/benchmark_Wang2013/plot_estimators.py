import itertools
import json
import pathlib

import matplotlib.pyplot as plt

rel_perm_constants_list = [
    {"model": "linear"},
    {"model": "Corey"},
    {"model": "Brooks-Corey"},
]
cap_press_constants_list = [{"model": None}, {"model": "Brooks-Corey"}]

for i, (rp_model_1, rp_model_2, cp_model) in enumerate(
    itertools.product(
        rel_perm_constants_list, rel_perm_constants_list, cap_press_constants_list
    )
):
    if i != 2:
        continue
    run_name: str = (
        f"rel.perm._{rp_model_1['model']}_to_rel.perm._{rp_model_2['model']}_cap.press._{cp_model['model']}"
    )
    solver_statistics_file: pathlib.Path = (
        pathlib.Path(__file__).parent / "results" / run_name / "solver_statistics.json"
    )
    with open(solver_statistics_file) as f:
        history = json.load(f)

    discretization_est: list[float] = []
    hc_est: list[float] = []
    linearization_est: list[float] = []
    for time_step in list(history.values())[1:]:
        for nl_step in list(time_step.values()):
            if isinstance(nl_step, int):
                continue
            discretization_est.extend(nl_step["discretization_error_estimates"])
            hc_est.extend(nl_step["hc_error_estimates"])
            linearization_est.extend(nl_step["linearization_error_estimates"])

    fig, ax = plt.subplots()
    ax.semilogy(discretization_est, label="Discretization estimator")
    ax.semilogy(hc_est, label="HC estimator")
    ax.semilogy(linearization_est, label="Linearization estimator")
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Estimator values")
    ax.legend()
    plt.show()
    fig.savefig(
        pathlib.Path(__file__).parent
        / "results"
        / run_name
        / "solver_convergence_hc.png"
    )
