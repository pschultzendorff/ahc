import json
import pathlib

import matplotlib.pyplot as plt

for hc_setup in ["linear to Corey", "linear to linear", "Corey to Corey"]:
    with open(
        pathlib.Path(__file__).parent
        / "results"
        / "hc_estimators_test"
        / hc_setup
        / "solver_statistics.json",
        "r",
    ) as f:
        history = json.load(f)

    discretization_est: list[float] = []
    hc_est: list[float] = []
    lin_est: list[float] = []
    for time_step in list(history.values())[1:]:
        for nl_step in list(time_step.values()):
            if isinstance(nl_step, int):
                continue
            discretization_est.extend(nl_step["discretization_error_estimates"])
            hc_est.extend(nl_step["hc_error_estimates"])
            lin_est.extend(nl_step["linearization_error_estimates"])

    fig, ax = plt.subplots()
    ax.semilogy(discretization_est, label="Discretization estimator")
    ax.semilogy(hc_est, label="HC estimator")
    ax.semilogy(lin_est, label="Linearization estimator")
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Estimator values")
    ax.legend()
    plt.show()
    fig.savefig(
        pathlib.Path(__file__).parent
        / "results"
        / "hc_estimators_test"
        / hc_setup
        / "solver_convergence.png"
    )
