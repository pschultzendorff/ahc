import json
import pathlib

import matplotlib.pyplot as plt

with open(
    pathlib.Path(__file__).parent
    / "results"
    / "hc_estimators_test"
    / "linear to Corey"
    / "solver_statistics.json",
    "r",
) as f:
    history = json.load(f)

discretization_est: list[float] = []
hc_est: list[float] = []
linearization_est: list[float] = []
for nl_step in list(history.values())[1:]:
    discretization_est.extend(nl_step["discretization_est"])
    hc_est.extend(nl_step["hc_est"])
    linearization_est.extend(nl_step["linearization_est"])

plt.semilogy(discretization_est, label="Discretization estimator")
plt.semilogy(hc_est, label="HC estimator")
plt.semilogy(linearization_est, label="Linearization estimator")
plt.xlabel("Nonlinear iteration")
plt.ylabel("Estimator")
plt.title(f"Estimator values")
plt.legend()
plt.show()
