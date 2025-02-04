import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt

# filename: pathlib.Path = (
#     pathlib.Path(__file__).parent
#     / "ahc"
#     / f"varying_rel_perm_models"
#     / "lay_80_cellsz_6_cp._None_init_s_0.2"
#     / "rp1_linear_rp2_Brooks-Corey"
#     / "solver_statistics.json"
# )
# with open(filename) as f:
#     history: dict[str, Any] = json.load(f)
#     history_list: list[Any] = list(history.values())[1:]

# fig, ax = plt.subplots()

# discretization_est: list[float] = []
# hc_est: list[float] = []
# linearization_est: list[float] = []
# for n, time_step in enumerate(history_list):
#     for nl_step in list(time_step.values())[:-5]:
#         discretization_est.extend(nl_step["discretization_error_estimates"])
#         hc_est.extend(nl_step["hc_error_estimates"])
#         linearization_est.extend(nl_step["linearization_error_estimates"])

#     # Draw a vertical line when the time increase in the next time step, i.e., the
#     # nonlinear iterations at the current time step converged.
#     time: float = time_step["current time"]
#     try:
#         next_time: float = history_list[n + 1]["current time"]
#         if time < next_time:
#             ax.axvline(x=len(discretization_est), color="gray", linestyle="--")
#     except IndexError:
#         pass

# ax.semilogy(discretization_est[:100], label="Discretization estimator")
# ax.semilogy(hc_est[:100], label="HC estimator")
# ax.semilogy(linearization_est[:100], label="Linearization estimator")
# ax.set_xlabel("Nonlinear iteration")
# ax.set_ylabel("Estimator")
# ax.set_title(f"Estimator values")
# ax.legend()
# plt.show()
# fig.savefig("ahc_convergence.png")

filename = (
    pathlib.Path(__file__).parent
    / "ahc_slow"
    / f"varying_rel_perm_models"
    / "lay_80_cellsz_6_cp._None_init_s_0.2"
    / "rp1_linear_rp2_Brooks-Corey"
    / "solver_statistics.json"
)
with open(filename) as f:
    history = json.load(f)
    history_list = list(history.values())[1:]

fig, ax = plt.subplots()

discretization_est = []
hc_est = []
linearization_est = []
for n, time_step in enumerate(history_list):
    for nl_step in list(time_step.values())[:-5]:
        discretization_est.extend(nl_step["discretization_error_estimates"])
        hc_est.extend(nl_step["hc_error_estimates"])
        linearization_est.extend(nl_step["linearization_error_estimates"])

    # Draw a vertical line when the time increase in the next time step, i.e., the
    # nonlinear iterations at the current time step converged.
    time: float = time_step["current time"]
    try:
        next_time: float = history_list[n + 1]["current time"]
        if time < next_time:
            ax.axvline(x=len(discretization_est), color="gray", linestyle="--")
    except IndexError:
        pass

ax.semilogy(discretization_est, label="Discretization estimator")
ax.semilogy(hc_est, label="HC estimator")
ax.semilogy(linearization_est, label="Linearization estimator")
ax.set_ylim(bottom=1e-6, top=1e3)
ax.set_xlabel("Nonlinear iteration")
ax.set_ylabel("Estimator")
ax.set_title("Estimator values")
ax.legend()
plt.show()
fig.savefig("ahc_slow_convergence.png")
