import matplotlib.pyplot as plt
import numpy as np
import json
import os

from tpf_lab.applications.convergence_analysis import plot_convergence_for_timestep

# foldername = os.path.join(
#     "results/buckley_leverett/homotopy_continuation/concave_frac_flow_to_perturbed_rel_perm_2_limited_False/max_newton_iterations_60/homotopy_continuation_decay_0.5"
# )

# with open(os.path.join(foldername, "temporal_error_analysis.json"), "r") as f:
#     results = json.load(f)

# lambdas = [0.5**i for i in range(36)]

# plot_convergence_for_timestep(
#     results,
#     foldername,
#     time_step=5,
#     refinement_level=2,
#     additional_keys=["residuals_wrt_homotopy", "residuals_wrt_goal_function"],
#     # additional_keys=["residuals_wrt_homotopy", "residuals_wrt_goal_function"],
#     additional_data={r"$\lambda$": lambdas},
# )

# delta_t = [0.125, 0.0625, 0.03125, 0.015625]
# time_steps_homotopy_continuation = np.array([0, 0, 32, 64])
# time_steps_nn = np.array([0, 0, 0, 64])
# time_steps_corey = np.array([0, 16, 32, 64])
# max_time_steps = np.array([8, 16, 32, 64])
# fig = plt.figure()
# plt.plot(
#     delta_t,
#     100 * time_steps_homotopy_continuation / max_time_steps,
#     label="homotopy continuation",
#     marker="v",
#     color="g",
# )

# plt.plot(
#     delta_t,
#     100 * time_steps_nn / max_time_steps,
#     label="NN rel perms",
#     marker="x",
#     color="r",
# )
# plt.plot(
#     delta_t,
#     100 * time_steps_corey / max_time_steps,
#     label="Corey rel perms",
#     marker="+",
#     color="b",
# )

# plt.xlabel(r"$\Delta t$")
# plt.xscale("log")
# plt.ylabel(r"successfull time steps (%)")
# plt.legend()
# fig.subplots_adjust(left=0.2, bottom=0.2)
# plt.savefig("time_steps_failure_corey_vs_nn.png")

delta_t = [0.125, 0.0625, 0.03125, 0.015625]
time_steps_nn = np.array([0, 0, 0, 64])
time_steps_corey = np.array([0, 16, 32, 64])
# time_steps_homotopy = np.array([0, 0, 32, 64])
max_time_steps = np.array([8, 16, 32, 64])
fig = plt.figure()

barWidth = 1 / 3
br1 = np.arange(len(delta_t))
br2 = [x + barWidth for x in br1]
# br3 = [x + 2 * barWidth for x in br1]

plt.bar(
    br1,
    100 * time_steps_nn / max_time_steps,
    label="nn rel. perms.",
    color="r",
    width=barWidth,
)


plt.bar(
    br2,
    100 * time_steps_corey / max_time_steps,
    label="Corey rel. perms.",
    color="b",
    width=barWidth,
)
# plt.bar(
#     br3,
#     100 * time_steps_homotopy / max_time_steps,
#     label="homotopy continuation",
#     color="g",
#     width=barWidth,
# )

plt.xlabel(r"$\Delta t$")
plt.ylabel(r"successfull time steps (%)")
# Change ylim so that even
# plt.ylim(-5, 100)
plt.xticks([r + barWidth for r in range(len(delta_t))], delta_t)
plt.legend()
fig.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig("time_steps_failure_corey_vs_nn.png")

# fig = plt.figure()
# xx = np.concatenate(
#     [np.linspace(0, 0.3, 100), np.linspace(0.3, 0.7, 100), np.linspace(0.7, 1.0, 100)]
# )
# yy = np.concatenate([np.zeros(100), np.linspace(0, 1, 100), np.ones(100)])
# plt.plot(xx, yy)
# plt.xlabel(r"$S_w t$")
# plt.ylabel(r"k_{r,w}")
# fig.subplots_adjust(left=0.2, bottom=0.2)
# plt.savefig("linear_rel_perm_w.png")
