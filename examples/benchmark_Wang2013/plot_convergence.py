import itertools
import json
import pathlib

import matplotlib.pyplot as plt

rel_perm_constants_list = [
    {"model": "linear"},
    {"model": "Corey"},
    {"model": "Brooks-Corey"},
]
cap_press_constants_list = [
    {"model": None},
    {"model": "Brooks-Corey"},
]

fig, ax = plt.subplots()
for i, (rp_model, cp_model) in enumerate(
    itertools.product(rel_perm_constants_list, cap_press_constants_list)
):
    run_name: str = f"rel.perm._{rp_model['model']}_cap.press._{cp_model['model']}"
    solver_statistics_file: pathlib.Path = (
        pathlib.Path(__file__).parent / "results" / run_name / "solver_statistics.json"
    )
    with open(solver_statistics_file) as f:
        history = json.load(f)

    nums_iterations: list[int] = [0]
    for time_step in list(history.values())[1:]:
        nums_iterations.append(nums_iterations[-1] + time_step["num_iteration"])
    ax.plot(nums_iterations, label=run_name)
    ax.set_xlabel("Time step (including wasted time steps)")
    ax.set_ylabel("Cumulative number of Newton iterations")
    ax.set_title(f"Newton iterations")
ax.legend()
plt.show()
fig.savefig(pathlib.Path(__file__).parent / "results" / "solver_convergence_no_hc.png")
