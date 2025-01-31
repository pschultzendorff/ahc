import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
from tpf.utils.constants_and_typing import FEET

dirname: pathlib.Path = pathlib.Path(__file__).parent


# region VARYING_CELL_SIZES

basedir: pathlib.Path = dirname / "newton_3_ts" / "lay_80"
fig, ax = plt.subplots()

for test_case in basedir.iterdir():
    if test_case.name == "cellsz_6_rp_linear_cp._None":
        color: str = "blue"
    elif test_case.name == "cellsz_3_rp_linear_cp._None":
        color = "red"
    else:
        continue
    file_path: pathlib.Path = test_case / "solver_statistics.json"
    with open(file_path, "r") as f:
        data: dict[str, Any] = json.load(f)
        data_list = list(data.values())[1:]

    global_energy_norm: list[float] = []
    for n, time_step in enumerate(data_list):
        global_energy_norm.extend(time_step["global_energy_norm"])

        # Draw a vertical line when the time increase in the next time step, i.e.,
        # the nonlinear iterations at the current time step converged.
        time: float = time_step["current time"]
        try:
            next_time: float = data_list[n + 1]["current time"]
            if time < next_time:
                ax.axvline(
                    x=len(global_energy_norm), color=color, linestyle="--", alpha=0.2
                )
        except IndexError:
            pass
    ax.plot(global_energy_norm, label=test_case.name, color=color)

ax.set_xlabel("Nonlinear iteration")
ax.set_ylabel("Global energy norm")
ax.set_ylim(bottom=1e-1, top=1e0)
ax.legend()
fig.savefig(dirname / "global_energy_norm.png")

# endregion
