import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
from tpf.utils.constants_and_typing import FEET

dirname: pathlib.Path = pathlib.Path(__file__).parent
solvers: list[str] = [
    "ahc",
    # "ahc_appleyard",
    "newton_adaptive_ts",
    "newton_appleyard_adaptive_ts",
]

# region VARYING_CELL_SIZES

cell_sizes: list[int] = [int(600 * FEET / 30), int(600 * FEET / 60)]
total_iterations: dict[str, list[int]] = {solver: [] for solver in solvers}

for solver in solvers:
    basedir: pathlib.Path = (
        dirname
        / solver
        / "varying_cell_sizes"
        / list((dirname / solver / "varying_cell_sizes").iterdir())[0]
    )
    for cell_size in cell_sizes:
        file_path: pathlib.Path = (
            basedir / f"cellsz_{cell_size}" / "solver_statistics.json"
        )
        with open(file_path, "r") as f:
            data: dict[str, Any] = json.load(f)
            if solver.startswith("ahc"):
                num_iterations: int = sum(
                    hc_step["num_iteration"]
                    # No iterations saved on the zeroth time step.
                    for time_step in list(data.values())[1:]
                    # The last 5 values for each time step are information regarding hc,
                    # not the individual Newton loops.
                    for hc_step in list(time_step.values())[:-5]
                )
            elif solver.startswith("newton"):
                num_iterations: int = sum(
                    time_step["num_iteration"]
                    # No iterations saved on the zeroth time step.
                    for time_step in list(data.values())[1:]
                )
            total_iterations[solver].append(num_iterations)

x = range(len(cell_sizes))
width: float = 0.15

fig, ax = plt.subplots()
ax.bar(x, total_iterations["ahc"], width, label="AHC")
# ax.bar(
#     [p + width for p in x],
#     total_iterations["ahc_appleyard"],
#     width,
#     label="AHC w. Appleyard",
# )
ax.bar(
    [p + width for p in x],
    total_iterations["newton_adaptive_ts"],
    width,
    label="Newton",
)
ax.bar(
    [p + 2 * width for p in x],
    total_iterations["newton_appleyard_adaptive_ts"],
    width,
    label="Newton w. Appleyard",
)

ax.set_xticks([p + width for p in x])
ax.set_xticklabels([str(size) for size in cell_sizes])
ax.set_yscale("log")
ax.set_ylabel("total # nonlinear iterations")
ax.set_xlabel("cell size [m]")
ax.legend()
ax.set_title(
    "Capillary pressure: None, SPE10 layer: 80, relative pressure: Corey w. power 2"
)

fig.savefig(dirname / "varying_cell_sizes.png")

# endregion

# region VARYING_CAP_PRESS_MODELS

# cap_press_models: list[str] = ["None", "linear"]
# total_iterations = {solver: [] for solver in solvers}

# for solver in solvers:
#     basedir: pathlib.Path = (
#         dirname
#         / solver
#         / "varying_cp_models"
#         / list((dirname / solver / "varying_cp_models").iterdir())[0]
#     )
#     for cap_press_model in cap_press_models:
#         file_path: pathlib.Path = (
#             basedir / f"cp._{cap_press_model}" / "solver_statistics.json"
#         )
#         with open(file_path, "r") as f:
#             data: dict[str, Any] = json.load(f)
#             if solver.startswith("ahc"):
#                 num_iterations: int = sum(
#                     hc_step["num_iteration"]
#                     # No iterations saved on the zeroth time step.
#                     for time_step in list(data.values())[1:]
#                     # The last 5 values for each time step are information regarding hc,
#                     # not the individual Newton loops.
#                     for hc_step in list(time_step.values())[:-5]
#                 )
#             elif solver.startswith("newton"):
#                 num_iterations: int = sum(
#                     time_step["num_iteration"]
#                     # No iterations saved on the zeroth time step.
#                     for time_step in list(data.values())[1:]
#                 )
#             total_iterations[solver].append(num_iterations)

# x = range(len(cap_press_models))

# fig, ax = plt.subplots()
# ax.bar(x, total_iterations["ahc"], width, label="AHC")
# ax.bar(
#     [p + width for p in x],
#     total_iterations["ahc_appleyard"],
#     width,
#     label="AHC w. Appleyard",
# )

# ax.set_xticks([p + width / 2 for p in x])
# ax.set_xticklabels(cap_press_models)
# ax.set_ylabel("total # nonlinear iterations")
# ax.set_xlabel("capillary pressure model")
# ax.legend()

# fig.savefig(dirname / "varying_cap_press_models.png")
# fig.show()

# endregion

# region VARYING_REL_PERM_MODELS
rel_perm_models = [
    {"model": "Corey", "power": 2},
    {"model": "Corey", "power": 3},
    {"model": "Brooks-Corey"},
    # {"model": "van Genuchten-Burdine"},
]
total_iterations = {solver: [] for solver in solvers}

for solver in solvers:
    basedir = (
        dirname
        / solver
        / "varying_rel_perm_models"
        / list((dirname / solver / "varying_rel_perm_models").iterdir())[0]
    )
    for model in rel_perm_models:
        if solver.startswith("ahc"):
            if model["model"] == "Corey":
                file_path = (
                    basedir
                    / f"rp1_linear_rp2_{model['model']}_power_{model['power']}"
                    / "solver_statistics.json"
                )
            else:
                file_path = (
                    basedir
                    / f"rp1_linear_rp2_{model['model']}"
                    / "solver_statistics.json"
                )
        elif solver.startswith("newton"):
            if model["model"] == "Corey":
                file_path = (
                    basedir
                    / f"rp_{model['model']}_power_{model['power']}"
                    / "solver_statistics.json"
                )
            else:
                file_path = basedir / f"rp_{model['model']}" / "solver_statistics.json"
        with open(file_path, "r") as f:
            data: dict[str, Any] = json.load(f)
            if solver.startswith("ahc"):
                num_iterations = sum(
                    hc_step["num_iteration"]
                    # No iterations saved on the zeroth time step.
                    for time_step in list(data.values())[1:]
                    # The last 5 values for each time step are information regarding hc,
                    # not the individual Newton loops.
                    for hc_step in list(time_step.values())[:-5]
                )
            elif solver.startswith("newton"):
                num_iterations = sum(
                    time_step["num_iteration"]
                    # No iterations saved on the zeroth time step.
                    for time_step in list(data.values())[1:]
                )
            total_iterations[solver].append(num_iterations)

x = range(len(rel_perm_models))

fig, ax = plt.subplots()
ax.bar(x, total_iterations["ahc"], width, label="AHC")
# ax.bar(
#     [p + width for p in x],
#     total_iterations["ahc_appleyard"],
#     width,
#     label="AHC w. Appleyard",
# )
ax.bar(
    [p + width for p in x],
    total_iterations["newton_adaptive_ts"],
    width,
    label="Newton",
)
ax.bar(
    [p + 2 * width for p in x],
    total_iterations["newton_appleyard_adaptive_ts"],
    width,
    label="Newton w. Appleyard",
)


ax.set_xticks([p + width for p in x])
ax.set_xticklabels(
    [
        (
            model["model"] + str(model["power"])
            if model["model"] == "Corey"
            else model["model"]
        )
        for model in rel_perm_models
    ]
)
ax.set_yscale("log")
ax.set_ylabel("total # nonlinear iterations")
ax.set_xlabel("rel. perm. model")
ax.legend()
ax.set_title("Capillary pressure: None, SPE10 layer: 80, cellsize: 6 m")

fig.savefig(dirname / "varying_rel_perm_models.png")

# endregion
