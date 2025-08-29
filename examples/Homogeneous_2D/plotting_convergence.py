import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tpf.utils.constants_and_typing import FEET

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()
solvers: list[str] = [
    "ahc_0.1",
    "ahc_0.01",
    "Newton_0.1",
    "Newton_0.01",
    "Newton_Appleyard_0.1",
    "Newton_Appleyard_0.01",
]

# region VARYING_CELL_SIZES
cell_sizes: list[float] = [
    600 * FEET / 7.5,
    600 * FEET / 15,
    600 * FEET / 30,
    600 * FEET / 60,
]
total_iterations: dict[str, dict[str, list[int]]] | dict[str, list[int]] = {
    solver: {} for solver in solvers
}
iterations_per_time_step: (
    dict[str, dict[str, dict[str, int]]] | dict[str, dict[str, int]]
) = {solver: {} for solver in solvers}

for solver in solvers:
    for saturation_case in (dirname / solver / "varying_cell_sizes").iterdir():
        basedir: pathlib.Path = (
            dirname / solver / "varying_cell_sizes" / saturation_case.name
        )
        total_iterations[solver][saturation_case.name[-5:]] = []
        for cell_size in cell_sizes:
            file_path: pathlib.Path = (
                basedir / f"cellsz_{int(cell_size)}" / "solver_statistics.json"
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
                elif solver.startswith("Newton"):
                    num_iterations: int = sum(
                        time_step["num_iteration"]
                        # No iterations saved on the zeroth time step.
                        for time_step in list(data.values())[1:]
                    )
                total_iterations[solver][saturation_case.name[-5:]].append(
                    num_iterations
                )

for saturation_case in total_iterations[solvers[0]]:
    x = range(len(cell_sizes))
    width: float = 0.15

    fig, ax = plt.subplots()
    ax.bar(
        x,
        total_iterations["ahc_0.1"][saturation_case],
        width,
        label="AHC 0.1",
        color=plt.cm.Blues(0.7),
        alpha=0.8,
        edgecolor=plt.cm.Blues(0.7),
    )
    ax.bar(
        [p + width for p in x],
        total_iterations["ahc_0.01"][saturation_case],
        width,
        label="AHC 0.01",
        color=plt.cm.Blues(0.7),
        alpha=0.8,
        edgecolor=plt.cm.Blues(0.7),
        hatch="\\\\\\",
    )
    ax.bar(
        [p + 2 * width for p in x],
        total_iterations["Newton_0.1"][saturation_case],
        width,
        label="Newton 0.1",
        color=plt.cm.Greys(0.7),
        alpha=0.8,
        edgecolor=plt.cm.Greys(0.7),
    )
    ax.bar(
        [p + 3 * width for p in x],
        total_iterations["Newton_0.01"][saturation_case],
        width,
        label="Newton 0.01",
        color=plt.cm.Greys(0.7),
        alpha=0.8,
        edgecolor=plt.cm.Greys(0.7),
        hatch="\\\\\\",
    )
    ax.bar(
        [p + 4 * width for p in x],
        total_iterations["Newton_Appleyard_0.1"][saturation_case],
        width,
        label="Newton Appleyard 0.1",
        color=plt.cm.Greys(0.555),
        alpha=0.8,
        edgecolor=plt.cm.Greys(0.55),
    )
    ax.bar(
        [p + 5 * width for p in x],
        total_iterations["Newton_Appleyard_0.01"][saturation_case],
        width,
        label="Newton Appleyard 0.01",
        color=plt.cm.Greys(0.55),
        alpha=0.8,
        edgecolor=plt.cm.Greys(0.55),
        hatch="\\\\\\",
    )

    ax.set_xticks([p + 2.5 * width for p in x])
    ax.set_xticklabels([str(int(cell_size)) for cell_size in cell_sizes])
    ax.set_yscale("log")
    ax.set_ylabel("total # nonlinear iterations")
    ax.set_xlabel("Cellsize [m]")
    ax.legend()
    ax.set_title("rel. perm.: Brooks-Corey, cap press.: None")

    fig.savefig(dirname / f"varying_cell_sizes_{saturation_case}.png")

    # # region VARYING_TIMES
    # total_iterations_time: dict[str, dict[str, list[int]]] | dict[str, list[int]] = {
    #     solver: {} for solver in solvers
    # }

    # for solver in solvers:
    #     for saturation_case in (dirname / solver / "varying_times").iterdir():
    #         basedir: pathlib.Path = (
    #             dirname / solver / "varying_times" / saturation_case.name
    #         )
    #         total_iterations_time[solver][saturation_case.name[-6:]] = []
    #         for time in times:
    #             file_path: pathlib.Path = (
    #                 basedir / f"time_{int(time)}" / "solver_statistics.json"
    #             )
    #             with open(file_path, "r") as f:
    #                 data: dict[str, Any] = json.load(f)
    #                 if solver.startswith("ahc"):
    #                     num_iterations: int = sum(
    #                         hc_step["num_iteration"]
    #                         for time_step in list(data.values())[1:]
    #                         for hc_step in list(time_step.values())[:-5]
    #                     )
    #                 elif solver.startswith("Newton"):
    #                     num_iterations: int = sum(
    #                         time_step["num_iteration"]
    #                         for time_step in list(data.values())[1:]
    #                     )
    #                 total_iterations_time[solver][saturation_case.name[-6:]].append(
    #                     num_iterations
    #                 )

    # for saturation_case in total_iterations_time[solvers[0]]:
    #     x = range(len(times))
    #     width: float = 0.15

    #     fig, ax = plt.subplots()
    #     ax.bar(
    #         x, total_iterations_time["ahc_0.1"][saturation_case], width, label="AHC 0.1"
    #     )
    #     ax.bar(
    #         [p + width for p in x],
    #         total_iterations_time["ahc_0.01"][saturation_case],
    #         width,
    #         label="AHC 0.01",
    #     )
    #     ax.bar(
    #         [p + 2 * width for p in x],
    #         total_iterations_time["Newton_0.1"][saturation_case],
    #         width,
    #         label="Newton 0.1",
    #     )
    #     ax.bar(
    #         [p + 3 * width for p in x],
    #         total_iterations_time["Newton_0.01"][saturation_case],
    #         width,
    #         label="Newton 0.01",
    #     )
    #     ax.bar(
    #         [p + 4 * width for p in x],
    #         total_iterations_time["Newton_Appleyard_0.1"][saturation_case],
    #         width,
    #         label="Newton Appleyard_0.1",
    #     )
    #     ax.bar(
    #         [p + 5 * width for p in x],
    #         total_iterations_time["Newton_Appleyard_0.01"][saturation_case],
    #         width,
    #         label="Newton Appleyard_0.01",
    #     )

    #     ax.set_xticks([p + 2.5 * width for p in x])
    #     ax.set_xticklabels([str(time) for time in times])
    #     ax.set_yscale("log")
    #     ax.set_ylabel("total # nonlinear iterations")
    #     ax.set_xlabel("Time [s]")
    #     ax.legend()
    #     ax.set_title("rel. perm.: Brooks-Corey, cap press.: None")

    #     fig.savefig(dirname / f"varying_times_{saturation_case}.png")

    # endregion

# endregion
