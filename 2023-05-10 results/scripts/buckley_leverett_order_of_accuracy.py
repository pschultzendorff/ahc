"""Test order of accuracy for a fractional flow model solving the Buckley-Leverett
problem."""

import json
import logging
import math
import os
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from scipy import interpolate
from buckley_leverett import (
    analytical_solution,
    functions,
    grid,
    misc,
    numerical_solution,
)

from tpf_lab.models.buckley_leverett import BuckleyLeverettEquations
from tpf_lab.models.run_models import run_time_dependent_model
from tpf_lab.utils import logging_redirect_tqdm
from tpf_lab.visualization.diagnostics import DiagnosticsMixinExtended

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BuckleyLeverett_Analytics(DiagnosticsMixinExtended, BuckleyLeverettEquations):
    ...
    """BuckleyLeverett class with diagnostics functionality and ability to save Newton
    iterations and errors each time step."""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self._newton_iterations: list[int] = []
        self._residuals: list[list[float]] = []

    # We change the type of ``errors`` to ``list[float]``, as that is the actual return
    # type for time dependent problems.
    def after_newton_convergence(
        self, solution: np.ndarray, errors: list[float], iteration_counter: int
    ) -> None:  # type: ignore
        self._residuals.append(errors)
        self._newton_iterations.append(iteration_counter)
        return super().after_newton_convergence(solution, errors, iteration_counter)

    # We change the type of ``errors`` to ``list[float]``, as that is the actual return
    # type for time dependent problems.
    def after_newton_failure(
        self, solution: np.ndarray, errors: list[float], iteration_counter: int
    ) -> None:
        self._residuals.append(errors)
        self._newton_iterations.append(iteration_counter)
        return super().after_newton_failure(solution, errors, iteration_counter)


####################
#### Parameters ####
####################
MAX_NEWTON_ITERATIONS = 30

DEFAULT_NUM_GRID_CELLS = 200
DEFAULT_PHYS_SIZE = 20
# Default grid boundaries for the BuckleyLeverett class
XMIN = -10
XMAX = 10

POROSITY = 1.0
VISCOSITY_W = 1.0
VISCOSITY_N = 1.0
DENSITY_W = 1.0
DENSITY_N = 1.0

RESIDUAL_SATURATION_W = 0.3
RESIDUAL_SATURATION_N = 0.3

REL_PERM_MODEL = "power"
REL_PERM_LINEAR_PARAM = 1.0
LIMIT_REL_PERM = True

INFLUX = 1.0
ANGLE = math.pi / 4

# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "buckley_leverett",
    "order_of_accuracy",
    "power",
    f"NEWTON_ITERATIONS_{MAX_NEWTON_ITERATIONS}",
)
try:
    os.makedirs(foldername)
except Exception:
    pass
fh: Optional[logging.FileHandler] = None


##############################################
#### Analytical and Lax-Friedrichs solver ####
##############################################
lax_friedrichs_grid = grid.create_grid(
    (XMIN, XMAX), (XMAX - XMIN) / DEFAULT_NUM_GRID_CELLS
)
initial_condition = np.full_like(lax_friedrichs_grid, RESIDUAL_SATURATION_W)
initial_condition[0 : int(DEFAULT_NUM_GRID_CELLS / 2) - 10] = 1 - RESIDUAL_SATURATION_N
initial_condition[
    int(DEFAULT_NUM_GRID_CELLS / 2) - 10 : int(DEFAULT_NUM_GRID_CELLS / 2) + 10
] = np.linspace(1 - RESIDUAL_SATURATION_N, RESIDUAL_SATURATION_W, 20)

params = {
    "porosity": POROSITY,
    "viscosity_w": VISCOSITY_W,
    "viscosity_n": VISCOSITY_N,
    "density_w": DENSITY_W,
    "density_n": DENSITY_N,
    "S_M": 1 - RESIDUAL_SATURATION_W,
    "S_m": RESIDUAL_SATURATION_N,
    "residual_saturation_w": RESIDUAL_SATURATION_W,
    "residual_saturation_n": RESIDUAL_SATURATION_N,
    "rel_perm_model": REL_PERM_MODEL,
    "rel_perm_linear_param": REL_PERM_LINEAR_PARAM,
    "limit_rel_perm": True,
    # Buckley-Leverett params
    "angle": ANGLE,
    "influx": INFLUX,
    # Lax-Friedrichs params
    "grid": lax_friedrichs_grid,
    "initial_condition": initial_condition,
}
lax_friedrichs = numerical_solution.BuckleyLeverett(params)
analytical = analytical_solution.BuckleyLeverett(params)


#######################
#### True solution ####
#######################
_, solution_func = analytical.concave_hull()
params.update(
    {
        "formulation": "n_pressure_w_saturation",
        "grid size": DEFAULT_NUM_GRID_CELLS,
        "phys size": DEFAULT_PHYS_SIZE,
        # Negative influx since the sizes are switched.
        "influx": -INFLUX,
    }
)
model = BuckleyLeverett_Analytics(params)
model.create_grid()

# Axis on the true solution are switched. Interpolate to fix this.
yy = np.linspace(1 - model._residual_saturation_w, model._residual_saturation_n, 500)
xx = solution_func(yy)
# Extend true solution on both sides with residual saturations.
xx_full = np.concatenate(
    (np.linspace(-20, xx[0], 200), xx, np.linspace(xx[-1], 20, 200))
)
yy_full = np.concatenate(
    (
        np.full(200, 1 - model._residual_saturation_n),
        yy,
        np.full(200, model._residual_saturation_w),
    )
)
solution_func_switched = interpolate.interp1d(xx_full, yy_full)

# The cell centers go from 0 to 20 instead of -10 to 10. Fix this.
adjusted_cell_centers = model.mdg.subdomains()[0].cell_centers[0, :] - 10
true_solution = solution_func_switched(adjusted_cell_centers)

# Time step fulfilling the CFL condition for the default grid cells number.
DEFAULT_MAX_TIME_STEP = lax_friedrichs.cfl_condition()


def convergence_analysis(
    params_list: list[float],
    param_type: Literal["time_step_size", "grid_size"],
    true_solution,
) -> None:
    """Run the model for all given parameters and create convergence analysis"""
    # Initiate statistics lists. Each list entry corresponds to one parameter value.
    final_errors: list[float] = []
    # Error w.r.t. to anayltical solution after the final iteration.
    residuals_list: list[list[list[float]]] = []  # Residuals for each time step.
    newton_iterations_list: list[list[int]] = []
    # Number of Newton iterations for each time step.

    # Run for variable params
    for param in params_list:
        # Set up default model params.
        # The model runs until time :math:`t=10`. We let the  model run for so long,
        # s.t. it makes sense to compare to the analytical solution. The shockwaves take
        # some time to develop. Furthermore, at the moment the model runs slower by a
        # factor of 10 for some reason.
        params.update(
            {
                "formulation": "n_pressure_w_saturation",
                "folder_name": foldername,
                "time step": DEFAULT_MAX_TIME_STEP,
                "schedule": np.array(
                    [
                        0,
                        1 + (DEFAULT_MAX_TIME_STEP - 1.0 % DEFAULT_MAX_TIME_STEP),
                    ]
                ),
                "grid size": DEFAULT_NUM_GRID_CELLS,
                "phys size": DEFAULT_PHYS_SIZE,
                # Negative influx since the sizes are switched.
                "influx": -INFLUX,
            }
        )
        # Set up variable model params.
        if param_type == "time_step_size":
            filename = f"time_step_size_{param}"
            params.update(
                {
                    "time step": param,
                    "schedule": np.array([0, 1 + (param - 1.0 % param)]),
                    "file_name": filename,
                }
            )
        elif param_type == "grid_size":
            filename = f"grid_size_{DEFAULT_PHYS_SIZE/param}"
            params.update({"grid size": param, "file_name": filename})

        # Set up logging file. Exchange file handler first.
        try:
            logger.removeHandler(fh)  # type: ignore
        except Exception:
            pass
        log_filename = os.path.join(foldername, filename + "_log.txt")
        fh = logging.FileHandler(filename=log_filename)
        logger.addHandler(fh)

        model = BuckleyLeverett_Analytics(params)
        model.prepare_simulation()

        # Run fractional flow model for all variable parameters.
        try:
            with logging_redirect_tqdm([logger]):
                run_time_dependent_model(
                    model,
                    {
                        "nl_convergence_tol": 1e-10,
                        "max_iterations": MAX_NEWTON_ITERATIONS,
                    },
                )
        except Exception:
            pass

        # If the param is ``grid_size``, calculate the analytical solution on the new
        # grid.
        if param_type == "grid_size":
            # The cell centers go from 0 to 20 instead of -10 to 10. Fix this.
            adjusted_cell_centers = model.mdg.subdomains()[0].cell_centers[0, :] - 10
            true_solution = solution_func_switched(adjusted_cell_centers)

        # Calculate final deviance from analytical solution. and cut s.t. it can be
        # compared to the true solution.
        solution = model.equation_system.get_variable_values(
            [model._ad.pressure_n, model._ad.saturation], iterate_index=0
        )[-1 : model._grid_size - 1 : -1]
        # Normalize the error by the size of the solution vector.
        error = float(np.linalg.norm(solution - true_solution)) / np.sqrt(solution.size)
        final_errors.append(error)
        # Save residuals after each time step.
        residuals_list.append(model._residuals)
        # Save number of Newton iterations for each time step.
        newton_iterations_list.append(model._newton_iterations)

    # Plot final error for each parameter.
    plt.figure()
    plt.plot(
        params_list,
        final_errors,
        "xb-",
        label=f"default power rel. perm ({MAX_NEWTON_ITERATIONS} iter)",
    )
    plt.xlabel(r"\log_{10}(\Delta t)")
    plt.ylabel(r"\log_{10}(\|e\|)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Convergence Analysis")
    plt.legend()
    plt.savefig(os.path.join(foldername, f"accuracy_varying_{param_type}.png"))

    # Plot average number of iterations for each param.
    print()
    assert 0 not in [
        len(newton_iterations) for newton_iterations in newton_iterations_list
    ]
    average_newton_iterations = [
        sum(newton_iterations) / len(newton_iterations)
        for newton_iterations in newton_iterations_list
        if len(newton_iterations_list)
    ]
    plt.figure()
    plt.plot(
        params_list,
        average_newton_iterations,
        "xb-",
        label=f"default power rel. perm",
    )
    plt.xlabel(r"\log_{10}(\Delta t)")
    plt.ylabel(r"n_{iterations}")
    plt.xscale("log")
    plt.title("average Newton iterations")
    plt.legend()
    plt.savefig(
        os.path.join(foldername, f"average_newton_iterations_varying_{param_type}.png")
    )

    # Save all statistics  in a json file. Quite complicated dict/list comprehension.
    statistics = [
        {
            "time_step_size": time_step_size,
            "final_error": final_error,
            "time_steps": [
                {
                    "index": i,
                    "final_newton_iteration": newton_iteration,
                    "residuals": residuals,
                }
                for i, (newton_iteration, residuals) in enumerate(
                    zip(newton_iterations, residuals_per_time_step)
                )
            ],
        }
        for time_step_size, final_error, newton_iterations, residuals_per_time_step in zip(
            time_step_sizes, final_errors, newton_iterations_list, residuals_list
        )
    ]

    with open(os.path.join(foldername, f"results_varying_{param_type}.json"), "w") as f:
        json.dump(statistics, f)


######################################
#### Convergence analysis in time ####
######################################
DEFAULT_MAX_TIME_STEP_LOG = math.log(DEFAULT_MAX_TIME_STEP, 10)
# time_step_sizes = np.logspace(
#     DEFAULT_MAX_TIME_STEP_LOG - 2, DEFAULT_MAX_TIME_STEP_LOG + 1, 10
# )
time_step_sizes = np.logspace(math.log(0.0005, 10), math.log(0.1, 10), 10)

# Run different time steps for a fixed number of iterations to get convergence order in
# time.
convergence_analysis(time_step_sizes, "time_step_size", true_solution)  # type:ignore


#######################################
#### Convergence analysis in space ####
#######################################
# grid_size_log = math.log(DEFAULT_NUM_GRID_CELLS, 10)
# GRID_SIZES = np.logspace(grid_size_log - 1, grid_size_log + 2, 15)

# # Run different grid sizes for a fixed number of iterations to get convergence order in
# # time.
# convergence_analysis(GRID_SIZES, "grid_size", np.zeros(1))  # type:ignore
