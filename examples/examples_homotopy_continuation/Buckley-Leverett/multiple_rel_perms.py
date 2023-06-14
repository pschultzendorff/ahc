import logging
import math
import os
import random
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from buckley_leverett import grid, misc, numerical_solution

from tpf_lab.applications.convergence_analysis import (
    ConvergenceAnalysisExtended,
    save_convergence_results,
)
from tpf_lab.models.buckley_leverett import (
    BuckleyLeverettSetup,
)
from tpf_lab.models.rel_perm import (
    PerturbedRelPermFractionalFlowSympy,
    BuckleyLeverettPerturbedRelPermSetup,
)

# Fix seed for reproducability.
random.seed(0)
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# LaTeX
plt.rcParams.update(
    {
        "text.latex.preamble": r"\usepackage{lmodern}",
        "text.usetex": True,
        "font.size": 16,
    }
)


####################
# Default parameters
####################
MAX_NEWTON_ITERATIONS = 60

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

REL_PERM_MODEL = "linear"
REL_PERM_LINEAR_PARAM_W = 1.0
REL_PERM_LINEAR_PARAM_N = 1.0
LIMIT_REL_PERM = False

INFLUX = 1.0
ANGLE = math.pi / 4

# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "buckley_leverett",
    "order_of_accuracy",
    f"linear_rel_perm_limited_{LIMIT_REL_PERM}",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
)
try:
    os.makedirs(foldername)
except Exception:
    pass


####################
# Linear rel. perms.
####################
lax_friedrichs_grid = grid.create_grid(
    (XMIN, XMAX), (XMAX - XMIN) / DEFAULT_NUM_GRID_CELLS
)
initial_condition = np.full_like(lax_friedrichs_grid, RESIDUAL_SATURATION_W)
initial_condition[0 : int(DEFAULT_NUM_GRID_CELLS / 2) - 10] = 1 - RESIDUAL_SATURATION_N
initial_condition[
    int(DEFAULT_NUM_GRID_CELLS / 2) - 10 : int(DEFAULT_NUM_GRID_CELLS / 2) + 10
] = np.linspace(1 - RESIDUAL_SATURATION_N, RESIDUAL_SATURATION_W, 20)

params = {
    # Base folder and file name. These will get changed by
    # ``ConvergenceAnalysisExtended``.
    "folder_name": foldername,
    "file_name": "setup",
    "max_iterations": MAX_NEWTON_ITERATIONS,
    "progressbars": True,
    "formulation": "n_pressure_w_saturation",
    # grid
    "meshing_arguments": {
        "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
    },
    "phys size": DEFAULT_PHYS_SIZE,
    # fluid and solid params
    "porosity": POROSITY,
    "viscosity_w": VISCOSITY_W,
    "viscosity_n": VISCOSITY_N,
    "density_w": DENSITY_W,
    "density_n": DENSITY_N,
    "S_M": 1 - RESIDUAL_SATURATION_W,
    "S_m": RESIDUAL_SATURATION_N,
    "residual_saturation_w": RESIDUAL_SATURATION_W,
    "residual_saturation_n": RESIDUAL_SATURATION_N,
    # rel. perm model
    "rel_perm_model": REL_PERM_MODEL,
    "rel_perm_linear_param_w": REL_PERM_LINEAR_PARAM_W,
    "rel_perm_linear_param_n": REL_PERM_LINEAR_PARAM_N,
    "limit_rel_perm": LIMIT_REL_PERM,
    # Buckley-Leverett params
    "angle": ANGLE,
    "influx": INFLUX,
    # Lax-Friedrichs params
    "grid": lax_friedrichs_grid,
    "initial_condition": initial_condition,
    # Linear flow function. Necessary parameter for the analytical solver.
    "linear_flow": True,
}
# lax_friedrichs = numerical_solution.BuckleyLeverett(params)
# # Time step fulfilling the CFL condition for the default grid cells number.
# COURANT_NUMBER = lax_friedrichs.cfl_condition()

# model = BuckleyLeverettSetup(params)
# model.prepare_simulation()
# exact_solution = model._exact_solution
# fig = plt.figure()
# plt.plot(
#     model.mdg.subdomains()[0].cell_centers[0] + model.domain.bounding_box["xmin"],
#     exact_solution,
#     label="analytical solution",
# )
# plt.xlabel(rf"$x$")
# plt.ylabel(rf"$S_w$")
# plt.legend()
# fig.subplots_adjust(left=0.2, bottom=0.2)
# plt.savefig(os.path.join(foldername, "analytical_solution.png"))
# plt.close()
# misc.map_fractional_flow(
#     model.analytical,
#     filename=os.path.join(foldername, "analytical_solution"),
# )

##############################
# Convergence analysis in time
##############################
# params.update(
#     {
#         "meshing_arguments": {
#             "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
#         },
#         "time_manager": pp.TimeManager(
#             schedule=np.array([0, 1]),
#             dt_init=0.1,
#             constant_dt=True,
#         ),
#     }
# )

# analysis = ConvergenceAnalysisExtended(
#     BuckleyLeverettSetup, params, levels=7, temporal_refinement_rate=2
# )
# results = analysis.run_analysis()
# analysis.export_results_to_json(
#     results,
#     variables_to_export=[
#         "l2_error",
#         "residuals",
#         "iteration_counter",
#         "time",
#         "time_index",
#     ],
#     file_name="temporal_error_analysis.json",
# )
# save_convergence_results(
#     analysis,
#     results,
#     "time",
#     courant_number=COURANT_NUMBER,
#     max_iterations=MAX_NEWTON_ITERATIONS,
#     foldername=foldername,
# )

###############################
# Convergence analysis in space
###############################
# params.update(
#     {
#         "meshing_arguments": {"cell_size": DEFAULT_PHYS_SIZE / 50.0},
#         "time_step_size": COURANT_NUMBER,
#     }
# )

# analysis = ConvergenceAnalysisExtended(
#     BuckleyLeverettSetup, params, levels=6, spatial_refinement_rate=2
# )
# results = analysis.run_analysis()
# analysis.export_results_to_json(
#     results,
#     variables_to_export=["l2_error", "residuals", "iteration_counter", "time", "time_index",],
#     file_name="temporal_error_analysis.json",
# )

# save_convergence_results(analysis, results, "space")

###################
# Power rel. perms.
###################
# Set up folder and files for logging/plots/saved time steps.
foldername = os.path.join(
    "results",
    "buckley_leverett",
    "order_of_accuracy",
    f"Corey_rel_perm_limited_{LIMIT_REL_PERM}",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
)
try:
    os.makedirs(foldername)
except Exception:
    pass

# Change model params.
REL_PERM_MODEL = "power"
params.update(
    {"rel_perm_model": REL_PERM_MODEL, "linear_flow": False, "folder_name": foldername}
)


lax_friedrichs = numerical_solution.BuckleyLeverett(params)
# Time step fulfilling the CFL condition for the default grid cells number.
COURANT_NUMBER = lax_friedrichs.cfl_condition()

model = BuckleyLeverettSetup(params)
model.prepare_simulation()
exact_solution = model._exact_solution
fig = plt.figure()
plt.plot(
    model.mdg.subdomains()[0].cell_centers[0] + model.domain.bounding_box["xmin"],
    exact_solution,
    label="analytical solution",
)
plt.xlabel(rf"$x$")
plt.ylabel(rf"$S_w$")
plt.legend()
fig.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig(os.path.join(foldername, "analytical_solution.png"))
plt.close()
misc.map_fractional_flow(
    model.analytical,
    filename=os.path.join(foldername, "analytical_solution"),
)

##############################
# Convergence analysis in time
##############################
params.update(
    {
        "meshing_arguments": {
            "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
        },
        "time_manager": pp.TimeManager(
            schedule=np.array([0, 1]),
            dt_init=1.0,
            constant_dt=True,
        ),
    }
)

analysis = ConvergenceAnalysisExtended(
    BuckleyLeverettSetup, params, levels=4, temporal_refinement_rate=2
)
results = analysis.run_analysis()
analysis.export_results_to_json(
    results,
    variables_to_export=[
        "l2_error",
        "residuals",
        "iteration_counter",
        "time",
        "time_index",
    ],
    file_name="temporal_error_analysis.json",
)
save_convergence_results(
    analysis,
    results,
    "time",
    courant_number=COURANT_NUMBER,
    max_iterations=MAX_NEWTON_ITERATIONS,
    foldername=foldername,
)

###############################
# Convergence analysis in space
###############################
# params.update(
#     {
#         "meshing_arguments": {"cell_size": DEFAULT_PHYS_SIZE / 50.0},
#         "time_step_size": COURANT_NUMBER,
#     }
# )

# analysis = ConvergenceAnalysisExtended(
#     BuckleyLeverettSetup, params, levels=6, spatial_refinement_rate=2
# )
# results = analysis.run_analysis()
# analysis.export_results_to_json(
#     results,
#     variables_to_export=["l2_error", "residuals", "iteration_counter", "time", "time_index",],
#     file_name="spatial_error_analysis.json",
# )

# save_convergence_results(analysis, results, "space")


####################
# Wobbyl rel. perms.
####################
# Parameters for wobbly rel. perm.
# YSCALES = np.maximum(np.random.rand(20), 0.5).tolist()
# YSCALES[:5] = np.arange(0.1, 0.5, 0.1)
# XSCALES = [20000.0] * 20
# OFFSETS = np.linspace(0.4, 0.6, 20).tolist()
# REL_PERM_MODEL = "power"

# # Set up folder and files for logging/plots/saved time steps.
# foldername = os.path.join(
#     "results",
#     "buckley_leverett",
#     "order_of_accuracy",
#     f"wobbly_rel_perm_limited_{LIMIT_REL_PERM}",
#     f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
# )
# try:
#     os.makedirs(foldername)
# except Exception:
#     pass

# # Change model params.
# params.update(
#     {
#         "yscales": YSCALES,
#         "xscales": XSCALES,
#         "offsets": OFFSETS,
#         "linear_flow": False,
#         "rel_perm_model": REL_PERM_MODEL,
#         "folder_name": foldername,
#     }
# )


# lax_friedrichs = numerical_solution.BuckleyLeverett(params)
# lax_friedrichs.fractionalflow = WobblyFractionalFlowSympy(params)
# lax_friedrichs.lambdify()
# # Time step fulfilling the CFL condition for the default grid cells number.
# COURANT_NUMBER = lax_friedrichs.cfl_condition()

# model = BuckleyLeverettPerturbatedRelPermSetup(params)

# model.prepare_simulation()
# exact_solution = model._exact_solution
# fig = plt.figure()
# plt.plot(
#     model.mdg.subdomains()[0].cell_centers[0] + model.domain.bounding_box["xmin"],
#     exact_solution,
#     label="analytical solution",
# )
# plt.xlabel(rf"$x$")
# plt.ylabel(rf"$S_w$")
# plt.legend()
# fig.subplots_adjust(left=0.2, bottom=0.2)
# plt.savefig(os.path.join(foldername, "analytical_solution.png"))
# plt.close()
# misc.map_fractional_flow(
#     model.analytical,
#     filename=os.path.join(foldername, "analytical_solution"),
# )

##############################
# Convergence analysis in time
##############################
# params.update(
#     {
#         "folder_name": foldername,
#         "meshing_arguments": {
#             "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
#         },
#         "time_manager": pp.TimeManager(
#             schedule=np.array([0, 1]),
#             dt_init=0.1,
#             constant_dt=True,
#         ),
#     }
# )

# analysis = ConvergenceAnalysisExtended(
#     BuckleyLeverettPerturbatedRelPermSetup, params, levels=7, temporal_refinement_rate=2
# )
# results = analysis.run_analysis()
# analysis.export_results_to_json(
#     results,
#     variables_to_export=[
#         "l2_error",
#         "residuals",
#         "iteration_counter",
#         "time",
#         "time_index",
#     ],
#     file_name="temporal_error_analysis.json",
# )
# save_convergence_results(
#     analysis,
#     results,
#     "time",
#     courant_number=COURANT_NUMBER,
#     max_iterations=MAX_NEWTON_ITERATIONS,
#     foldername=foldername,
# )

###############################
# Convergence analysis in space
###############################
# params.update(
#     {
#         "meshing_arguments": {"cell_size": DEFAULT_PHYS_SIZE / 50.0},
#         "time_step_size": COURANT_NUMBER,
#     }
# )

# analysis = ConvergenceAnalysisExtended(
#     BuckleyLeverettPerturbatedRelPermSetup, params, levels=6, spatial_refinement_rate=2
# )
# results = analysis.run_analysis()
# analysis.export_results_to_json(
#     results,
#     variables_to_export=["l2_error", "residuals", "iteration_counter", "time", "time_index",],
#     file_name="temporal_error_analysis.json",
# )
# ooc = analysis.order_of_convergence(results, [])

# save_convergence_results(analysis, results, "space")
