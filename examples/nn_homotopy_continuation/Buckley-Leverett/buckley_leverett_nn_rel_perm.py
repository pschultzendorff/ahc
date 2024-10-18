"""Analyze convergence of Buckley-Leverett with machine learned rel. perms.

An analytical solution is not avaible, hence only the residuals are analyzed.

"""

import logging
import math
import os

import numpy as np
import porepy as pp
from tpf_lab.applications.convergence_analysis import (
    BuckleyLeverettSaveData,
    ConvergenceAnalysisExtended,
    save_convergence_results,
)
from tpf_lab.models.buckley_leverett import (
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettDataSaving,
    BuckleyLeverettDefaultGeometry,
    BuckleyLeverettEquations,
    BuckleyLeverettSolutionStrategy,
    VariablesTPF,
    VerificationUtils,
)
from tpf_lab.models.rel_perm import RelPermNNEquations, RelPermNNSolutionStrategy


class Setup(
    BuckleyLeverettEquations,
    RelPermNNEquations,
    VariablesTPF,
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettSolutionStrategy,
    RelPermNNSolutionStrategy,
    BuckleyLeverettDefaultGeometry,
    BuckleyLeverettDataSaving,
    VerificationUtils,
): ...


# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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

# Parameters for machine learned rel. perm.
REL_PERM_W_NN_PARAMS: dict = {"depth": 5, "final_act": "linear"}
REL_PERM_W_NN_PATH: str = os.path.join("results", "rel_perm_nn", "wetting.pt")
REL_PERM_N_NN_PARAMS: dict = {"depth": 7, "final_act": "linear"}
REL_PERM_N_NN_PATH: str = os.path.join("results", "rel_perm_nn", "nonwetting.pt")
LIMIT_REL_PERM = False


INFLUX = 1.0
ANGLE = math.pi / 4

# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "buckley_leverett",
    "order_of_accuracy",
    f"nn_rel_perm_limited_{LIMIT_REL_PERM}",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
)
try:
    os.makedirs(foldername)
except Exception:
    pass

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
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 1]),
        dt_init=0.1,
        constant_dt=True,
    ),
    # fluid and solid params
    "porosity": POROSITY,
    "viscosity_w": VISCOSITY_W,
    "viscosity_n": VISCOSITY_N,
    "density_w": DENSITY_W,
    "density_n": DENSITY_N,
    "residual_saturation_w": RESIDUAL_SATURATION_W,
    "residual_saturation_n": RESIDUAL_SATURATION_N,
    # rel. perm model
    "rel_perm_w_nn_path": REL_PERM_W_NN_PATH,
    "rel_perm_w_nn_params": REL_PERM_W_NN_PARAMS,
    "rel_perm_n_nn_path": REL_PERM_N_NN_PATH,
    "rel_perm_n_nn_params": REL_PERM_N_NN_PARAMS,
    "limit_rel_perm": LIMIT_REL_PERM,
    # Buckley-Leverett params
    "angle": ANGLE,
    "influx": INFLUX,
}

##############################
# Convergence analysis in time
##############################
analysis = ConvergenceAnalysisExtended(
    Setup,
    params,
    levels=3,
    temporal_refinement_rate=2,
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
    max_iterations=MAX_NEWTON_ITERATIONS,
    foldername=foldername,
)
