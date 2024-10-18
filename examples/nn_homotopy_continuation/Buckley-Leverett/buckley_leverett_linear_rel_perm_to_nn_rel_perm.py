"""Analyze convergence etc. of homotopy continuation from a linear relative permeability
model to a data-driven relative permeability model (a simple ff nn that was trained on a
small noisy data set).

"""

import logging
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from buckley_leverett import grid, misc, numerical_solution
from tpf_lab.applications.convergence_analysis import (
    ConvergenceAnalysisExtended,
    save_convergence_results,
)
from tpf_lab.models.buckley_leverett import (
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettDataSaving,
    BuckleyLeverettDefaultGeometry,
    BuckleyLeverettEquations,
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettSolutionStrategy,
    DiagnosticsMixinExtended,
    VariablesTPF,
    VerificationUtils,
)
from tpf_lab.models.homotopy_continuation import (
    HomotopyContinuationRelPerm_LineartoNN_SolutionStrategy,
    HomotopyContinuationRelPermEquations_LineartoNN,
)
from tpf_lab.models.rel_perm import (
    BuckleyLeverettPerturbedRelPermSolutionStrategy,
    PerturbedRelPermFractionalFlowSympy,
)
from tpf_lab.utils import fix_seeds

# Fix seed for reproducability.
fix_seeds(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BuckleyLeverettSetup_HomotopyContinuation_RelPerm_LineartoPerturbedCorey(  # type: ignore
    BuckleyLeverettEquations,
    HomotopyContinuationRelPermEquations_LineartoNN,
    VariablesTPF,
    BuckleyLeverettBoundaryConditions,
    # Solution strategy
    BuckleyLeverettPerturbedRelPermSolutionStrategy,
    HomotopyContinuationRelPerm_LineartoNN_SolutionStrategy,
    #
    BuckleyLeverettDefaultGeometry,
    #
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettDataSaving,
    VerificationUtils,
    DiagnosticsMixinExtended,
): ...


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

REL_PERM_LINEAR_PARAM_W = 1.0
REL_PERM_LINEAR_PARAM_N = 1.0
LIMIT_REL_PERM = False

rel_perm_w_nn_path = os.path.join()
rel_perm_n_nn_path = os.path.join()

INFLUX = 1.0
ANGLE = math.pi / 4

# Parameters for wobbly rel. perm.
YSCALES = np.linspace(0.1, 0.3, 3).tolist()
XSCALES = [20000.0] * 3
OFFSETS = np.linspace(0.4, 0.6, 3).tolist()


# Set up folder and files for logging/plots/saved time steps.
base_foldername: str = os.path.join(
    "results",
    "buckley_leverett",
    "homotopy_continuation",
    f"linear_rel_perm_to_nn_rel_perm_rel_perm_limited_{LIMIT_REL_PERM}",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
)

try:
    os.makedirs(base_foldername)
except Exception:
    pass


########################################################################################
# Set up model. Map analytical solution and find Courant number with the Lax-Friedrichs
# solver.
########################################################################################
lax_friedrichs_grid = grid.create_grid(
    (XMIN, XMAX), (XMAX - XMIN) / DEFAULT_NUM_GRID_CELLS
)
initial_condition = np.full_like(lax_friedrichs_grid, RESIDUAL_SATURATION_W)
initial_condition[0 : int(DEFAULT_NUM_GRID_CELLS / 2) - 10] = 1 - RESIDUAL_SATURATION_N
initial_condition[
    int(DEFAULT_NUM_GRID_CELLS / 2) - 10 : int(DEFAULT_NUM_GRID_CELLS / 2) + 10
] = np.linspace(1 - RESIDUAL_SATURATION_N, RESIDUAL_SATURATION_W, 20)

params = {
    "folder_name": base_foldername,
    "file_name": "setup",
    "max_iterations": MAX_NEWTON_ITERATIONS,
    "progressbars": True,
    "formulation": "n_pressure_w_saturation",
    # grid and time
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
    "S_M": 1 - RESIDUAL_SATURATION_W,
    "S_m": RESIDUAL_SATURATION_N,
    "residual_saturation_w": RESIDUAL_SATURATION_W,
    "residual_saturation_n": RESIDUAL_SATURATION_N,
    # rel. perm model
    "rel_perm_model": REL_PERM_MODEL,
    "rel_perm_linear_param_w": REL_PERM_LINEAR_PARAM_W,
    "rel_perm_linear_param_n": REL_PERM_LINEAR_PARAM_N,
    "limit_rel_perm": LIMIT_REL_PERM,
    # Wobbly
    "yscales": YSCALES,
    "xscales": XSCALES,
    "offsets": OFFSETS,
    # Buckley-Leverett params
    "angle": ANGLE,
    "influx": INFLUX,
    # Lax-Friedrichs params
    "grid": lax_friedrichs_grid,
    "initial_condition": initial_condition,
    # Linear flow function. Necessary parameter for the analytical solver.
    "linear_flow": False,
}


lax_friedrichs = numerical_solution.BuckleyLeverett(params)
# Exchange flow function
lax_friedrichs.fractionalflow = PerturbedRelPermFractionalFlowSympy(params)
lax_friedrichs.lambdify()
courant_number = lax_friedrichs.cfl_condition()

model = BuckleyLeverettSetup_HomotopyContinuation_RelPerm_LineartoPerturbedCorey(params)
# Exchange flow function
model.analytical.fractionalflow = PerturbedRelPermFractionalFlowSympy(params)
model.analytical.lambdify()

model.prepare_simulation()
exact_solution = model._exact_solution
fig = plt.figure()
plt.plot(
    model.mdg.subdomains()[0].cell_centers[0] + model.domain.bounding_box["xmin"],
    exact_solution,
    label="analytical solution",
)
plt.xlabel(rf"x")
plt.ylabel(rf"S_w")
plt.legend()
fig.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig(os.path.join(base_foldername, "analytical_solution.png"))
plt.close()
misc.map_fractional_flow(
    model.analytical,
    filename=os.path.join(base_foldername, "analytical_solution"),
)


#####################
# Analyze decay rates
#####################
decays = np.linspace(0.3, 0.7, 5)
for decay in decays:
    # Set up folder and files for logging/plots/saved time steps.
    foldername = os.path.join(base_foldername, f"homotopy_continuation_decay_{decay}")

    try:
        os.makedirs(foldername)
    except Exception:
        pass

    params.update(
        {
            # Base folder and file name. These will get changed by
            # ``ConvergenceAnalysisExtended``.
            "folder_name": foldername,
            "file_name": "setup",
            "homotopy_continuation_decay": decay,
        }
    )

    analysis = ConvergenceAnalysisExtended(
        BuckleyLeverettSetup_HomotopyContinuation_RelPerm_LineartoPerturbedCorey,
        params,
        levels=7,
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
        courant_number=courant_number,
        max_iterations=MAX_NEWTON_ITERATIONS,
        foldername=foldername,
    )
