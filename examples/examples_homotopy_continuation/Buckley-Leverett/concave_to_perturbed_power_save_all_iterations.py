"""Analyze the Newton iterations of one time step of homotopy continuation from a
concave flow function to a perturbed Corey relative permeability model."""

import logging
import math
import os
import random

import numpy as np
import porepy as pp

from tpf_lab.models.buckley_leverett import (
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettDataSaving,
    BuckleyLeverettDefaultGeometry,
    BuckleyLeverettSemiAnalyticalSolution,
    DiagnosticsMixinExtended,
    TwoPhaseFlowVariables,
    VerificationUtils,
)
from tpf_lab.models.homotopy_continuation import (
    HomotopyContinuationRelPermSolutionStrategy,
    HomotopyContinuationRelPermEquations_ConcavetoPerturbedCorey,
)
from tpf_lab.models.rel_perm import (
    BuckleyLeverettPerturbedRelPermSolutionStrategy,
)

from tpf_lab.utils import save_params_and_run_model

# Fix seed for reproducability.
random.seed(0)
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Setup the model class
class BuckleyLeverettSetup_HomotopyContinuation_RelPerm_ConcavetoPerturbedPower(  # type: ignore
    HomotopyContinuationRelPermEquations_ConcavetoPerturbedCorey,
    TwoPhaseFlowVariables,
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettPerturbedRelPermSolutionStrategy,
    HomotopyContinuationRelPermSolutionStrategy,
    #
    BuckleyLeverettDefaultGeometry,
    #
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettDataSaving,
    VerificationUtils,
    DiagnosticsMixinExtended,
):
    ...


##########################
# Default model parameters
##########################
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

REL_PERM_MODEL = "power"
REL_PERM_LINEAR_PARAM_W = 1.0
REL_PERM_LINEAR_PARAM_N = 1.0
LIMIT_REL_PERM = False

INFLUX = 1.0
ANGLE = math.pi / 4

# Parameters for wobbly rel. perm.
YSCALES = np.linspace(0.1, 0.3, 3).tolist()
XSCALES = [20000.0] * 3
OFFSETS = np.linspace(0.4, 0.6, 3).tolist()


decay = 0.5


# Set up folder and files for logging/plots/saved time steps.
foldername = os.path.join(
    "results",
    "buckley_leverett",
    "homotopy_continuation",
    f"concave_fractional_flow_to_perturbed_power_rel_perm",
    f"rel_perm_limited_{LIMIT_REL_PERM}",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
    f"homotopy_continuation_decay_{decay}",
)

try:
    os.makedirs(foldername)
except Exception:
    pass

# Set model params.
params = {
    # General params
    "folder_name": foldername,
    "filename": "setup",
    "progressbars": True,
    "export_each_iteration": True,
    "formulation": "n_pressure_w_saturation",
    "max_iterations": MAX_NEWTON_ITERATIONS,
    # grid and time
    "meshing_arguments": {
        "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
    },
    "phys size": DEFAULT_PHYS_SIZE,
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 0.1 * (0.5**4)]),
        dt_init=0.1 * (0.5**4),
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
    "rel_perm_model": REL_PERM_MODEL,
    "rel_perm_linear_param_w": REL_PERM_LINEAR_PARAM_W,
    "rel_perm_linear_param_n": REL_PERM_LINEAR_PARAM_N,
    "limit_rel_perm": LIMIT_REL_PERM,
    "homotopy_continuation_decay": decay,
    # Wobbly
    "yscales": YSCALES,
    "xscales": XSCALES,
    "offsets": OFFSETS,
    # Buckley-Leverett params
    "angle": ANGLE,
    "influx": INFLUX,
}

###################
# Run one time step
###################
setup = BuckleyLeverettSetup_HomotopyContinuation_RelPerm_ConcavetoPerturbedPower(
    params
)
save_params_and_run_model(setup, params)
