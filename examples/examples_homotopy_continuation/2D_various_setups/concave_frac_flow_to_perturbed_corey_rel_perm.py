"""Analyze convergence etc. of homotopy continuation from a concave fractional flow
model to a perturbed Corey relative permeability model.

The easiest way to get a concave fractional flow model is to have a linear nonwetting
relative permeability and a polynomial wetting relative permeability.

In comparison to e.g., ``linear_to_perturbed_power_1.py``, the perturbations are less
in this script.

The function is analyzed on the 2D test case in section 3.1. of "Unconditionally
convergent nonlinear solver for hyperbolic conservation laws with S-shaped flux
functions" [Jenny et. al, 2009].

"""

import logging
import os
import random

import numpy as np
import porepy as pp

from porepy.utils.examples_utils import VerificationUtils
from tpf_lab.applications.convergence_analysis import (
    ConvergenceAnalysisExtended,
    save_convergence_results,
)
from tpf_lab.models.two_phase_flow import (
    TwoPhaseFlowEquations,
    TwoPhaseFlowBoundaryConditions,
    TwoPhaseFlowVariables,
)
from tpf_lab.models.rel_perm import (
    PerturbedRelPermSolutionStrategy,
)
from tpf_lab.models.homotopy_continuation import (
    HomotopyContinuationRelPermEquations_ConcavetoPerturbedCorey,
    HomotopyContinuationRelPermSolutionStrategy,
)
from tpf_lab.visualization.diagnostics import TwoPhaseFlowDataSaving

# Fix seed for reproducability.
random.seed(0)
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ModifiedGeometry(pp.ModelGeometry):
    def set_domain(self) -> None:
        r"""Set domain of the problem.

        The domain is :math:`\Omega=[0,120]\times[0,60]`.

        """
        bounding_box: dict[str, pp.number] = {
            "xmin": 0,
            "xmax": 120,
            "ymin": 0,
            "ymax": 60,
        }
        self._domain = pp.Domain(bounding_box)


class ModifiedBoundaryConditions(TwoPhaseFlowBoundaryConditions):
    ...


class ModifiedEquations(TwoPhaseFlowEquations):
    r"""Modifications to the two-phase flow model:

    Source term in the subdomain :math:`\Omega_{in}=[0,1]\times[0,1]`; sink term in
    the subdomain :math:`\Omega_{out}=[119,120]\times[59,60]`.

    Varying permeability across the domain.

    """

    def _permeability(self, g: pp.Grid) -> np.ndarray:
        # Function for base-10 log. of permeability along x-axis.
        def function_x(x: float) -> float:
            return 3 * x * (x - 0.7) * (x - 1.7)

        # Function for base-10 log. of permeability along y-axis.
        def function_y(x: float) -> float:
            return 3 * (x - 0.2) * (x - 0.8) * (x + 2)

        log_permeability = np.zeros([60, 120], dtype=float)
        for i, row in enumerate(log_permeability):
            for j, _ in enumerate(row):
                log_permeability[i, j] = function_x(i / 60.0) * function_y(j / 120.0)

        # Add noise.
        log_permeability += np.random.normal(0, 0.2, [60, 120])
        return (10**log_permeability).flatten()

    def _source_w(self, g: pp.Grid) -> np.ndarray:
        array = super()._source_w(g)
        array[0] = 1
        array[-1] = -1
        return array


class Setup(  # type: ignore
    HomotopyContinuationRelPermEquations_ConcavetoPerturbedCorey,
    ModifiedEquations,
    TwoPhaseFlowVariables,
    TwoPhaseFlowBoundaryConditions,
    # Solution strategy
    HomotopyContinuationRelPermSolutionStrategy,
    # To read perturbations from param list.
    PerturbedRelPermSolutionStrategy,
    #
    ModifiedGeometry,
    #
    TwoPhaseFlowDataSaving,
    VerificationUtils,
):
    ...


####################
# Default parameters
####################
MAX_NEWTON_ITERATIONS = 60

DEFAULT_MESHING_ARGS: dict[str, float] = {"cell_size": 1.0}

POROSITY = 0.1
VISCOSITY_W = 1.0
VISCOSITY_N = 1.0

DENSITY_W = 600.0
DENSITY_N = 800.0

RESIDUAL_SATURATION_W = 0.3
RESIDUAL_SATURATION_N = 0.3

REL_PERM_MODEL = "power"
REL_PERM_LINEAR_PARAM_W = 1.0
REL_PERM_LINEAR_PARAM_N = 1.0
LIMIT_REL_PERM = False

# Parameters for wobbly rel. perm.
YSCALES = np.linspace(0.1, 0.3, 3).tolist()
XSCALES = [20000.0] * 3
OFFSETS = np.linspace(0.4, 0.6, 3).tolist()


# Set up folder and files for logging/plots/saved time steps.
base_foldername: str = os.path.join(
    "results",
    "2D_two_phase_flow",
    "homotopy_continuation",
    f"concave_to_perturbed_rel_perm_2_limited_{LIMIT_REL_PERM}",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
)

try:
    os.makedirs(base_foldername)
except Exception:
    pass


params = {
    "max_iterations": MAX_NEWTON_ITERATIONS,
    "progressbars": True,
    "formulation": "n_pressure_w_saturation",
    # grid and time
    "meshing_arguments": DEFAULT_MESHING_ARGS,
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 10]),
        dt_init=0.01,
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
    # Perturbations
    "yscales": YSCALES,
    "xscales": XSCALES,
    "offsets": OFFSETS,
}


#####################
# Analyze decay rates
#####################
decays = np.linspace(0.3, 0.7, 5)
decays = [0.5]  # type: ignore
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
        Setup,
        params,
        levels=7,
        temporal_refinement_rate=2,
    )
    results = analysis.run_analysis()
    analysis.export_results_to_json(
        results,
        variables_to_export=[
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
