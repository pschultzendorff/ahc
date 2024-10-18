"""Analyze convergence etc. with the default Corey relative permeability model.

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
    BoundaryConditionsTPF,
    ConstitutiveLawsTPF,
    EquationsTPF,
    SolutionStrategyTPF,
    VariablesTPF,
)
from tpf_lab.visualization.diagnostics import DataSavingTwoPhaseFlow

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


class ModifiedEquations(EquationsTPF):
    r"""Modifications to the two-phase flow model:

    - Source term in the subdomain :math:`\Omega_{in}=[0,1]\times[0,1]`; sink term in
    the subdomain :math:`\Omega_{out}=[119,120]\times[59,60]`.
    - Gravity is on.
    - Varying permeability across the domain.

    """

    def _permeability(self, g: pp.Grid) -> np.ndarray:
        # Function for base-10 log. of permeability along x-axis.
        def function_x(x: float) -> float:
            return 3 * x * (x - 0.7) * (x - 2.3)

        # Function for base-10 log. of permeability along y-axis.
        def function_y(x: float) -> float:
            return 5 * (x - 0.2) * (x - 0.8) * (x + 2)

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

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting vector source. Corresponds to the wetting buoyancy flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[..., -1] = pp.GRAVITY_ACCELERATION * self._w_density

        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[..., -1] = pp.GRAVITY_ACCELERATION * self._density_w
        # For some reason this needs to be a flat array.
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric nonwetting vector source. Corresponds to the nonwetting buoyancy
        flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[..., -1] = pp.GRAVITY_ACCELERATION * self._n_density

        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[..., -1] = pp.GRAVITY_ACCELERATION * self._density_n
        # For some reason this needs to be a flat array.
        return vals.ravel()


class Setup(  # type: ignore
    ModifiedEquations,
    VariablesTPF,
    ConstitutiveLawsTPF,
    BoundaryConditionsTPF,
    # Solution strategy
    SolutionStrategyTPF,
    #
    ModifiedGeometry,
    #
    DataSavingTwoPhaseFlow,
    VerificationUtils,
): ...


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

CAP_PRESS_MODEL = None
CAP_PRESS_LINEAR_PARAM = 0.1


# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "2D_two_phase_flow",
    "convergence_analysis",
    f"Corey_rel_perm_limited_{LIMIT_REL_PERM}_cap_pressure_{CAP_PRESS_MODEL}_gravity_on",
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
    "formulation": "fractional_flow",
    # grid and time
    "meshing_arguments": DEFAULT_MESHING_ARGS,
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 20]),
        dt_init=1.0,
        constant_dt=True,
    ),
    # fluid and solid params
    # TODO: Change this to MaterialParams
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
    # capillary pressure model
    "cap_pressure_model": CAP_PRESS_MODEL,
    "cap_pressure_linear_param": CAP_PRESS_LINEAR_PARAM,
}


#####################
# Analyze convergence
#####################
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
