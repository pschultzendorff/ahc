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


class ModifiedBoundaryConditions(BoundaryConditionsTPF):
    """Homogeneous Neumann bc (no flow) at the top and bottom. Homogeneous Dirichlet bc
    (open boundaries) at the sides."""

    def _bc_type_pressure_w(self, g: pp.Grid) -> pp.BoundaryCondition:
        east = self.domain_boundary_sides(g).east  # type: ignore
        west = self.domain_boundary_sides(g).west  # type: ignore
        return pp.BoundaryCondition(g, np.logical_or(east, west), "dir")

    def _bc_type_pressure_n(self, g: pp.Grid) -> pp.BoundaryCondition:
        east = self.domain_boundary_sides(g).east  # type: ignore
        west = self.domain_boundary_sides(g).west  # type: ignore
        return pp.BoundaryCondition(g, np.logical_or(east, west), "dir")


class ModifiedEquations(EquationsTPF):
    r"""Modifications to the two-phase flow model:

    - Source term in the subdomain :math:`\Omega_{in}=[0,1]\times[0,1]`; sink term in
    the subdomain :math:`\Omega_{out}=[119,120]\times[59,60]`.
    - Gravity is on.

    """

    def _source_w(self, g: pp.Grid) -> np.ndarray:
        array = super()._source_w(g)
        array[60] = 1
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


class ModifiedSolutionStrategy(SolutionStrategyTPF):
    """Modifiy the initital saturation and upwind direction."""

    def set_discretization_parameters(self) -> None:
        """Initialize the upwind discretization in upstream direction of the buoyancy
        flow."""
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            # Boundary conditions and parameters
            diffusivity = pp.SecondOrderTensor(self._permeability(sd))
            # all_bf, *_ = self._domain_boundary_sides(sd)
            # Parameters that are not used for discretization.
            pp.initialize_data(
                sd,
                data,
                self.w_flux_key,
                {
                    "source_w": self._source_w(sd),
                    "bc": self._bc_type_pressure_w(sd),
                    "bc_values": self._dirichlet_bc_values_pressure_w(sd),
                    "darcy_flux": -np.ones(sd.num_faces),
                    "second_order_tensor": diffusivity,
                },
            )
            # Parameters for nonwetting phase.
            pp.initialize_data(
                sd,
                data,
                self.n_flux_key,
                {
                    "source_n": self._source_n(sd),
                    "bc": self._bc_type_pressure_n(sd),
                    "bc_values": self._dirichlet_bc_values_pressure_n(sd),
                    "darcy_flux": -np.ones(sd.num_faces),
                    "second_order_tensor": diffusivity,
                },
            )

    def initial_condition(self) -> None:
        """Set initial wetting saturation to residual saturation."""
        super().initial_condition()
        self.equation_system.set_variable_values(
            np.full(
                self.mdg.subdomains()[0].num_cells, self._residual_saturation_w + 0.01
            ),
            [self.saturation_var],
            time_step_index=self.time_manager.time_index,
        )


class Setup(  # type: ignore
    ModifiedEquations,
    VariablesTPF,
    ModifiedBoundaryConditions,
    # Solution strategy
    ModifiedSolutionStrategy,
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

DENSITY_W = 1.0
DENSITY_N = 2.0

RESIDUAL_SATURATION_W = 0.3
RESIDUAL_SATURATION_N = 0.3

REL_PERM_MODEL = "power"
REL_PERM_LINEAR_PARAM_W = 1.0
REL_PERM_LINEAR_PARAM_N = 1.0
LIMIT_REL_PERM = False

CAP_PRESS_MODEL = "linear"
CAP_PRESS_LINEAR_PARAM = 0.1


# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "2D_two_phase_flow",
    "convergence_analysis",
    f"Corey_rel_perm_limited_{LIMIT_REL_PERM}",
    f"_cap_pressure_{CAP_PRESS_MODEL}_homogeneous_permeability",
    f"_gravity_on_density_w_{int(DENSITY_W)}_density_n_{int(DENSITY_N)}",
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
    # grid and time
    "meshing_arguments": DEFAULT_MESHING_ARGS,
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 30]),
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
    levels=3,
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
