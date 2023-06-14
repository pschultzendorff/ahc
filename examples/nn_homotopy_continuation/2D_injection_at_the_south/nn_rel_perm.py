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
from tpf_lab.models.two_phase_flow import (
    TwoPhaseFlowBoundaryConditions,
    TwoPhaseFlowDataSaving,
    TwoPhaseFlowSolutionStrategy,
    TwoPhaseFlowEquations,
    VerificationUtils,
    TwoPhaseFlowVariables,
)
from tpf_lab.models.rel_perm import (
    RelPermNNEquations,
    RelPermNNSolutionStrategy,
)


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


class ModifiedEquations(TwoPhaseFlowEquations):
    r"""Modifications to the two-phase flow model:

    - Source term in the subdomain :math:`\Omega_{in}=[0,1]\times[0,1]`; sink term in
    the subdomain :math:`\Omega_{out}=[119,120]\times[59,60]`.
    - Gravity is on.
    - Varying permeability across the domain.

    """

    # def _permeability(self, g: pp.Grid) -> np.ndarray:
    #     # Function for base-10 log. of permeability along x-axis.
    #     def function_x(x: float) -> float:
    #         return 3 * x * (x - 0.7) * (x - 2.3)

    #     # Function for base-10 log. of permeability along y-axis.
    #     def function_y(x: float) -> float:
    #         return 5 * (x - 0.2) * (x - 0.8) * (x + 2)

    #     log_permeability = np.zeros([60, 120], dtype=float)
    #     for i, row in enumerate(log_permeability):
    #         for j, _ in enumerate(row):
    #             log_permeability[i, j] = function_x(i / 60.0) * function_y(j / 120.0)

    #     # Add noise.
    #     log_permeability += np.random.normal(0, 0.2, [60, 120])
    #     return (10**log_permeability).flatten()

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


class ModifiedSolutionStrategy(TwoPhaseFlowSolutionStrategy):
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


class Setup(
    ModifiedEquations,
    RelPermNNEquations,
    TwoPhaseFlowVariables,
    TwoPhaseFlowBoundaryConditions,
    # Solution strategy
    ModifiedSolutionStrategy,
    RelPermNNSolutionStrategy,
    #
    ModifiedGeometry,
    #
    TwoPhaseFlowDataSaving,
    VerificationUtils,
):
    ...


# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


####################
# Default parameters
####################
MAX_NEWTON_ITERATIONS = 60
N1_CONVERGENCE_TOL: float = 1e-9

CELL_SIZE = 1.0

POROSITY = 0.1
VISCOSITY_W = 1.0
VISCOSITY_N = 1.0
DENSITY_W = 1.0
DENSITY_N = 1.2

RESIDUAL_SATURATION_W = 0.3
RESIDUAL_SATURATION_N = 0.3

# Parameters for machine learned rel. perm.
REL_PERM_W_NN_PARAMS: dict = {"depth": 5, "final_act": "linear"}
REL_PERM_W_NN_PATH: str = os.path.join("results", "rel_perm_nn", "wetting.pt")
REL_PERM_N_NN_PARAMS: dict = {"depth": 7, "final_act": "linear"}
REL_PERM_N_NN_PATH: str = os.path.join("results", "rel_perm_nn", "nonwetting.pt")
LIMIT_REL_PERM = False

CAP_PRESSURE_MODEL = "linear"
CAP_PRESSURE_LINEAR_PARAM = 0.1


# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "2D_injection_at_the_south",
    f"nn_rel_perm",
    "homogeneous_permeability",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
    f"n1_convergence_tol_{N1_CONVERGENCE_TOL}",
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
    "nl_convergence_tol": N1_CONVERGENCE_TOL,
    "progressbars": True,
    "formulation": "n_pressure_w_saturation",
    # grid
    "meshing_arguments": {"cell_size": CELL_SIZE},
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
    # cap. pressure model
    "cap_pressure_model": CAP_PRESSURE_MODEL,
    "cap_pressure_linear_param": CAP_PRESSURE_LINEAR_PARAM,
}

##############################
# Convergence analysis in time
##############################
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
        "solution_norms",
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
