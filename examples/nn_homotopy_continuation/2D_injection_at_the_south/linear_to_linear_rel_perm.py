"""Sanity check that nothing is wrong with the homotopy continuation. This should
produce the same results as a linear rel. perm. model."""

import logging
import os
from typing import Any, Optional

import numpy as np
import porepy as pp
from tpf_lab.applications.convergence_analysis import (
    ConvergenceAnalysisExtended,
    save_convergence_results,
)
from tpf_lab.models.homotopy_continuation import (
    HomotopyContinuationRelPermEquations_LineartoNN,
    HomotopyContinuationRelPermSolutionStrategy,
)
from tpf_lab.models.rel_perm import RelPermNNSolutionStrategy
from tpf_lab.models.two_phase_flow import (
    BoundaryConditionsTPF,
    DataSavingTwoPhaseFlow,
    EquationsTPF,
    SolutionStrategyTPF,
    VariablesTPF,
    VerificationUtils,
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


class ModifiedEquations(EquationsTPF):
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


class ModifiedSolutionStrategy(SolutionStrategyTPF):
    """Modifiy the initital saturation and upwind direction."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Parameters for the homotopy continuation.
        self._homotopy_continuation_param: float = 1
        self._homotopy_continuation_param_ad: pp.ad.Scalar = pp.ad.Scalar(
            self._homotopy_continuation_param
        )
        self._homotopy_continuation_param_min: float = params.get(
            "homotopy_continuation_param_min", 0.0
        )
        self._homotopy_continuation_decay: float = params.get(
            "homotopy_continuation_decay", 0.5
        )
        self.residuals_wrt_homotopy: list[float] = []
        """Store the residuals of the equation w.r.t. the homotopy."""
        self.residuals_wrt_goal_function: list[float] = []
        """Store the residuals of the equation w.r.t. the goal function, i.e., w.r.t.
        :math:`\lambda=0`."""

    def before_nonlinear_loop(self) -> None:
        # Reset continuation parameter.
        self._homotopy_continuation_param = 1
        # Reset residuals arrays.
        self.residuals_wrt_homotopy = []
        self.residuals_wrt_goal_function = []
        # Update ad homotopy continuation parameter.
        setattr(
            self._homotopy_continuation_param_ad,
            "_value",
            self._homotopy_continuation_param,
        )
        return super().before_nonlinear_loop()

    def before_nonlinear_iteration(self) -> None:
        return super().before_nonlinear_iteration()

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

    def after_nonlinear_iteration(self, solution: np.ndarray) -> None:
        # Compute residual (PorePys residual is actually the norm of the solution).
        b = self.linear_system[1]
        self.residuals_wrt_homotopy.append(float(np.linalg.norm(b)) / np.sqrt(b.size))
        return super().after_nonlinear_iteration(solution)

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[float, bool, bool]:
        """Extend the convergence check of the super class s.t. it fails when only one
        nonlinear iteration has passed.

        This is to ensure, that the homotopy continuation problem gets solved instead of
        the problem at :math:`\lambda=1`, i.e. the initial problem of the homotopy
        continuation.

        Parameters:
            solution: Newly obtained solution vector prev_solution: Solution obtained in
            the previous non-linear iteration. init_solution: Solution obtained from the
            previous time-step. nl_params: Dictionary of parameters used for the
            convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            The method returns the following tuple:

            float:
                Error, computed to the norm in question.
            boolean:
                True if the solution is converged according to the test implemented by
                this method.
            boolean:
                True if the solution is diverged according to the test implemented by
                this method.

        """
        error, converged, diverged = super().check_convergence(
            solution, prev_solution, init_solution, nl_params
        )
        if self._nonlinear_iteration == 1:
            converged = False
        return error, converged, diverged


class Setup(
    ModifiedEquations,
    HomotopyContinuationRelPermEquations_LineartoNN,
    VariablesTPF,
    BoundaryConditionsTPF,
    # Solution strategy
    ModifiedSolutionStrategy,
    RelPermNNSolutionStrategy,
    # HomotopyContinuationRelPermSolutionStrategy,
    #
    ModifiedGeometry,
    #
    DataSavingTwoPhaseFlow,
    VerificationUtils,
): ...


# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


####################
# Default parameters
####################
# Solver setup
MAX_NEWTON_ITERATIONS = 60
N1_CONVERGENCE_TOL: float = 1e-5

CELL_SIZE = 1.0

POROSITY = 0.1
VISCOSITY_W = 1.0
VISCOSITY_N = 1.0
DENSITY_W = 1.0
DENSITY_N = 1.2

RESIDUAL_SATURATION_W = 0.3
RESIDUAL_SATURATION_N = 0.3

# Parameters for linear rel. perm.
REL_PERM_LINEAR_PARAM_W = 1.0
REL_PERM_LINEAR_PARAM_N = 1.0

# Parameters for machine learned rel. perm.
REL_PERM_W_NN_PARAMS: dict = {"depth": 5, "final_act": "linear"}
REL_PERM_W_NN_PATH: str = os.path.join("results", "rel_perm_nn", "wetting.pt")
REL_PERM_N_NN_PARAMS: dict = {"depth": 7, "final_act": "linear"}
REL_PERM_N_NN_PATH: str = os.path.join("results", "rel_perm_nn", "nonwetting.pt")
LIMIT_REL_PERM = False

# Parameters for homotopy continuation
HOMOTOPY_CONTINUATION_PARAM_MIN: float = 1.0
HOMOTOPY_CONTINUATION_DECAY: float = 0.5

CAP_PRESSURE_MODEL = "linear"
CAP_PRESSURE_LINEAR_PARAM = 0.1


# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "2D_injection_at_the_south",
    f"linear_to_linear_rel_perm",
    "homogeneous_permeability",
    f"homotopy_cont_decay_{HOMOTOPY_CONTINUATION_DECAY}",
    f"homotopy_cont_param_min_{HOMOTOPY_CONTINUATION_PARAM_MIN}",
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
        schedule=np.array([0, 1.0]),
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
    "rel_perm_linear_param_w": REL_PERM_LINEAR_PARAM_W,
    "rel_perm_linear_param_n": REL_PERM_LINEAR_PARAM_N,
    "rel_perm_w_nn_path": REL_PERM_W_NN_PATH,
    "rel_perm_w_nn_params": REL_PERM_W_NN_PARAMS,
    "rel_perm_n_nn_path": REL_PERM_N_NN_PATH,
    "rel_perm_n_nn_params": REL_PERM_N_NN_PARAMS,
    "limit_rel_perm": LIMIT_REL_PERM,
    "homotopy_continuation_param_min": HOMOTOPY_CONTINUATION_PARAM_MIN,
    "homotopy_continuation_decay": HOMOTOPY_CONTINUATION_DECAY,
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
        "residuals_wrt_goal_function",
        "residuals_wrt_homotopy",
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
