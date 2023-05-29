"""Implementation of the Buckley-Leverett model in the fractional flow formulation."""

import logging
import math
from functools import partial
from typing import Any, Callable

import numpy as np
import porepy as pp
import sympy
from buckley_leverett import analytical_solution, functions
from porepy.utils.examples_utils import VerificationUtils
from scipy import interpolate

from tpf_lab.models.two_phase_flow import (
    TwoPhaseFlowBoundaryConditions,
    TwoPhaseFlowEquations,
    TwoPhaseFlowSolutionStrategy,
    TwoPhaseFlowVariables,
)
from tpf_lab.numerics.ad.functions import ad_pow
from tpf_lab.visualization.diagnostics import (
    BuckleyLeverettDataSaving,
    BuckleyLeverettSaveData,
    DiagnosticsMixinExtended,
)

# Setup logging.
logger = logging.getLogger(__name__)


class BuckleyLeverettEquations(TwoPhaseFlowEquations):
    _angle: float

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting vector source. Corresponds to the wetting buoyancy flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the x dimension. This is scaled by the angle of the domain.
            vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) *
            self._density_w
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) * self._density_w
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric nonwetting vector source. Corresponds to the nonwetting buoyancy
        flow.

        Defined by gravity times :math:`cos(angle)` of the tube.

        To assign a gravity-like vector source, add a non-zero contribution in
        the x dimension. This is scaled by the angle of the domain.
            vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) *
            self._density_n
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) * self._density_n
        return vals.ravel()


class BuckleyLeverettBoundaryConditions(TwoPhaseFlowBoundaryConditions):
    _influx: float

    def _bc_type_pressure_w(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Wetting pressure boundary conditions.

        Neumann conditions on the left, Dirichlet conditions on the right.

        """
        rhs = np.array([g.num_faces - 1])
        return pp.BoundaryCondition(g, rhs, "dir")

    def _bc_type_pressure_n(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Nonwetting pressure boundary conditions.

        Neumann conditions on the left (index 0), Dirichlet conditions on the right
        (index -1).

        """
        rhs = np.array([g.num_faces - 1])
        return pp.BoundaryCondition(g, rhs, "dir")

    def _bc_type_pressure_c(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Capillary pressure boundary conditions.

        Neumann bc on both sides.

        """
        return pp.BoundaryCondition(g)

    def _dirichlet_bc_values_pressure_n(self, g: pp.Grid) -> np.ndarray:
        """Zero pressure at the right (index 0)."""
        array = np.zeros(g.num_faces)
        return array

    def _dirichlet_bc_values_pressure_w(self, g: pp.Grid) -> np.ndarray:
        """Wetting pressure equals the nonwetting pressure."""
        array = np.zeros(g.num_faces)
        return array

    def _neumann_bc_values_flux_n(self, g: pp.Grid) -> np.ndarray:
        """Injection at the left (index 0). Note that this equals the total flux.

        For some PorePy reason the influx needs to be negative.

        """
        array = np.zeros(g.num_faces)
        array[0] = -self._influx
        return array

    def _bc_values_mobility_w(self, g: pp.Grid) -> np.ndarray:
        array = super()._bc_values_mobility_w(g)
        # Set the wetting mobility at the boundaries to the wetting mobility at residual
        # saturations.
        # For PorePy reasons the values needs to be negative at the Dirichlet boundary.
        # At the Neumann boundary the sign depends on the flow direction.
        array[0] = 0.99
        array[-1] = -0.01
        return array

    def _bc_values_mobility_n(self, g: pp.Grid) -> np.ndarray:
        array = super()._bc_values_mobility_n(g)
        # Set the nonwetting mobility at the boundaries to the nonwetting mobility at
        # residual saturations.
        # For PorePy reasons the values needs to be negative at the Dirichlet boundary.
        # At the Neumann boundary the sign depends on the flow direction.
        array[0] = 0.01
        array[-1] = -0.99
        return array


class BuckleyLeverettDefaultGeometry(pp.ModelGeometry):
    _phys_size: float

    def set_geometry(self) -> None:
        self.set_domain()
        phys_dims: np.ndarray = np.array([self._phys_size])
        cell_dims: np.ndarray = np.array(
            [int(self._phys_size / self.meshing_arguments()["cell_size"])]
        )
        g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
        g_cart.compute_geometry()
        self.mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart]])
        self.nd: int = self.mdg.dim_max()

    def set_domain(self) -> None:
        self._domain = pp.Domain(
            bounding_box={
                "xmin": -10,
                "xmax": self._phys_size - 10,
                "ymin": 0,
                "ymax": 0,
            }
        )


class BuckleyLeverettSolutionStrategy(TwoPhaseFlowSolutionStrategy):
    save_data_time_step: Callable
    """Provided by ``BuckleyLeverettDataSaving``."""
    _flux_t: Callable
    _bc_values_mobility_n: Callable
    _mobility_w: Callable
    _mobility_n: Callable
    calc_exact_solution: Callable
    """Provided by ``BuckleyLeverettSemiAnalyticalSolution``."""

    def __init__(self, params: dict | None) -> None:
        super().__init__(params)
        if params is None:
            params = {}

        # Grid size:
        self._phys_size: float = params.get("phys size", 2)

        self._angle: float = params.get("angle", math.pi / 4)
        """Angle of the tube domain."""
        self._influx: float = params.get("influx", 1.0)
        """Total constant influx at the LHS of the tube."""
        self._cap_pressure_model = None
        """Capillary pressure is ignored in Buckley-Leverett."""

        # Some params for the analytical solution.
        params.update(
            {
                "S_M": 1 - params.get("residual_saturation_n", 0.3),
                "S_m": params.get("residual_saturation_w", 0.3),
            }
        )
        self.analytical = analytical_solution.BuckleyLeverett(params)
        self.resolution: int = params.get("resolution", 500)
        """Resolution for data for interpolation."""
        self.linear_flow: bool = params.get("linear_flow", False)
        """Linear flow function."""

        self.export_each_iteration: bool = params.get("export_each_iteration", False)
        """Export at each iteration. Used for debugging."""

        # Data saving.
        self.results: list[BuckleyLeverettSaveData] = []
        """List of stored results from the convergence analysis."""

    def prepare_simulation(self) -> None:
        super().prepare_simulation()
        self.calc_exact_solution()

    def initial_condition(self) -> None:
        """Residual nonwetting saturation in the left side of the domain. Residual
        wetting saturation in the right side of the domain. A transition zone in the
        middle."""
        super().initial_condition()
        num_cells: int = self.mdg.subdomains()[0].num_cells
        self.initial_saturation = np.full(num_cells, 1 - self._residual_saturation_n)
        self.initial_saturation[int(num_cells / 2) :] = self._residual_saturation_w
        self.initial_saturation[
            int(num_cells / 2) - 10 : int(num_cells / 2) + 10
        ] = np.linspace(
            1 - self._residual_saturation_n, self._residual_saturation_w, 20
        )
        self.equation_system.set_variable_values(
            self.initial_saturation,
            [self.saturation_var],
            time_step_index=self.time_manager.time_index,
        )
        # Set pressure gradient roughly s.t. the initial flux equals one everywhere.
        self.equation_system.set_variable_values(
            np.linspace(157, 0.4, num_cells),
            [self.pressure_w_var],
            time_step_index=self.time_manager.time_index,
        )
        self.equation_system.set_variable_values(
            np.linspace(157, 0.4, num_cells),
            [self.pressure_n_var],
            time_step_index=self.time_manager.time_index,
        )

    # def _export(self):
    #     """Export only each 10th time step."""
    #     if self.time_manager.time_index % 10 == 0:
    #         super()._export()

    def _export_iteration(self):
        if hasattr(self, "exporter"):
            primary_pressure = self.equation_system.get_variable_values(
                [self.primary_pressure_var], iterate_index=0
            )
            saturation = self.equation_system.get_variable_values(
                [self.saturation_var], iterate_index=0
            )
            self.exporter.write_vtu(
                (self.saturation_var, saturation),
                # Cheating a little bit to get an individual index for each iteration.
                time_step=self.time_manager.time_index * 100
                + self._nonlinear_iteration,
            )
            self.exporter.write_vtu(
                (self.primary_pressure_var, primary_pressure),
                # Cheating a little bit to get an individual index for each iteration.
                time_step=self.time_manager.time_index * 1000
                + self._nonlinear_iteration,
            )

    def after_nonlinear_iteration(self, solution: np.ndarray) -> None:
        super().after_nonlinear_iteration(solution)
        if self.export_each_iteration:
            self._export_iteration()

    # Ignore mypy complaining about uncompatible signature for ``save_data_time_step``.
    def after_nonlinear_failure(  # type: ignore
        self, solution: np.ndarray, errors: list[float], iteration_counter: int
    ) -> None:
        # Since the L2 error is not of interest when the model did not reach the last
        # time step, data can be saved before the time step solution is distributed.
        # ``super().after_nonlinear_failure()`` will then raise an ``ValueError``.
        self.save_data_time_step(errors, iteration_counter)
        super().after_nonlinear_failure(solution, errors, iteration_counter)

    # Ignore mypy complaining about uncompatible signature for ``save_data_time_step``.
    def after_nonlinear_convergence(  # type: ignore
        self, solution: np.ndarray, errors: list[float], iteration_counter: int
    ) -> None:
        super().after_nonlinear_convergence(solution, errors, iteration_counter)
        # Save data (and calculate L2 error only after the time step solution was
        # distributed).
        self.save_data_time_step(errors, iteration_counter)


class BuckleyLeverettSemiAnalyticalSolution:
    """Mixin containing a semi-analytical solution to the 1d Buckley-Leverett
    solution."""

    mdg: pp.MixedDimensionalGrid
    """Grid for the numerical solution"""
    domain: pp.Domain
    """Physical domain for the numerical solution."""
    analytical: analytical_solution.BuckleyLeverett
    """Analytical solution class."""
    linear_flow: bool
    """Linear/nonlinear flow."""
    _residual_saturation_w: float
    _residual_saturation_n: float
    initial_saturation: np.ndarray
    resolution: int

    def calc_exact_solution(
        self,
    ) -> None:
        """Calculate exact solution on the given grid."""
        # Get the x coordinates of the cell centers. The cell centers range from 0
        # to ``self.phys_size``. Adjust for that.
        adjusted_cell_centers = (
            self.mdg.subdomains()[0].cell_centers[0, :]
            + self.domain.bounding_box["xmin"]
        )
        if self.linear_flow:
            # In the case of a linear flow function, the solution is the initial
            # condition shifted by the flow speed.
            speed = self.analytical.total_flow_prime(1 - self._residual_saturation_n)
            # Get the cell centers that do not get "shifted out" of the domain by
            # the flux.""
            indices = np.argwhere(
                np.logical_not(
                    np.logical_or(
                        adjusted_cell_centers + speed < adjusted_cell_centers[0],
                        adjusted_cell_centers + speed > adjusted_cell_centers[-1],
                    )
                )
            )
            if indices.shape[0] == adjusted_cell_centers.shape[0]:
                solution = self.initial_saturation
            else:
                if speed > 0:
                    # Shift to the right.
                    solution = np.concatenate(
                        [
                            np.full(
                                (adjusted_cell_centers.shape[0] - indices.shape[0]),
                                1 - self._residual_saturation_n,
                            ),
                            self.initial_saturation[indices.flatten()],
                        ]
                    )
                else:
                    # Shift to the left.
                    solution = np.concatenate(
                        [
                            self.initial_saturation[indices.flatten()],
                            np.full(
                                (adjusted_cell_centers.shape[0] - indices.shape[0]),
                                self._residual_saturation_w,
                            ),
                        ]
                    )
            self._exact_solution: np.ndarray = solution

        else:
            # In other cases the semianalytical solution can be constructed by the
            # concave hull.
            # NOTE: This holds also for fully convex or concave cases, so no need to
            # differentiate.
            # Get solution func.
            _, solution_func = self.analytical.concave_hull()

            # Axis on the true solution are switched. Interpolate to fix this.
            yy = np.linspace(
                1 - self._residual_saturation_w,
                self._residual_saturation_n,
                self.resolution,
            )
            xx = solution_func(yy)
            # Extend true solution on both sides with residual saturations.
            xx_full = np.concatenate(
                (
                    np.linspace(
                        self.domain.bounding_box["xmin"], xx[0], self.resolution
                    ),
                    xx,
                    np.linspace(
                        xx[-1], self.domain.bounding_box["xmax"], self.resolution
                    ),
                )
            )
            yy_full = np.concatenate(
                (
                    np.full(self.resolution, 1 - self._residual_saturation_n),
                    yy,
                    np.full(self.resolution, self._residual_saturation_w),
                )
            )

            # Sanitize the solution to avoid numerical kinks etc.
            xx_full, yy_full = analytical_solution.sanitize_function(xx_full, yy_full)

            # Create an interpolation class from the results.
            solution_func_switched = interpolate.interp1d(xx_full, yy_full, "previous")

            self._exact_solution = solution_func_switched(adjusted_cell_centers)


# Ignore mypy complaining about uncompatible signature between
# ``BuckleyLeverettSolutionStrategy`` and ``BuckleyLeverettDataSaving``.
class BuckleyLeverettSetup(  # type: ignore
    BuckleyLeverettEquations,
    TwoPhaseFlowVariables,
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettSolutionStrategy,
    #
    BuckleyLeverettDefaultGeometry,
    #
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettDataSaving,
    VerificationUtils,
    DiagnosticsMixinExtended,
):
    ...


class BuckleyLeverettEquations_WobblyRelPerm(BuckleyLeverettEquations):
    _yscales: list[float]
    _xscales: list[float]
    _offsets: list[float]

    def _rel_perm_w(self) -> pp.ad.Operator:
        """Add a perturbation to the wetting phase rel. perm."""
        rel_perm_w = super()._rel_perm_w()
        return rel_perm_w + self._error_function_deriv()

    def _error_function_deriv(self) -> pp.ad.Operator:
        """Returns the derivative of the error function w.r.t. the saturation.

        This can be used to simulate perturbations in the cap. pressure and rel. perm.
        models.

        Returns:
            Derivative of the error function in terms of :math:`S_w`.
        """
        s = self.equation_system.md_variable(self.saturation_var)
        xscales = [pp.ad.Scalar(xscale) for xscale in self._xscales]
        yscales = [pp.ad.Scalar(yscale) for yscale in self._yscales]
        offsets = [pp.ad.Scalar(offset) for offset in self._offsets]
        exp_func = pp.ad.Function(pp.ad.functions.exp, "exp")
        square_func = pp.ad.Function(partial(ad_pow, exponent=2), "square")
        error = pp.ad.Scalar(0) * s
        for xscale, yscale, offset in zip(xscales, yscales, offsets):
            error = error + yscale * exp_func(
                pp.ad.Scalar(-1) * xscale * square_func(s - offset)
            )
        return error


class BuckleyLeverettSolutionStrategy_WobblyRelPerm(BuckleyLeverettSolutionStrategy):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Parameters for the error function derivative:
        self._yscales: list[float] = params.get("yscales", [1.0])
        self._xscales: list[float] = params.get("xscales", [200])
        self._offsets: list[float] = params.get("offsets", [0.5])
        # Change flow function for the analytical solution.
        self.analytical.fractionalflow = WobblyFractionalFlowSympy(params)
        self.analytical.lambdify()


class BuckleyLeverettSetup_WobblyRelPerm(  # type: ignore
    BuckleyLeverettEquations_WobblyRelPerm,
    TwoPhaseFlowVariables,
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettSolutionStrategy_WobblyRelPerm,
    #
    BuckleyLeverettDefaultGeometry,
    #
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettDataSaving,
    VerificationUtils,
    DiagnosticsMixinExtended,
):
    ...


class WobblyFractionalFlowSympy(functions.FractionalFlowSymPy):
    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        # Parameters for the error function derivative:
        self.yscales: list[float] = params.get("yscales", [1.0])
        self.xscales: list[float] = params.get("xscales", [200])
        self.offsets: list[float] = params.get("offsets", [0.5])

    def lambda_w(self):
        r"""Wetting phase mobility.

        Power model
        .. math::
            k_{r,w}(S_w)=S_w^3 + \epsilon(S_w)

        """
        return self.S_normalized() ** 3 + self.error_function_deriv()

    def error_function_deriv(self):
        return sympy.Add(
            *[
                yscale * sympy.exp(-xscale * (self.S_w - offset) ** 2)
                for xscale, yscale, offset in zip(
                    self.xscales, self.yscales, self.offsets
                )
            ]
        )
