r"""This module implements an adaptive Newton method based on a posteriori error
estimators.

while True:
    1. Newton step
    2. Compute discretization error estimator and linearization error estimate
    3. If linearization error estimator <= tol * discretization error estimator:
        break

[M. Vohralík and M. F. Wheeler, “A posteriori error estimates, stopping criteria, and
adaptivity for two-phase flows,” Comput Geosci, vol. 17, no. 5, pp. 789–812, Oct. 2013,
doi: 10.1007/s10596-013-9356-0.]

"""

import itertools
import logging
import typing
from typing import Any, Literal

import numpy as np
import porepy as pp

from tpf.models.error_estimate import (
    DataSavingEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
)
from tpf.models.flow_and_transport import SolutionStrategyTPF, TwoPhaseFlow
from tpf.models.protocol import (
    AdaptiveNewtonProtocol,
    EstimatesProtocol,
    ReconstructionProtocol,
    TPFProtocol,
)
from tpf.models.reconstruction import (
    EquationsRecMixin,
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
)
from tpf.numerics.quadrature import Integral
from tpf.utils.constants_and_typing import FLUX_NAME, TOTAL_FLUX, WETTING_FLUX

logger = logging.getLogger(__name__)


class ErrorEstimateANewtonMixin(
    AdaptiveNewtonProtocol, EstimatesProtocol, ReconstructionProtocol, TPFProtocol
):
    def local_temp_est(self, flux_name: FLUX_NAME) -> None:
        r"""Calculate the local-in-space temporal error estimators.

        We assume the following sub-dictionaries to be present in the data dictionary:
            ``iterate_dictionary``, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in ``iterate_dictionary`` will be updated:
            - ``{flux_name}_T_estimator``, storing the local time error estimator.

        Note: The local estimators read

        .. math::
            \|\mathbf{u}_{\alpha,h,\tau}(t) - \mathbf{u}_{\alpha,h}^{n,i,k}\|_K^2,

        where :math:`\mathbf{u}_{\alpha,h,\tau}(t)` is the FV flux at time :math:`t`,
        and :math:`\mathbf{u}_{\alpha,h}^{n,i,k}` is the (non equilibrated) FV flux at
        the current time step, continuation iteration, and Newton iteration.

        To obtain the global estimator, the local estimators are summed over the domain
        and integrated in time. The time integral is approximated with the trapezoidal
        rule.  As the difference is piecewise affine on the time integral and zero at
        :math:`t^n`, where both fluxes are equal, it suffices to evaluate the difference
        at :math:`t^{n-1}`. The time integral is then approximate as

        .. math::
            \frac{\Delta t}{2}
            \|\mathbf{u}_{\alpha,h,\tau}(t^{n-1}) - \mathbf{u}_{\alpha,h}^{n,i,k}\|_K^2.

        Parameters:
            flux_name: Name of the flux to calculate the estimator for.

        """
        # Retrieve flux w.r.t. goal rel. perm. and nonequilbirated flux coeffs.
        fv_coeffs_new = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs",
            self.g_data,
            iterate_index=0,
        )
        fv_coeffs_old = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs",
            self.g_data,
            time_step_index=0,
        )

        def integrand(x: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs_new[..., 0] - fv_coeffs_old[..., 0]
            ) * x[..., 0] + (fv_coeffs_new[..., 1] - fv_coeffs_old[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs_new[..., 0] - fv_coeffs_old[..., 0]
            ) * x[..., 1] + (fv_coeffs_new[..., 2] - fv_coeffs_old[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise and store the result.
        integral: Integral = self.quadrature_est.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"{flux_name}_T_estimator",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def local_linearization_est(self, flux_name: FLUX_NAME) -> None:
        """

        We assume the following sub-dictionaries to be present in the data dictionary:
            ``iterate_dictionary``, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in ``iterate_dictionary`` will be updated:
            - ``{flux_name}_L_estimator``, storing the local in time and space
              linearization error estimator.

        """
        # Retrieve flux w.r.t. goal rel. perm. and nonequilbirated flux coeffs.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", self.g_data, iterate_index=0
        )
        fv_equilibrated_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs", self.g_data, iterate_index=0
        )

        def integrand(x: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs[..., 0] - fv_equilibrated_coeffs[..., 0]
            ) * x[..., 0] + (fv_coeffs[..., 1] - fv_equilibrated_coeffs[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs[..., 0] - fv_equilibrated_coeffs[..., 0]
            ) * x[..., 1] + (fv_coeffs[..., 2] - fv_equilibrated_coeffs[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise and store the result.
        integral: Integral = self.quadrature_est.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        # To calculate the global estimator, only the estimators fromt the most recent
        # nonlinear iteration are needed. No need to shift anything.
        pp.set_solution_values(
            f"{flux_name}_L_estimator",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def global_res_est(self) -> float:
        r"""Sum local residual estimators, integrate in time, and sum total and
         wetting estimators.

        Contrary to :meth:`ErrorEstimatesMixin.global_res_and_flux_est`, the local flux
        estimator does not contribute to the spatial discretization error. Instead, it
        is decomposed and separated into the temporal and linearization
        estimator.

        The remaining residual error estimate is zero in theory and negligible in
        practice. For faster evaluation, it may not be evaluated.

        Note: The residual estimator is not time dependent, hence we multiply the
        value at :math:`t_n` by :math:`\Delta t` to get the time integral.

        Returns:
            estimator: Global discretization error estimator.

        """
        if self.params.get("anewton_fast_evaluation", True):
            return 0.0
        else:
            estimators: dict[str, float] = {}
            for flux_name in (TOTAL_FLUX, WETTING_FLUX):
                # Calculate local estimators.
                self.local_residual_est(flux_name)
                # Load spatial integrals from current nonlinear iteration.
                local_integral_R: np.ndarray = pp.get_solution_values(
                    f"{flux_name}_R_estimator", self.g_data, iterate_index=0
                )
                # Calculate global values at current iteration.
                # NOTE The values stored were the squares of the elementwise norms, hence we
                # take the square root first
                global_integral: float = local_integral_R.sum()
                # Integrate in time by multiplying .
                estimators[flux_name] = self.time_manager.dt * global_integral
            # Sum estimators for both equations.
            estimator: float = sum(estimators.values()) ** 1 / 2
            logger.info(f"Global residual error estimator: {estimator}")
            return estimator

    def global_spatial_est(self) -> float:
        """Evaluate the global spatial discretization error estimator."""
        residual_estimator: float = self.global_res_est()
        nc_estimators: tuple[float, float] = self.global_nonconformity_est()
        estimator: float = residual_estimator + sum(nc_estimators)
        logger.info(f"Global spatial discretization error estimator: {estimator}")
        return estimator

    def global_temp_est(self) -> float:
        """Evaluate the global temporal discretization error estimator."""
        estimators: dict[str, float] = {}
        for flux_name in (TOTAL_FLUX, WETTING_FLUX):
            # Calculate local estimators.
            self.local_temp_est(flux_name)
            # Load spatial integral from current nonlinear iteration.
            local_integral: np.ndarray = pp.get_solution_values(
                f"{flux_name}_T_estimator", self.g_data, iterate_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # do not need to square here.
            global_integral: float = local_integral.sum()
            # Integrate in time by multiplying with time step size / 3. This comes from
            # writing the local estimator as an affine function of :math:`t` that is
            # zero at :math:`t^n` and calculating the integral exactly.
            estimators[flux_name] = self.time_manager.dt / 3.0 * global_integral
        # Sum estimators for both equations.
        estimator: float = sum(estimators.values()) ** (1 / 2)
        logger.info(f"Global temporal discretization error estimator: {estimator}")
        return estimator

    def global_discretization_est(self) -> float:
        """Evaluate the global discretization error estimator."""
        spatial_estimator: float = self.nonlinear_solver_statistics.spatial_est[-1]
        temp_estimator: float = self.nonlinear_solver_statistics.temp_est[-1]
        estimator: float = spatial_estimator + temp_estimator
        logger.info(f"Global discretization error estimator: {estimator}")
        return estimator

    def global_linearization_est(self) -> float:
        r"""Compute the global linearization error estimate.


        Returns:
            estimator: The global linearization estimator.

        """

        estimators: dict[str, float] = {}
        for flux_name in (TOTAL_FLUX, WETTING_FLUX):
            # Calculate local estimators.
            self.local_linearization_est(flux_name)
            # Load spatial integral from current nonlinear iteration.
            local_integral: np.ndarray = pp.get_solution_values(
                f"{flux_name}_L_estimator", self.g_data, iterate_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # do not need to square here.
            global_integral: float = local_integral.sum()
            # Integrate in time by multiplying with time step size.
            estimators[flux_name] = self.time_manager.dt * global_integral
        # Sum estimators for both equations.
        estimator: float = sum(estimators.values()) ** (1 / 2)
        logger.info(f"Global linearization error estimator: {estimator}")
        return estimator


class SolutionStrategyANewtonMixin(
    AdaptiveNewtonProtocol, EstimatesProtocol, TPFProtocol
):
    def set_initial_estimators(self) -> None:
        """Initialize iterate and time step values for additional error estimators."""
        # In ``EstimatesProtocol``, this method is abstract and
        # not implemented, which mypy complains about
        super().set_initial_estimators()  # type: ignore
        for flux_name, specifier in itertools.product(
            ["total", "wetting"],
            ["T_estimator", "L_estimator"],
        ):
            pp.set_solution_values(
                f"{flux_name}_{specifier}",
                np.zeros(self.g.num_cells),
                self.g_data,
                time_step_index=0,
                iterate_index=0,
            )

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        # NOTE Here, we explicitely do NOT want to call
        # ``SolutionStrategyEstMixin.check_convergence``, but
        # ``TwoPhaseFlow.check_convergence``. The former logs estimators we are not
        # interested in.
        converged, diverged = SolutionStrategyTPF.check_convergence(
            self,  # type: ignore
            nonlinear_increment,
            residual,
            reference_residual,
            nl_params,
        )

        linearization_est: float = self.global_linearization_est()
        self.nonlinear_solver_statistics.log_error(
            # NOTE The discretization error estimate does not need to be calculated
            # at this point. After HC convergence is sufficient if we want the code
            # to be more efficient.
            spatial_est=self.global_spatial_est(),
            temp_est=self.global_temp_est(),
            linearization_est=linearization_est,
        )

        # Adaptive stopping criterion.
        if not diverged and nl_params["nl_adaptive"]:
            nonlinear_increment_norm: float = self.compute_nonlinear_increment_norm(  # type: ignore
                nonlinear_increment
            )
            discretization_est: float = self.global_discretization_est()
            # If Newton diverges, the estimators lose their meaning and the adaptive
            # criterion might incorrectly stop the HC loop. Hence, we check that the
            # nonlinear increment norm is not too large.
            if (
                linearization_est <= nl_params["nl_error_ratio"] * discretization_est
                and nonlinear_increment_norm <= nl_params["nl_adaptive_convergence_tol"]
            ):
                logger.info(
                    f"Linearization error {linearization_est} smaller than"
                    + f" {nl_params['nl_error_ratio']} * discretization error"
                    + f" {discretization_est}. Stopping Newton loop."
                )
                converged = True

        return converged, diverged

    def after_nonlinear_convergence(self) -> None:  # type: ignore
        """Export and move to the next homotopy continuation step.

        When ``self._limit_saturation_change == True``, check if the wetting saturation
        has changed too much

        """
        super().after_nonlinear_convergence()  # type: ignore
        for flux_name, specifier in itertools.product(
            ["total", "wetting"],
            [
                "T_estimator",
                "L_estimator",
            ],
        ):
            flux_values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{specifier}", self.g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{flux_name}_{specifier}", flux_values, self.g_data, hc_index=0
            )


class TwoPhaseFlowANewton(
    ErrorEstimateANewtonMixin,
    SolutionStrategyANewtonMixin,
    # Estimator mixins:
    ErrorEstimateMixin,
    DataSavingEst,
    SolutionStrategyEst,
    # Reconstruction mixins:
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    EquationsRecMixin,
    # The rest
    TwoPhaseFlow,
): ...  # type: ignore
