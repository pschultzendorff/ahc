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

import logging
from typing import Any

import numpy as np
from tpf.models.error_estimate import (
    DataSavingEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
)
from tpf.models.flow_and_transport import TwoPhaseFlow
from tpf.models.protocol import EstimatesProtocol, TPFProtocol
from tpf.models.reconstruction import (
    EquationsRecMixin,
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
)

logger = logging.getLogger(__name__)


class ErrorEstimateANewtonMixin(EstimatesProtocol):

    def global_discretization_est(self) -> float:
        r"""Compute the global discretization error estimate.

        In the CCFVM, the local residual estimates :math:`\eta_{R,\alpha,K}` are
        (almost) zero, hence the global discretization error estimate is equal to the
        sum of both global nonconformity error estimates given by
        :meth:`ErrorEstimateMixin.global_nonconformity_est`.

        Note: To avoid computing the local and global estimates twice, this method has
            to be called **after** :meth:`SolutionStrategyEst.check_convergence`, which
            evaluates and stores the global nonconformity estimates.

        Returns:
            estimator: The global discretization estimator.

        """
        return sum(self.nonlinear_solver_statistics.nonconformity_est[-1].values())

    def global_linearization_est(self) -> float:
        r"""Compute the global linearization error estimate.

        When solving the nonlinear system with Newton, and assuming that the linear
        system is solved exactly at each Newton iteration, the local linearization error
        estimate is given by summing the local flux estimates over the domain,
        integrating in time and summing the esimates for both fluxes.

        In the CCFVM, the local residual estimates :math:`\eta_{R,\alpha,K}` are
        (almost) zero, hence it holds

        .. math::
            \left{\sum_{\alpha \in \{w,t\}} \int_{I_n} \sum_{K \in \mathcal{T}_h}
                (\eta_{R,\alpha,K} + \eta_{F,\alpha,K})^2 dt \right}^{1/2}
            =
            \left{\sum_{\alpha \in \{w,t\}} \int_{I_n} \sum_{K \in \mathcal{T}_h}
                \eta_{F,\alpha,K}^2 dt \right}^{1/2},

        i.e., the global linearization error estimate is equal to the global flux and
        residual error given by :meth:`ErrorEstimateMixin.global_res_and_flux_est`.

        Note: To avoid computing the local and global estimates twice, this method has
            to be called **after** :meth:`SolutionStrategyEst.check_convergence`, which
            evaluates and stores the global flux and residual estimate.

        Returns:
            estimator: The global linearization estimator.

        """

        return self.nonlinear_solver_statistics.residual_and_flux_est[-1]


class SolutionStrategyANewtonMixin(EstimatesProtocol, TPFProtocol):

    def check_convergence(
        self,
        nonlinear_increment: np.ndarray,
        residual: np.ndarray,
        reference_residual: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        converged, diverged = super().check_convergence(  # type: ignore
            nonlinear_increment, residual, reference_residual, nl_params
        )

        # Adaptive stopping criterion.
        if not diverged and nl_params["nl_adaptive"]:
            nonlinear_increment_norm: float = self.compute_nonlinear_increment_norm(  # type: ignore
                nonlinear_increment
            )
            discretization_est: float = self.global_discretization_est()
            linearization_est: float = self.global_linearization_est()

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
