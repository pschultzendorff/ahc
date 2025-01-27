import itertools
import logging
import typing
from typing import Any, Literal, Optional

import numpy as np
import porepy as pp
from porepy.viz.exporter import DataInput
from tpf.models.constitutive_laws_tpf import RelativePermeability, RelPermConstants
from tpf.models.error_estimate import (
    DataSavingEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
)
from tpf.models.flow_and_transport import SolutionStrategyTPF, TwoPhaseFlow
from tpf.models.phase import FluidPhase
from tpf.models.protocol import (
    EstimatesProtocol,
    HCProtocol,
    ReconstructionProtocol,
    TPFProtocol,
)
from tpf.models.reconstruction import (
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
)
from tpf.numerics.quadrature import Integral
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    PHASENAME,
    PRESSURE_KEY,
)

logger = logging.getLogger(__name__)


# ``HCProtocol`` and ``TPFProtocol`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class RelativePermeabilityHC(HCProtocol, RelativePermeability):  # type: ignore
    """


    Note: As an alternate construction, the phase and total fluxes could be calculated
    both in terms of the target and starting relative permeabilities, and then added
    together, weighted by the homotopy parameter. If evaluation of operators in PorePy
    would be more efficient (i.e., reusing already evaluated expressions), this could
    save time. Currently, this would require double evaluation of the pressure
    potentials.

    """

    @typing.override
    def set_rel_perm_constants(self) -> None:
        rel_perm_constants: dict[str, dict] = self.params.get("rel_perm_constants", {})
        rel_perm_1_constants: dict[str, Any] = rel_perm_constants.get("model_1", {})
        rel_perm_2_constants: dict[str, Any] = rel_perm_constants.get("model_2", {})

        self._rel_perm_constants_1 = RelPermConstants(**rel_perm_1_constants)
        self._rel_perm_constants_2 = RelPermConstants(**rel_perm_2_constants)

    # Ignore mypy complaining about incompatible signature with supertype.
    @typing.override
    def rel_perm(self, saturation_w: pp.ad.Operator, phase: FluidPhase) -> pp.ad.Operator:  # type: ignore[override]
        # Return homotopy continuation relative permeability.
        # Mypy gives an error here, because it thinks the empty ``HCProtocol.rel_perm``
        # is called. During runtime, ``HCProtocol`` does not have this method, hence we
        # can ignore the error. Additionally, we ignore complaints about the wrong
        # number of arguments and wrong arguemnt types.
        if self.hc_rel_perm_toggle:
            rel_perm_1: pp.ad.Operator = super().rel_perm(  # type: ignore
                saturation_w,  # type: ignore
                phase,  # type: ignore
                rel_perm_constants=self._rel_perm_constants_1,  # type: ignore
            )
            rel_perm_2: pp.ad.Operator = super().rel_perm(  # type: ignore
                saturation_w,  # type: ignore
                phase,  # type: ignore
                rel_perm_constants=self._rel_perm_constants_2,  # type: ignore
            )
            return (
                self.nonlinear_solver_statistics.hc_lambda_ad * rel_perm_1
                + (pp.ad.Scalar(1) - self.nonlinear_solver_statistics.hc_lambda_ad)
                * rel_perm_2
            )

        # Return goal relative permeability.
        else:
            return super().rel_perm(  # type: ignore
                saturation_w, phase, rel_perm_constants=self._rel_perm_constants_2  # type: ignore
            )

    @typing.override
    def rel_perm_np(
        self,
        saturation_w: np.ndarray,
        phase: FluidPhase,
    ) -> np.ndarray:
        if self.hc_rel_perm_toggle:
            rel_perm_1: np.ndarray = super().rel_perm_np(  # type: ignore
                saturation_w,  # type: ignore
                phase,  # type: ignore
                rel_perm_constants=self._rel_perm_constants_1,  # type: ignore
            )
            rel_perm_2: np.ndarray = super().rel_perm_np(  # type: ignore
                saturation_w,  # type: ignore
                phase,  # type: ignore
                rel_perm_constants=self._rel_perm_constants_2,  # type: ignore
            )
            return (
                self.nonlinear_solver_statistics.hc_lambda_fl * rel_perm_1
                + (1 - self.nonlinear_solver_statistics.hc_lambda_fl) * rel_perm_2
            )

        # Return goal relative permeability.
        else:
            return super().rel_perm_np(  # type: ignore
                saturation_w, phase, rel_perm_constants=self._rel_perm_constants_2  # type: ignore
            )


# The various protocols define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class EstimatesHCMixin(EstimatesProtocol, ReconstructionProtocol, TPFProtocol):  # type: ignore

    def local_hc_est(self, flux_name: Literal["total", "wetting_from_ff"]) -> None:
        """

        We assume the following sub-dictionaries to be present in the data dictionary:
            iterate_dictionary, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in iterate_dictionary will be shifted and updated:
            - {flux_name}_C_estimate, storing the local in time and space continuation
                error estimate.

        """
        _, g_data = self.mdg.subdomains(return_data=True)[0]
        # Retrieve flux w.r.t. goal rel. perm. and nonequilbirated flux coeffs.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", g_data, iterate_index=0
        )
        fv_goal_rel_perm_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_wrt_goal_rel_perm_RT0_coeffs", g_data, iterate_index=0
        )

        def integrand(x: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs[..., 0] - fv_goal_rel_perm_coeffs[..., 0]
            ) * x[..., 0] + (fv_coeffs[..., 1] - fv_goal_rel_perm_coeffs[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs[..., 0] - fv_goal_rel_perm_coeffs[..., 0]
            ) * x[..., 1] + (fv_coeffs[..., 2] - fv_goal_rel_perm_coeffs[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise, shift values, and store the result.
        integral: Integral = self.quadrature_estimate.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{flux_name}_C_estimate",
            g_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_C_estimate",
            integral.elementwise,
            g_data,
            iterate_index=0,
        )

    def local_linearization_est(
        self, flux_name: Literal["total", "wetting_from_ff"]
    ) -> None:
        """

        We assume the following sub-dictionaries to be present in the data dictionary:
            iterate_dictionary, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in iterate_dictionary will be shifted and updated:
            - {flux_name}_L_estimate, storing the local in time and space linearization
                error estimate.

        """
        _, g_data = self.mdg.subdomains(return_data=True)[0]
        # Retrieve flux w.r.t. goal rel. perm. and nonequilbirated flux coeffs.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", g_data, iterate_index=0
        )
        fv_equilibrated_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs", g_data, iterate_index=0
        )

        def integrand(x: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs[..., 0] - fv_equilibrated_coeffs[..., 0]
            ) * x[..., 0] + (fv_coeffs[..., 1] - fv_equilibrated_coeffs[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs[..., 0] - fv_equilibrated_coeffs[..., 0]
            ) * x[..., 1] + (fv_coeffs[..., 2] - fv_equilibrated_coeffs[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise, shift values, and store the result.
        integral: Integral = self.quadrature_estimate.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{flux_name}_L_estimate",
            g_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_L_estimate",
            integral.elementwise,
            g_data,
            iterate_index=0,
        )

    def global_discretization_est(self) -> float:
        """Evaluate the global discretization error estimate."""
        residual_estimate: float = self.global_res_and_flux_est()
        nonconformity_estimates: tuple[float, float] = self.global_nonconformity_est()
        logger.info(
            "Global discretization error estimate:"
            + f" {3 * residual_estimate + sum(nonconformity_estimates)}"
        )
        return 3 * residual_estimate + sum(nonconformity_estimates)

    def global_hc_est(self) -> float:
        _, g_data = self.mdg.subdomains(return_data=True)[0]
        estimators: dict[str, float] = {}
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

            # Calculate local estimate.
            self.local_hc_est(flux_name)
            # Load spatial integral from current time step.
            integral_C_new: np.ndarray = pp.get_solution_values(
                f"{flux_name}_C_estimate", g_data, iterate_index=0
            )
            # Load spatial integral from previous time step.
            integral_C_old: np.ndarray = pp.get_solution_values(
                f"{flux_name}_C_estimate", g_data, time_step_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # do not need to square here.
            global_integral_new: float = integral_C_new.sum()
            global_integral_old: float = integral_C_old.sum()
            # Integrate in time by trapezoidal rule.
            estimators[flux_name] = (
                self.time_manager.dt / 2 * (global_integral_new + global_integral_old)
            ) ** (1 / 2)
        logger.info(
            f"Global continuation error estimate: {3 * sum(estimators.values())}"
        )
        return 3 * sum(estimators.values())

    def global_linearization_est(self) -> float:
        _, g_data = self.mdg.subdomains(return_data=True)[0]
        estimators: dict[str, float] = {}
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

            # Calculate local estimate.
            self.local_linearization_est(flux_name)
            # Load spatial integral from current time step.
            integral_L_new: np.ndarray = pp.get_solution_values(
                f"{flux_name}_L_estimate", g_data, iterate_index=0
            )
            # Load spatial integral from previous time step.
            integral_L_old: np.ndarray = pp.get_solution_values(
                f"{flux_name}_L_estimate", g_data, time_step_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # do not need to square here.
            global_integral_new: float = integral_L_new.sum()
            global_integral_old: float = integral_L_old.sum()
            # Integrate in time by trapezoidal rule.
            estimators[flux_name] = (
                self.time_manager.dt / 2 * (global_integral_new + global_integral_old)
            ) ** (1 / 2)
        logger.info(
            f"Global linearization error estimate: {3 * sum(estimators.values())}"
        )
        return 3 * sum(estimators.values())

    def global_res_and_flux_est(self) -> float:
        r"""Sum local residual estimators, integrate in time, and sum total and
         wetting estimators.

        Contrary to :meth:`ErrorEstimatesMixin.global_res_and_flux_est`, the local flux
        estimator does not contribute to the discretization error. Instead, it is
        decomposed and separated into the continuation and linearization estimator.

        The remaining residual error estimate is zero in theory and negligible in
        practice. For faster evaluation, it may not be evaluated.

        Returns:
            estimate: Global discretization error estimate.

        """
        if self.params.get("hc_fast_evaluation", True):
            return 0.0
        else:
            _, g_data = self.mdg.subdomains(return_data=True)[0]

            estimators: dict[str, float] = {}
            for flux_name in ["total", "wetting_from_ff"]:
                # Satisfy mypy.
                flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

                # Calculate local estimates.
                self.local_residual_est(flux_name)
                # Load spatial integrals from current time step.
                integral_R_new: np.ndarray = pp.get_solution_values(
                    f"{flux_name}_R_estimate", g_data, iterate_index=0
                )
                # Load spatial integrals from previous time step.
                integral_R_old: np.ndarray = pp.get_solution_values(
                    f"{flux_name}_R_estimate", g_data, time_step_index=0
                )
                # Calculate global values at current and previous time step.
                # NOTE The values stored were the squares of the elementwise norms, hence we
                # take the square root first
                global_integral_new: float = integral_R_new.sum()
                global_integral_old: float = integral_R_old.sum()
                # Integrate in time by trapezoidal rule.
                estimators[flux_name] = (
                    self.time_manager.dt
                    / 2
                    * (global_integral_new + global_integral_old)
                ) ** (1 / 2)
            logger.info(f"Global residual error estimate: {sum(estimators.values())}")
            return sum(estimators.values())

    def relative_global_discretization_est(self) -> float:
        """Return relative global discretization error estimate."""
        return self.global_discretization_est() / self.global_energy_norm()

    def relative_global_hc_est(self) -> float:
        """Return relative global homotopy continuation error estimate."""
        return self.global_hc_est() / self.global_energy_norm()

    def relative_global_linearization_est(self) -> float:
        """Return relative global linearization error estimate."""
        return self.global_linearization_est() / self.global_energy_norm()

    def total_est(self) -> float:
        """Return total error estimate, consisting of discretization, homotopy
        continuation, and linearization components.

        """
        return (
            self.global_discretization_est()
            + self.global_hc_est()
            + self.global_linearization_est()
        )


# The various protocols define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SolutionStrategyHC(HCProtocol, EstimatesProtocol, SolutionStrategyTPF):  # type: ignore

    @property
    def hc_indices(self) -> list[int]:
        """Return the indices of the homotopy continuation variables."""
        return [0]

    @property
    def uses_hc(self) -> bool:
        return True

    @property
    def hc_is_converged(self) -> bool:
        return self._hc_is_converged

    @hc_is_converged.setter
    def hc_is_converged(self, value: bool) -> None:
        self._hc_is_converged = value

    @property
    def hc_is_diverged(self) -> bool:
        return self._hc_is_diverged

    @hc_is_diverged.setter
    def hc_is_diverged(self, value: bool) -> None:
        self._hc_is_diverged = value

    def prepare_simulation(self) -> None:
        # This is mixed with more Solutionstrategies that implement
        # ``prepare_simulation``. We ignore the mypy error.
        super().prepare_simulation()  # type: ignore
        self.setup_hc(self.params)

    def setup_hc(self, hc_params: dict[str, Any]) -> None:
        self.hc_constant_decay: bool = hc_params["hc_constant_decay"]
        self.hc_init_decay: float = hc_params["hc_lambda_decay"]
        self.hc_decay: float = self.hc_init_decay
        self.hc_decay_min_max: tuple[float, float] = hc_params["hc_decay_min_max"]
        self.nl_iter_optimal_range: tuple[int, int] = hc_params["nl_iter_optimal_range"]
        self.nl_iter_relax_factors: tuple[float, float] = hc_params[
            "nl_iter_relax_factors"
        ]

        self.hc_decay_recomp_max: int = hc_params["hc_decay_recomp_max"]
        self.hc_decay_recomp_counter: int = 0

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        g = self.mdg.subdomains()[0]
        self.equation_system.set_variable_values(
            np.full(g.num_cells * 2, 0.0),
            [self.wetting.p, self.nonwetting.p],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )
        self.equation_system.set_variable_values(
            np.full(g.num_cells * 2, 0.5),
            [self.wetting.s, self.nonwetting.s],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )

    # region HC LOOP
    def before_hc_loop(self) -> None:
        """Reset HC parameter and residuals."""
        # Reset lambda and decay.
        self.nonlinear_solver_statistics.hc_reset()
        self.hc_decay = self.hc_init_decay
        self.hc_decay_recomp_counter = 0
        self.convergence_status = False
        # We do not need to specifically set the solution from the previous time step as
        # a first guess, as this solution is stored at ``hc_index = 0`` anyways.

    def before_hc_iteration(self) -> None:
        pass

    def after_hc_iteration(self) -> None:
        """Decay lambda and increase iteration counter."""
        self.nonlinear_solver_statistics.hc_lambda_fl *= self.hc_decay
        self.nonlinear_solver_statistics.hc_lambda_ad.set_value(
            self.nonlinear_solver_statistics.hc_lambda_fl
        )
        self.nonlinear_solver_statistics.hc_lambdas.append(
            self.nonlinear_solver_statistics.hc_lambda_fl
        )
        logger.info(
            f"Decayed hc_lambda to"
            + f" {self.nonlinear_solver_statistics.hc_lambda_fl:.2f}"
        )
        self.nonlinear_solver_statistics.hc_num_iteration += 1

    def after_hc_convergence(self) -> None:
        """Move to the next time step."""
        time_step_solution = self.equation_system.get_variable_values(hc_index=0)
        self.equation_system.shift_time_step_values(
            max_index=len(self.time_step_indices)
        )
        self.equation_system.set_variable_values(
            time_step_solution, time_step_index=0, additive=False
        )

        self.convergence_status = True
        self.save_data_time_step()
        # Update the time step magnitude if the dynamic scheme is used.
        if not self.time_manager.is_constant:
            self.time_manager.compute_time_step(
                iterations=self.nonlinear_solver_statistics.hc_num_iteration
            )

        # Shift reconstructions and estimates to the next time step.
        g_data = self.mdg.subdomains(return_data=True)[0][1]
        for pressure_key, specifier in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["postprocessed_coeffs", "reconstructed_coeffs", "NC_estimate"],
        ):
            pp.shift_solution_values(
                f"{pressure_key}_{specifier}",
                g_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            pressure_values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_{specifier}", g_data, hc_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_{specifier}",
                pressure_values,
                g_data,
                time_step_index=0,
            )
        for flux_name, estimate_name in itertools.product(
            ["total", "wetting_from_ff"],
            ["R_estimate", "F_estimate", "C_estimate", "L_estimate"],
        ):
            pp.shift_solution_values(
                f"{flux_name}_{estimate_name}",
                g_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            flux_values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{estimate_name}", g_data, hc_index=0
            )
            pp.set_solution_values(
                f"{flux_name}_{estimate_name}", flux_values, g_data, time_step_index=0
            )
        for energy_norm_term in [
            "energy_norm_saturation_part",
            "energy_norm_global_pressure_part",
            "energy_norm_complimentary_pressure_part",
        ]:
            pp.shift_solution_values(
                energy_norm_term,
                g_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            local_term: np.ndarray = pp.get_solution_values(
                energy_norm_term, g_data, hc_index=0
            )
            pp.set_solution_values(
                energy_norm_term, local_term, g_data, time_step_index=0
            )

    def after_hc_failure(self) -> None:
        self.convergence_status = False
        self.save_data_time_step()

        if self.time_manager.is_constant:
            # We cannot decrease the constant time step.
            raise ValueError("HC iterations did not converge.")
        else:
            # Update the time step magnitude if the dynamic scheme is used.
            # Note: It will also raise a ValueError if the minimal time step is reached.
            self.time_manager.compute_time_step(recompute_solution=True)

            # Reset the iterate values. This ensures that the initial guess for an
            # unknown time step equals the known time step.
            prev_solution = self.equation_system.get_variable_values(time_step_index=0)
            self.equation_system.set_variable_values(prev_solution, hc_index=0)

    def hc_check_convergence(
        self,
        nl_is_converged: bool,
        hc_params: dict[str, Any],
    ) -> tuple[bool, bool]:
        r"""Check whether the homotopy continuation is converged or diverged.

        We differentiate the following cases:
        - If homotopy continuation error is sufficiently small in comparison to
            the discretization error and the nonlinear solver is converged ->
            Converged.
        - If the maximum number of HC iterations or the minimal of :math:`\lambda` is
            reached without convergence -> Diverged.
        - If the nonlinear solver did not converge on the first HC step (i.e.,
          :math:`\lambda = 1`) -> Diverged.


        """
        # NOTE The following does not need to be evalauted when hc_params["hc_adaptive"]
        # is False. However, to compare HC and apdative HC, we still evaluate it.
        hc_est: float = self.nonlinear_solver_statistics.hc_est[-1][-1]
        discr_est: float = self.nonlinear_solver_statistics.discretization_est[-1][-1]

        # Divergence checks.
        if (
            self.nonlinear_solver_statistics.hc_num_iteration == 0
            and not nl_is_converged
        ):
            logger.info("Nonlinear solver did not converge on first HC step.")
            self.hc_is_diverged = True
        elif (
            self.nonlinear_solver_statistics.hc_num_iteration
            >= hc_params["hc_max_iterations"]
        ):
            logger.info(
                f"Reached maximum number of HC iterations "
                f"{hc_params["hc_max_iterations"]} without convergence."
            )
            self.hc_is_diverged = True

        # Adaptive stopping criterion.
        elif hc_params["hc_adaptive"]:
            # Check if the HC error is smaller than the discretization error.
            if hc_est <= hc_params["hc_error_ratio"] * discr_est and nl_is_converged:
                logger.info(
                    f"HC converged with HC error {hc_est} smaller than "
                    + f" {hc_params['hc_error_ratio']}"
                    + f" * discretization error {discr_est}. "
                )
                self.hc_is_converged = True

        # Non-adaptive stopping criterion.
        elif (
            not hc_params["hc_adaptive"]
            and self.nonlinear_solver_statistics.hc_lambda_fl
            <= hc_params["hc_lambda_min"]
        ):
            logger.info(
                f"HC converged as HC parameter decreased below minimal value"
                f" {hc_params["hc_lambda_min"]}."
            )
            self.hc_is_converged = True

        return self.hc_is_converged, self.hc_is_diverged

    def compute_hc_decay(
        self,
        nl_iterations: Optional[int] = None,
        recompute_decay: bool = False,
    ) -> None:
        """Adjust the decay for the homotopy continuation parameter.

        Parameters:
            hc_params: Homotopy continuation parameters.

        Returns:
            float: Updated lambda.

        """
        if self.hc_constant_decay:
            return None
        else:
            if nl_iterations is not None:
                self._hc_adaptation_based_on_iterations(nl_iterations)
            elif recompute_decay:
                self._hc_adaptation_based_on_recomputation()
            else:
                msg: str = (
                    "Cannot recompute decay because neither `nl_iterations`"
                    + " nor `recompute_decay` are provided."
                )
                raise ValueError(msg)

    def _hc_adaptation_based_on_recomputation(self) -> None:
        """Same as ``pp.TimeManager._adaptation_based_on_recomputation`` but for
        homotopy continuation.

        """
        if self.hc_decay >= self.hc_decay_min_max[1]:
            msg: str = (
                "Recomputation will not have any effect since the hc_decay achieved its"
                + " maximum admissible value -> hc_decay >= hc_decay_max ="
                + f" {self.hc_decay_min_max[1]}."
            )
            logger.info(msg)
            self.hc_is_diverged = True
            return None
        elif self.hc_decay_recomp_counter == self.hc_decay_recomp_max:
            msg = (
                "Reached maximum number of recomputations"
                + f" {self.hc_decay_recomp_max} for the HC decay."
            )
            logger.info(msg)
            self.hc_is_diverged = True
            return None
        self.hc_decay *= self.nl_iter_relax_factors[1]
        self._hc_correction_based_on_hc_decay_min_max()
        self.hc_decay = min(self.hc_decay, self.hc_decay_min_max[1])
        self.hc_decay_recomp_counter += 1
        logger.info(f"Slowing HC decay. Next decay = {self.hc_decay}.")

    def _hc_adaptation_based_on_iterations(self, iterations: int) -> None:
        """Same as ``pp.TimeManager._adaptation_based_on_iterations`` but for
        homotopy continuation.

        """
        if iterations < self.nl_iter_optimal_range[0]:
            self.hc_decay *= self.nl_iter_relax_factors[0]
            logger.info(f"Speeding up HC decay. Next decay = {self.hc_decay}.")
        elif iterations >= self.nl_iter_optimal_range[1]:
            self.hc_decay *= self.nl_iter_relax_factors[1]
            logger.info(f"Slowing HC decay. Next decay = {self.hc_decay}.")
        self._hc_correction_based_on_hc_decay_min_max()
        self.hc_decay_recomp_counter = 0

    def _hc_correction_based_on_hc_decay_min_max(self) -> None:
        if self.hc_decay < self.hc_decay_min_max[0]:
            self.hc_decay = self.hc_decay_min_max[0]
            logger.info(
                "Calculated hc_decay < hc_decay_min. Using hc_decay_min ="
                + f" {self.hc_decay_min_max[0]} instead."
            )
        elif self.hc_decay > self.hc_decay_min_max[1]:
            self.hc_decay = self.hc_decay_min_max[1]
            logger.info(
                "Calculated hc_decay > hc_decay_max. Using hc_decay_max ="
                + f" {self.hc_decay_min_max[1]} instead."
            )

    # endregion

    # region NONLINEAR LOOP
    def before_nonlinear_loop(self) -> None:
        """Set the starting estimate to the solution from the previous continuation
        step."""
        # Update time step size and empty statistics.
        self.ad_time_step.set_value(self.time_manager.dt)
        self.nonlinear_solver_statistics.reset()

        assembled_variables = self.equation_system.get_variable_values(hc_index=0)
        self.equation_system.set_variable_values(
            assembled_variables, iterate_index=0, additive=False
        )
        # FIXME Check convergence once before starting the Newton loop. Perhaps the
        # solution from the previous HC iteration is already good enough. This way, we
        # would avoid one Newton iteration.

    def after_nonlinear_convergence(self) -> None:  # type: ignore
        """Export and move to the next homotopy continuation step.

        When ``self._limit_saturation_change == True``, check if the wetting saturation
        has changed too much

        """
        # Distribute nonlinear solution to hc solution.

        # Primary and secondary variables.
        hc_solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.shift_hc_values(max_index=len(self.hc_indices))
        self.equation_system.set_variable_values(
            hc_solution, hc_index=0, additive=False
        )

        # Reconstructions and estimates.
        g_data = self.mdg.subdomains(return_data=True)[0][1]
        for pressure_key, specifier in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["postprocessed_coeffs", "reconstructed_coeffs", "NC_estimate"],
        ):
            pp.shift_solution_values(
                f"{pressure_key}_{specifier}",
                g_data,
                pp.HC_ITERATE_SOLUTIONS,
                len(self.hc_indices),
            )
            pressure_values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_{specifier}", g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_{specifier}", pressure_values, g_data, hc_index=0
            )
        for flux_name, estimate_name in itertools.product(
            ["total", "wetting_from_ff"],
            ["R_estimate", "F_estimate", "C_estimate", "L_estimate"],
        ):
            pp.shift_solution_values(
                f"{flux_name}_{estimate_name}",
                g_data,
                pp.HC_ITERATE_SOLUTIONS,
                len(self.hc_indices),
            )
            flux_values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{estimate_name}", g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{flux_name}_{estimate_name}", flux_values, g_data, hc_index=0
            )
        for energy_norm_term in [
            "energy_norm_saturation_part",
            "energy_norm_global_pressure_part",
            "energy_norm_complimentary_pressure_part",
        ]:
            pp.shift_solution_values(
                energy_norm_term,
                g_data,
                pp.HC_ITERATE_SOLUTIONS,
                len(self.hc_indices),
            )
            local_term: np.ndarray = pp.get_solution_values(
                energy_norm_term, g_data, iterate_index=0
            )
            pp.set_solution_values(energy_norm_term, local_term, g_data, hc_index=0)

        # Adapt decay rate based on number of nonlinear iterations.
        self.compute_hc_decay(
            nl_iterations=self.nonlinear_solver_statistics.num_iteration
        )

    def after_nonlinear_failure(self) -> None:
        """Method to be called if the non-linear solver fails to converge."""
        if self.hc_constant_decay:
            # We cannot change the constant HC decay.
            self.hc_is_diverged = True
            logger.info(
                "HC decay is constant and cannot be recomputed. Proceeding (if"
                + " possible) with time step recomputation."
            )

        else:
            # Reset lambda and adapt decay rate.
            self.nonlinear_solver_statistics.hc_lambda_fl /= self.hc_decay
            self.compute_hc_decay(recompute_decay=True)

            # Reset the initial guess for the nonlinear solver.
            prev_solution = self.equation_system.get_variable_values(hc_index=0)
            self.equation_system.set_variable_values(prev_solution, iterate_index=0)

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
            self, nonlinear_increment, residual, reference_residual, nl_params
        )

        # NOTE The following does not need to be evaluated when hc_params["hc_adaptive"]
        # is False. However, to compare HC and apdative HC, we still evaluate it.
        hc_est: float = self.global_hc_est()
        linearization_est: float = self.global_linearization_est()
        self.nonlinear_solver_statistics.log_error(
            # NOTE The discretization error estimate does not need to be calculated
            # at this point. After HC convergence is sufficient if we want the code
            # to be more efficient.
            discretization_est=self.global_discretization_est(),
            hc_est=hc_est,
            linearization_est=linearization_est,
        )

        # Adaptive stopping criterion.
        if nl_params["hc_adaptive"]:
            if linearization_est <= nl_params["nl_error_ratio"] * hc_est:
                logger.info(
                    f"Linearization error {linearization_est} smaller than"
                    + f" {nl_params['nl_error_ratio']} * HC error {hc_est}."
                    + " Stopping Newton loop."
                )
                converged = True

        return converged, diverged

    # endregion


# We subclass ``SolutionStrategyEstMixin`` to avoid the following mypy error:
# ``self.iniitialize_estimate_vals`` calls a super method, defined in
# ``SolutionStrategyEstMixin``. In ``EstimatesProtocol``, this method is abstract and
# not implemented, which mypy complains about.
# abstractly
# The various protocols define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SolutionStrategyEstHC(  # type: ignore
    SolutionStrategyHC,
    SolutionStrategyEst,
):

    def __init__(self, params=None) -> None:
        super().__init__(params=params)
        self.hc_rel_perm_toggle: bool = True

    # Initialize time step and iterate values for local estimators.
    @typing.override
    def initialize_estimate_vals(self) -> None:
        super().initialize_estimate_vals()
        sd, g_data = self.mdg.subdomains(return_data=True)[0]
        # Initialize time step values for local estimators.
        for flux_name in ["total", "wetting_from_ff"]:
            pp.set_solution_values(
                f"{flux_name}_C_estimate",
                np.zeros(sd.num_cells),
                g_data,
                time_step_index=0,
                hc_index=0,
                iterate_index=0,
            )
            pp.set_solution_values(
                f"{flux_name}_L_estimate",
                np.zeros(sd.num_cells),
                g_data,
                time_step_index=0,
                hc_index=0,
                iterate_index=0,
            )

    @typing.override
    def eval_additional_vars(self, prepare_simulation: bool = False) -> None:
        """Calculate numerical fluxes w.r.t. the goal relative permeabilities.

        This is done for the fluxes used in pressure reconstruction as well as for
        the fluxes used in the continuation estimator.

        """
        if prepare_simulation:
            time_step_index: Optional[int] = 0
        else:
            time_step_index = None

        g, g_data = self.mdg.subdomains(return_data=True)[0]

        # Switch rel. perm. to goal rel. perm, calculate fluxes, and switch back.
        self.hc_rel_perm_toggle = False

        for flux_name in [
            "total",
            "wetting_from_ff",
            "total_by_t_mobility",
            "total_times_fractional_flow",
        ]:
            if flux_name == "total":
                flux: np.ndarray = self.total_flux(g).value(self.equation_system)
            elif flux_name == "wetting_from_ff":
                flux = self.wetting_flux_from_fractional_flow(g).value(
                    self.equation_system
                )
            elif flux_name == "total_by_t_mobility":
                flux = (self.total_flux(g) / self.total_mobility(g)).value(
                    self.equation_system
                )
            elif flux_name == "total_times_fractional_flow":
                flux = (
                    self.total_flux(g)
                    / self.total_mobility(g)
                    * self.phase_mobility(g, self.wetting)
                ).value(self.equation_system)
            pp.shift_solution_values(
                f"{flux_name}_flux_wrt_goal_rel_perm",
                g_data,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{flux_name}_flux_wrt_goal_rel_perm",
                flux,
                g_data,
                time_step_index=time_step_index,
                iterate_index=0,
            )
        self.hc_rel_perm_toggle = True

    @typing.override
    def postprocess_solution(
        self, nonlinear_increment: np.ndarray, prepare_simulation: bool = False
    ) -> None:
        """Extend and equilibrate fluxes, postprocess and reconstruct pressures."""
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

            # Extend both the nonequilibrated and equilibrated flux to compare in
            # the flux estimator. The nonequilibrated wetting_from_ff flux is also used
            # in the pressure reconstruction.
            self.extend_fv_fluxes(
                flux_name,
            )

            # Equilibration can only be run during Newton.
            if not prepare_simulation:
                # In ``nonlinear_increment``, the saturation variable comes first, then
                # the pressure variable, just as required by
                # ``equilibrate_flux_during_Newton``.
                self.equilibrate_flux_during_Newton(flux_name, nonlinear_increment)

                self.extend_fv_fluxes(
                    flux_name,
                    flux_specifier="_equilibrated",
                )

        # NOTE The fluxes w.r.t. goal rel. perm are only used to post-process the global
        # and complimentary pressures and in the contination estimators and do not need
        # to be equilibrated.
        for flux_name in [
            "total",
            "wetting_from_ff",
            "total_by_t_mobility",
            "total_times_fractional_flow",
        ]:
            # Satisfy mypy.
            flux_name = typing.cast(
                Literal[
                    "total",
                    "wetting_from_ff",
                    "total_by_t_mobility",
                    "total_times_fractional_flow",
                ],
                flux_name,
            )
            self.extend_fv_fluxes(
                flux_name,
                flux_specifier="_wrt_goal_rel_perm",
                prepare_simulation=prepare_simulation,
            )

        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            # Satisfy mypy.
            pressure_key = typing.cast(PRESSURE_KEY, pressure_key)

            self.reconstruct_pressure_vohralik(
                pressure_key,
                flux_specifier="_wrt_goal_rel_perm",
                prepare_simulation=prepare_simulation,
            )

        self.equilibrated_flux_mismatch()


class DataSavingHC(DataSavingEst):

    def _data_to_export(
        self, time_step_index: Optional[int] = None, iterate_index: Optional[int] = None
    ) -> list[DataInput]:
        """Append the continuation and linearization error estimates to the exported
        data.

        """
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index,
            iterate_index=iterate_index,
        )
        # Only export for nonzero time steps or nonlinear steps. Otherwise, this causes
        # an error, as the function is called via
        # ``SolutionStrategyTPF.prepare_simulation`` BEFORE the initial values are set
        # by ``SolutionStrategyEst.prepare_simulation``.
        if (time_step_index is not None and self.time_manager.time_index > 0) or (
            iterate_index is not None
        ):
            g, g_data = self.mdg.subdomains(return_data=True)[0]
            for flux_name, est_name in itertools.product(
                ["total", "wetting_from_ff"], ["C_estimate", "L_estimate"]
            ):
                data.append(
                    (
                        g,
                        f"{flux_name}_{est_name}",
                        pp.get_solution_values(
                            f"{flux_name}_{est_name}",
                            g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
        return data


class TwoPhaseFlowHC(
    RelativePermeabilityHC,
    EstimatesHCMixin,
    SolutionStrategyEstHC,
    DataSavingHC,
    # Estimator mixins:
    ErrorEstimateMixin,
    SolutionStrategyEst,
    # Reconstruction mixins:
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    # The rest
    TwoPhaseFlow,
): ...  # type: ignore
