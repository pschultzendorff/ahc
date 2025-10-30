import itertools
import logging
import typing
from typing import Any, Literal

import numpy as np
import porepy as pp
from porepy.viz.exporter import DataInput

from tpf.models.constitutive_laws_tpf import (
    CapillaryPressure,
    CapPressConstants,
    RelativePermeability,
    RelPermConstants,
)
from tpf.models.error_estimate import (
    DataSavingEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
)
from tpf.models.flow_and_transport import DarcyFluxes, SolutionStrategyTPF, TwoPhaseFlow
from tpf.models.phase import FluidPhase
from tpf.models.protocol import (
    EstimatesProtocol,
    HCProtocol,
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
from tpf.utils.constants_and_typing import (
    COMPLEMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    PRESSURE_KEY,
)

logger = logging.getLogger(__name__)

# If the local estimators are very small (~1e-160), taking a square during their
# computation will result in an underflow error. These errors should NOT be raised.
# Treating the local estimators as zero is fine.
np.seterr(under="ignore")


# ``HCProtocol`` and ``TPFProtocol`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class HybridUpwindingHC(HCProtocol, DarcyFluxes):
    r"""Mobility upwinded for viscous flux and averaged for capillary flux.

    F. P. Hamon, B. T. Mallison, and H. A. Tchelepi, “Implicit Hybrid Upwinding for
          two-phase flow in heterogeneous porous media with buoyancy and
          capillarity,” Computer Methods in Applied Mechanics and Engineering, vol.
          331, pp. 701–727, Apr. 2018, doi: 10.1016/j.cma.2017.10.008.

    """

    # def prepare_simulation(self) -> None:
    #     super().prepare_simulation()
    #     self.calc_capillary_diffusion_interpolants()

    # def calc_capillary_diffusion_interpolants(self) -> None:
    #     hybrid_upwind_constants: dict[str, Any] = self.params.get(
    #         "hc_hybrid_upwind_constants", {}
    #     )
    #     self.s_interpol_vals: np.ndarray = np.linspace(
    #         self.wetting.residual_saturation + self.wetting.saturation_epsilon,
    #         1
    #         - self.nonwetting.residual_saturation
    #         - self.nonwetting.saturation_epsilon,
    #         hybrid_upwind_constants.get("interpolation_degree", 100),
    #     )

    #     def capillary_diffusion(s_w: np.ndarray) -> np.ndarray:
    #         return (
    #             self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #             * self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #             / (
    #                 self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #                 * self.wetting.viscosity
    #                 / self.nonwetting.viscosity
    #                 * self.rel_perm_np(s_w, self.wetting, self._rel_perm_constants_1)
    #             )
    #             * self.cap_press_deriv_np(s_w, self._cap_press_constants_1)
    #         )

    #     self.capillary_diffusion_vals: np.ndarray = capillary_diffusion(
    #         self.s_interpol_vals
    #     )

    # def capillary_flux(self, g: pp.Grid) -> pp.ad.Operator:
    #     tpfa = pp.ad.TpfaAd(self.flux_key, [g])
    #     diffusion_coeff = np.interp(
    #         self.s_interpol_vals, self.capillary_diffusion_vals
    #     )(self.wetting.s)
    #     cap_flux = (
    #         tpfa.flux() @ self.wetting.s
    #         + tpfa.bound_flux() @ self.bc_dirichlet_saturation_values(g, self.wetting)
    #     )


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

    @typing.override
    def rel_perm(
        self,
        saturation_w: pp.ad.Operator,
        phase: FluidPhase,
        # Ignore mypy complaining about incompatible signature with supertype.
    ) -> pp.ad.Operator:  # type: ignore[override]
        # Return homotopy continuation relative permeability.
        # Mypy gives an error here, because it thinks the empty ``HCProtocol.rel_perm``
        # is called. During runtime, ``HCProtocol`` does not have this method, hence we
        # can ignore the error. Additionally, we ignore complaints about the wrong
        # number of arguments and wrong arguemnt types.
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
        hc_rel_perm: pp.ad.Operator = (
            self.nonlinear_solver_statistics.hc_lambda_ad * rel_perm_1
            + (pp.ad.Scalar(1) - self.nonlinear_solver_statistics.hc_lambda_ad)
            * rel_perm_2
        )
        return (
            self.hc_toggle_ad * hc_rel_perm
            + (pp.ad.Scalar(1) - self.hc_toggle_ad) * rel_perm_2
        )

    @typing.override
    def rel_perm_np(
        self,
        saturation_w: np.ndarray,
        phase: FluidPhase,
    ) -> np.ndarray:
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
        hc_rel_perm: np.ndarray = (
            self.nonlinear_solver_statistics.hc_lambda_fl * rel_perm_1
            + (1 - self.nonlinear_solver_statistics.hc_lambda_fl) * rel_perm_2
        )
        return self.hc_toggle_fl * hc_rel_perm + (1 - self.hc_toggle_fl) * rel_perm_2


# ``HCProtocol`` and ``TPFProtocol`` define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class CapillaryPressureHC(HCProtocol, CapillaryPressure):  # type: ignore
    @typing.override
    def set_cap_press_constants(self) -> None:
        cap_press_constants: dict[str, dict] = self.params.get(
            "cap_press_constants", {}
        )
        cap_press_1_constants: dict[str, Any] = cap_press_constants.get("model_1", {})
        cap_press_2_constants: dict[str, Any] = cap_press_constants.get("model_2", {})

        self._cap_press_constants_1 = CapPressConstants(**cap_press_1_constants)
        self._cap_press_constants_2 = CapPressConstants(**cap_press_2_constants)

    @typing.override
    def cap_press(
        self,
        saturation_w: pp.ad.Operator,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs,
        # Ignore mypy complaining about incompatible signature with supertype.
    ) -> pp.ad.Operator:  # type: ignore[override]
        # Return homotopy continuation capillary pressure.
        # Mypy gives an error here, because it thinks the empty ``HCProtocol.cap_press``
        # is called. During runtime, ``HCProtocol`` does not have this method, hence we
        # can ignore the error. Additionally, we ignore complaints about the wrong
        # number of arguments and wrong arguemnt types.
        cap_press_1: pp.ad.Operator = super().cap_press(  # type: ignore
            saturation_w,  # type: ignore
            cap_press_constants=self._cap_press_constants_1,  # type: ignore
            **kwargs,
        )
        cap_press_2: pp.ad.Operator = super().cap_press(  # type: ignore
            saturation_w,  # type: ignore
            cap_press_constants=self._cap_press_constants_2,  # type: ignore
            **kwargs,
        )
        hc_cap_press: pp.ad.Operator = (
            self.nonlinear_solver_statistics.hc_lambda_ad * cap_press_1
            + (pp.ad.Scalar(1) - self.nonlinear_solver_statistics.hc_lambda_ad)
            * cap_press_2
        )
        return (
            self.hc_toggle_ad * hc_cap_press
            + (pp.ad.Scalar(1) - self.hc_toggle_ad) * cap_press_2
        )

    @typing.override
    def cap_press_np(
        self,
        saturation_w: np.ndarray,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs,
    ) -> np.ndarray:
        cap_press_1: np.ndarray = super().cap_press_np(  # type: ignore
            saturation_w,  # type: ignore
            cap_press_constants=self._cap_press_constants_1,  # type: ignore
            **kwargs,
        )
        cap_press_2: np.ndarray = super().cap_press_np(  # type: ignore
            saturation_w,  # type: ignore
            cap_press_constants=self._cap_press_constants_2,  # type: ignore
            **kwargs,
        )
        hc_cap_press: np.ndarray = (
            self.nonlinear_solver_statistics.hc_lambda_fl * cap_press_1
            + (1 - self.nonlinear_solver_statistics.hc_lambda_fl) * cap_press_2
        )
        return self.hc_toggle_fl * hc_cap_press + (1 - self.hc_toggle_fl) * cap_press_2

    # Ensure that global and complementary pressure tables are calculated with the
    # target capillary pressure function.
    @typing.override
    def cap_press_deriv_np(
        self,
        saturation_w: np.ndarray,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return super().cap_press_deriv_np(
            saturation_w, cap_press_constants=self._cap_press_constants_2, **kwargs
        )


# The various protocols define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class EstimatesHCMixin(
    HCProtocol, EstimatesProtocol, ReconstructionProtocol, TPFProtocol
):  # type: ignore
    def local_temp_est(self, flux_name: Literal["total", "wetting_from_ff"]) -> None:
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
        Importantly, both are w.r.t. to the goal relative permeability.

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
            f"{flux_name}_flux_wrt_goal_rel_perm_RT0_coeffs",
            self.g_data,
            iterate_index=0,
        )
        fv_coeffs_old = pp.get_solution_values(
            f"{flux_name}_flux_wrt_goal_rel_perm_RT0_coeffs",
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
        integral: Integral = self.quadrature_estimate.integrate(
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

    def local_hc_est(self, flux_name: Literal["total", "wetting_from_ff"]) -> None:
        """

        We assume the following sub-dictionaries to be present in the data dictionary:
            ``iterate_dictionary``, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in ``iterate_dictionary`` will be updated:
            - ``{flux_name}_C_estimator``, storing the local in time and space
              continuation error estimator.

        """
        # Retrieve flux w.r.t. goal rel. perm. and nonequilbirated flux coeffs.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", self.g_data, iterate_index=0
        )
        fv_goal_rel_perm_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_wrt_goal_rel_perm_RT0_coeffs",
            self.g_data,
            iterate_index=0,
        )

        def integrand(x: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs[..., 0] - fv_goal_rel_perm_coeffs[..., 0]
            ) * x[..., 0] + (fv_coeffs[..., 1] - fv_goal_rel_perm_coeffs[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs[..., 0] - fv_goal_rel_perm_coeffs[..., 0]
            ) * x[..., 1] + (fv_coeffs[..., 2] - fv_goal_rel_perm_coeffs[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise and store the result.
        integral: Integral = self.quadrature_estimate.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"{flux_name}_C_estimator",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def local_linearization_est(
        self, flux_name: Literal["total", "wetting_from_ff"]
    ) -> None:
        """

        We assume the following sub-dictionaries to be present in the data dictionary:
            ``iterate_dictionary``, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in ``iterate_dictionary`` will be updated:
            - ``{flux_name}_L_estimator``, storing the local in time and space
              linearization error estimator.

        """
        # Retrieve flux w.r.t. goal rel. perm. and equilbirated flux coeffs.
        # TODO Could be made more efficient by just storing the Newton update times
        # derivative of the fluxes.
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
        integral: Integral = self.quadrature_estimate.integrate(
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
        is decomposed and separated into the temporal, continuation and linearization
        estimator.

        The remaining residual error estimate is zero in theory and negligible in
        practice. For faster evaluation, it may not be evaluated.

        Note: The residual estimator is not time dependent, hence we multiply the
        value at :math:`t_n` by :math:`\Delta t` to get the time integral.

        Returns:
            estimator: Global discretization error estimator.

        """
        if self.params.get("hc_fast_evaluation", True):
            return 0.0
        else:
            estimators: dict[str, float] = {}
            for flux_name in ["total", "wetting_from_ff"]:
                # Satisfy mypy.
                flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

                # Calculate local estimatorss.
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
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

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
        spatial_estimator: float = self.nonlinear_solver_statistics.spatial_est[-1][-1]
        temp_estimator: float = self.nonlinear_solver_statistics.temp_est[-1][-1]
        estimator: float = spatial_estimator + temp_estimator
        logger.info(f"Global discretization error estimator: {estimator}")
        return estimator

    def global_hc_est(self) -> float:
        """Calculate the global in space and semi-global (i.e, integrated over one time
        step) continuation error estimator. Summed over both fluxes.

        Note: The local estimators are constant for the entire time step. Thus, they are
        summed over the domain and multiplied with the time step size.

        Returns:
            est: Global homotopy continuation error estimator.

        """
        estimators: dict[str, float] = {}
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

            # Calculate local estimators.
            self.local_hc_est(flux_name)
            # Load spatial integral from current nonlinear iteration.
            local_integral: np.ndarray = pp.get_solution_values(
                f"{flux_name}_C_estimator", self.g_data, iterate_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # do not need to square here.
            global_integral: float = local_integral.sum()
            # Integrate in time by multiplying with time step size.
            estimators[flux_name] = self.time_manager.dt * global_integral
        # Sum estimators for both equations.
        estimator: float = sum(estimators.values()) ** (1 / 2)
        logger.info(f"Global continuation error estimator: {estimator}")
        return estimator

    def global_linearization_est(self) -> float:
        estimators: dict[str, float] = {}
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

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

    def relative_global_discretization_est(self) -> float:
        """Return relative global discretization error estimator."""
        return self.global_discretization_est() / self.global_energy_norm()

    def relative_global_hc_est(self) -> float:
        """Return relative global homotopy continuation error estimator."""
        return self.global_hc_est() / self.global_energy_norm()

    def relative_global_linearization_est(self) -> float:
        """Return relative global linearization error estimator."""
        return self.global_linearization_est() / self.global_energy_norm()

    def total_est(self) -> float:
        """Return total error estimator, consisting of discretization, homotopy
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
class SolutionStrategyAHC(
    HCProtocol, EstimatesProtocol, ReconstructionProtocol, SolutionStrategyTPF
):  # type: ignore
    def __init__(self, params=None) -> None:
        super().__init__(params=params)
        self.hc_toggle_fl: float = 1.0
        self.hc_toggle_ad: pp.ad.Scalar = pp.ad.Scalar(1.0)

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
        # Switch to goal cap. press. and rel. perms. to calculate interpolants for
        # global and complementary pressure.
        self.hc_toggle_fl = 0.0
        self.hc_toggle_ad.set_value(self.hc_toggle_fl)

        # This is mixed with more Solutionstrategies that implement
        # ``prepare_simulation``. We ignore the mypy error.
        super().prepare_simulation()  # type: ignore

        # Switch back.
        self.hc_toggle_fl = 1.0
        self.hc_toggle_ad.set_value(self.hc_toggle_fl)

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

    @typing.override
    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        self.equation_system.set_variable_values(
            np.full(self.g.num_cells * 2, 0.0),
            [self.wetting.p, self.nonwetting.p],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )
        self.equation_system.set_variable_values(
            np.full(self.g.num_cells * 2, 0.5),
            [self.wetting.s, self.nonwetting.s],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )

    def set_initial_estimators(self) -> None:
        """Initialize time step and iterate values for local continuation and
        linearization estimators.

        """
        # NOTE The super call does **NOT** set initial values for the local flux,
        # residual, and nonconformity estimators at the ``hc_index``. This is never
        # needed, hence not an issue.
        # In ``EstimatesProtocol``, this method is abstract and
        # not implemented, which mypy complains about
        super().set_initial_estimators()  # type: ignore

        # Initialize iterate values for local estimators.
        # NOTE HC and time step values for the continuation and linearization estimators
        # are not needed at any point. The initial iterate values are technically also
        # not needed, but by settimg them, they are exported at the zeroth time step.
        # Check :meth:`DataSavingHC._data_to_export`.
        for flux_name in ["total", "wetting_from_ff"]:
            pp.set_solution_values(
                f"{flux_name}_C_estimator",
                np.zeros(self.g.num_cells),
                self.g_data,
                iterate_index=0,
            )
            pp.set_solution_values(
                f"{flux_name}_L_estimator",
                np.zeros(self.g.num_cells),
                self.g_data,
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
            "Decayed hc_lambda to"
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
        # Update the time step magnitude if the dynamic scheme is used.
        if not self.time_manager.is_constant:
            self.time_manager.compute_time_step(
                iterations=self.nonlinear_solver_statistics.hc_num_iteration
            )

        # Set time step values for reconstructions and estimators.
        for pressure_key, specifier in itertools.product(
            [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE],
            ["", "_postprocessed_coeffs", "_reconstructed_coeffs", "_NC_estimator"],
        ):
            pressure_values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}{specifier}", self.g_data, hc_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}{specifier}",
                pressure_values,
                self.g_data,
                time_step_index=0,
            )
        # NOTE The local residual, continuation, and linearization error estimators are
        # only needed at the current iteration. We set the time step values for
        # completeness and to avoid extra code in :meth:`_data_to_export`.
        for flux_name, specifier in itertools.product(
            ["total", "wetting_from_ff"],
            [
                "R_estimator",
                "F_estimator",
                "T_estimator",
                "C_estimator",
                "L_estimator",
                "flux_wrt_goal_rel_perm_RT0_coeffs",
            ],
        ):
            if specifier.startswith("energy"):
                name: str = f"energy_norm_{flux_name}_flux_part"
            else:
                name = f"{flux_name}_{specifier}"
            flux_values: np.ndarray = pp.get_solution_values(
                name, self.g_data, hc_index=0
            )
            pp.set_solution_values(name, flux_values, self.g_data, time_step_index=0)

        # Save only after the time step values are updated..
        self.save_data_time_step()

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
        nl_is_diverged: bool,
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
        # Divergence checks.
        if (
            not nl_is_converged or nl_is_diverged
        ) and self.nonlinear_solver_statistics.hc_num_iteration == 1:
            # If the nonlinear solver did not converge or diverged on the first HC step,
            # :meth:`after_hc_iteration` was already run, hence `hc_num_iteration` is
            # updated to one and not zero.
            logger.info("Nonlinear solver did not converge on first HC step.")
            self.hc_is_diverged = True

        # Convergence checks.
        elif nl_is_converged:
            # Adaptive stopping criterion. Check if the HC error is smaller than the
            # discretization error.
            if hc_params["hc_adaptive"]:
                hc_est: float = self.nonlinear_solver_statistics.hc_est[-1][-1]
                discr_est: float = self.global_discretization_est()
                if hc_est <= hc_params["hc_error_ratio"] * discr_est:
                    logger.info(
                        f"HC converged with HC error {hc_est} smaller than "
                        + f" {hc_params['hc_error_ratio']}"
                        + f" * discretization error {discr_est}. "
                    )
                    self.hc_is_converged = True
            # Non-adaptive stopping criterion.
            elif (
                not hc_params["hc_adaptive"]
                # Check the lambda before decay, which was actually solved.
                and self.nonlinear_solver_statistics.hc_lambda_fl / self.hc_decay
                <= hc_params["hc_lambda_min"]
            ):
                logger.info(
                    f"HC converged as HC parameter decreased below minimal value"
                    f" {hc_params['hc_lambda_min']}."
                )
                self.hc_is_converged = True

        # If not converged, but maximum number of HC iterations is reached, the HC loop
        # is diverged.
        elif (
            self.nonlinear_solver_statistics.hc_num_iteration
            >= hc_params["hc_max_iterations"]
        ):
            logger.info(
                f"Reached maximum number of HC iterations "
                f"{hc_params['hc_max_iterations']} without convergence."
            )
            self.hc_is_diverged = True

        return self.hc_is_converged, self.hc_is_diverged

    def compute_hc_decay(
        self,
        nl_iterations: int | None = None,
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
        """Set the starting estimator to the solution from the previous continuation
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

    def eval_additional_vars(self, time_step_index: int | None = None) -> None:
        """Calculate numerical fluxes w.r.t. the goal relative permeabilities.

        This is done for the fluxes used in pressure reconstruction as well as for
        the fluxes used in the continuation estimator.

        """

        # Save FV P0 pressures.
        self.eval_glob_compl_pressure_on_domain(time_step_index=time_step_index)

        # Switch rel. perm. to goal rel. perm, calculate fluxes.
        self.hc_toggle_fl = 0.0
        self.hc_toggle_ad.set_value(self.hc_toggle_fl)

        for flux_name, flux_eq in zip(
            [
                "total",
                "wetting_from_ff",
                "total_by_t_mobility",
                "total_times_fractional_flow",
            ],
            [
                self.total_flux_eq,
                self.wetting_flux_from_ff_eq,
                self.total_flux_by_total_mobility_eq,
                self.total_flux_times_fractional_flow_eq,
            ],
        ):
            # Take the negative of the values since ``equation_system.assemble`` returns
            # the negative of the RHS.
            flux: np.ndarray = -self.equation_system.assemble(
                evaluate_jacobian=False, equations=[flux_eq]
            )
            pp.set_solution_values(
                f"{flux_name}_flux_wrt_goal_rel_perm",
                flux,
                self.g_data,
                time_step_index=time_step_index,
                iterate_index=0,
            )

        # Switch back to hc rel perm.
        self.hc_toggle_fl = 1.0
        self.hc_toggle_ad.set_value(self.hc_toggle_fl)

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
        # and complementary pressures and in the contination estimators and do not need
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

        for pressure_key in [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE]:
            # Satisfy mypy.
            pressure_key = typing.cast(PRESSURE_KEY, pressure_key)
            self.postprocess_pressure_vohralik(
                pressure_key,
                flux_specifier="_wrt_goal_rel_perm",
                prepare_simulation=prepare_simulation,
            )
            self.reconstruct_pressure_vohralik(
                pressure_key,
                prepare_simulation=prepare_simulation,
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
            self, nonlinear_increment, residual, reference_residual, nl_params
        )

        # Switch rel. perm. to goal rel. perm. to evaluate the nonconformity estimators.
        self.hc_toggle_fl = 0.0
        self.hc_toggle_ad.set_value(self.hc_toggle_fl)

        # NOTE The following does not need to be evaluated when hc_params["hc_adaptive"]
        # is False. However, to compare HC and apdative HC, we still evaluate it.
        hc_est: float = self.global_hc_est()
        linearization_est: float = self.global_linearization_est()
        self.nonlinear_solver_statistics.log_error(
            # NOTE The discretization error estimate does not need to be calculated
            # at this point. After HC convergence is sufficient if we want the code
            # to be more efficient.
            global_energy_norm=self.global_energy_norm(),
            equilibrated_flux_mismatch=self.equilibrated_flux_mismatch(),
            spatial_est=self.global_spatial_est(),
            temp_est=self.global_temp_est(),
            hc_est=hc_est,
            linearization_est=linearization_est,
        )

        # Switch rel. perm. back.
        self.hc_toggle_fl = 1.0
        self.hc_toggle_ad.set_value(self.hc_toggle_fl)

        # Adaptive stopping criterion.
        if not diverged and nl_params["hc_adaptive"]:
            nonlinear_increment_norm: float = self.compute_nonlinear_increment_norm(
                nonlinear_increment
            )

            # If Newton diverges, the estimators lose their meaning and the adaptive
            # criterion might incorrectly stop the HC loop. Hence, we check that the
            # nonlinear increment norm is not too large.
            if (
                linearization_est <= nl_params["nl_error_ratio"] * hc_est
                and nonlinear_increment_norm <= nl_params["hc_nl_convergence_tol"]
            ):
                logger.info(
                    f"Linearization error {linearization_est} smaller than"
                    + f" {nl_params['nl_error_ratio']} * HC error {hc_est}."
                    + " Stopping Newton loop."
                )
                converged = True

        return converged, diverged

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

        # Reconstructions and estimators.
        for pressure_key, specifier in itertools.product(
            [GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE],
            ["", "_postprocessed_coeffs", "_reconstructed_coeffs", "_NC_estimator"],
        ):
            pressure_values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}{specifier}", self.g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}{specifier}", pressure_values, self.g_data, hc_index=0
            )
        for flux_name, specifier in itertools.product(
            ["total", "wetting_from_ff"],
            [
                "R_estimator",
                "F_estimator",
                "T_estimator",
                "C_estimator",
                "L_estimator",
                "energy_norm_flux_part",
                "flux_wrt_goal_rel_perm_RT0_coeffs",
            ],
        ):
            if specifier.startswith("energy"):
                name: str = f"energy_norm_{flux_name}_flux_part"
            else:
                name = f"{flux_name}_{specifier}"
            flux_values: np.ndarray = pp.get_solution_values(
                name, self.g_data, iterate_index=0
            )
            pp.set_solution_values(name, flux_values, self.g_data, hc_index=0)

        # Adapt decay rate based on number of nonlinear iterations. Do this only AFTER
        # at least one succesfull decay.
        if self.nonlinear_solver_statistics.hc_num_iteration >= 1:
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
            # No need to reset the initial guess for the nonlinear solver, as this is
            # done by :meth:`before_nonlinear_loop`. anyways.
            # TODO Possible failure to do this?

    # endregion


class DataSavingHC(DataSavingEst):
    def _data_to_export(
        self, time_step_index: int | None = None, iterate_index: int | None = None
    ) -> list[DataInput]:
        """Append the continuation and linearization error estimators to the exported
        data.

        """
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index,
            iterate_index=iterate_index,
        )
        for flux_name, est_name in itertools.product(
            ["total", "wetting_from_ff"], ["T_estimator", "C_estimator", "L_estimator"]
        ):
            try:
                data.append(
                    (
                        self.g,
                        f"{flux_name}_{est_name}",
                        pp.get_solution_values(
                            f"{flux_name}_{est_name}",
                            self.g_data,
                            iterate_index=iterate_index,
                            time_step_index=time_step_index,
                        ),
                    )
                )
            except KeyError:
                pass
        return data


class TwoPhaseFlowAHC(
    # HC constitutive laws mixins:
    RelativePermeabilityHC,
    CapillaryPressureHC,
    # DarcyFluxesHC,
    # Adaptive HC mixins:
    EstimatesHCMixin,
    SolutionStrategyAHC,
    DataSavingHC,
    # Estimator mixins:
    ErrorEstimateMixin,
    SolutionStrategyEst,
    # Reconstruction mixins:
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    EquationsRecMixin,
    # The rest
    TwoPhaseFlow,
): ...  # type: ignore
