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
    COMPLIMENTARY_PRESSURE,
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
class DarcyFluxesHC(HCProtocol, DarcyFluxes):
    r"""Mobility is averaged when :math:`\beta=1` and upwinded when :math:`\beta=0`."""

    def phase_mobility(
        self,
        g: pp.Grid,
        phase: FluidPhase,
    ) -> pp.ad.Operator:
        r"""See :meth:`tpf.models.flow_and_transport.DarcyFluxes.phase_mobility` for
        documentation of the upwinded case. Here, we add cell-averaging.

        """
        upwinded_mobility = super().phase_mobility(g, phase)

        saturation_w = self.wetting.s
        saturation_w_bc = pp.ad.DenseArray(
            self.bc_dirichlet_saturation_values(g, self.wetting),
            name=f"{self.wetting.name}_s_bc",
        )
        viscosity = pp.ad.Scalar(phase.viscosity, name=f"{phase.name}_viscosity")

        # Take mobility in cells, map to faces and divide by 2 to obtain averaged
        # mobility.
        cells_to_faces: pp.ad.Operator = pp.ad.SparseArray(g.cell_faces)

        # Create a mask hiding all internal faces.
        boundary_faces_mask_np: np.ndarray = np.zeros(g.num_faces)
        boundary_faces_mask_np[g.get_boundary_faces()] = 1.0
        boundary_faces_mask: pp.ad.Operator = pp.ad.DenseArray(boundary_faces_mask_np)

        # The bc rel. perms. are defined on all faces, we just mask the internal faces.
        # The internal rel. perms. are defined on cells and mapped to faces.
        averaged_mobility: pp.ad.Operator = (
            cells_to_faces @ self.rel_perm(saturation_w, phase)
            + boundary_faces_mask * self.rel_perm(saturation_w_bc, phase)
        ) / (pp.ad.Scalar(2) * viscosity)

        # Flux on Neumann faces is treated in :meth:`wetting_flux_from_fractional_flow`
        # and :meth:`total_flux`. The mobility at these faces is set to zero to not
        # introduce any inconsistencies.
        dir_faces_mask_np: np.ndarray = np.zeros(g.num_faces)
        dir_faces_mask_np[self.bc_type(g).is_dir] = 1.0
        dir_faces_mask: pp.ad.Operator = pp.ad.DenseArray(dir_faces_mask_np)
        averaged_mobility *= dir_faces_mask

        # Add some epsilon to avoid zero mobility.
        averaged_mobility += pp.ad.Scalar(1e-7)

        hc_mobility: pp.ad.Operator = (
            self.nonlinear_solver_statistics.hc_lambda_ad * averaged_mobility
            + (pp.ad.Scalar(1) - self.nonlinear_solver_statistics.hc_lambda_ad)
            * upwinded_mobility
        )
        hc_mobility.set_name(f"{phase.name} hc mobility")

        return (
            self.hc_toggle_ad * hc_mobility
            + (pp.ad.Scalar(1) - self.hc_toggle_ad) * upwinded_mobility
        )


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
    def rel_perm(
        self, saturation_w: pp.ad.Operator, phase: FluidPhase
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

    # Ignore mypy complaining about incompatible signature with supertype.
    @typing.override
    def cap_press(
        self,
        saturation_w: pp.ad.Operator,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs,
    ) -> pp.ad.Operator:  # type: ignore[override]
        # Return homotopy continuation relative permeability.
        # Mypy gives an error here, because it thinks the empty ``HCProtocol.rel_perm``
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
        # The global residual estimator is the square root of the sum over both fluxes,
        # integral over time and domain of the squared local estimators. Its part in the
        # global error estimator multiplied by 3 before taking the square root. Instead
        # of writing
        # estimator: float = (3 * residual_estimator**2) ** (1 / 2) +
        # sum(nc_estimators),
        # we multiply by np.sqrt(3).
        estimator: float = np.sqrt(3) * residual_estimator + sum(nc_estimators)
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
            # zero at :math:`t^n` and calcualting the integral exactly.
            estimators[flux_name] = self.time_manager.dt / 3.0 * global_integral
        # Sum estimators for both equations.
        estimator: float = (3 * sum(estimators.values())) ** (1 / 2)
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
        estimator: float = (3 * sum(estimators.values())) ** (1 / 2)
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
        estimator: float = (3 * sum(estimators.values())) ** (1 / 2)
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
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
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
                and self.nonlinear_solver_statistics.hc_lambda_fl
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

    def eval_additional_vars(self, prepare_simulation: bool = False) -> None:
        """Calculate numerical fluxes w.r.t. the goal relative permeabilities.

        This is done for the fluxes used in pressure reconstruction as well as for
        the fluxes used in the continuation estimator.

        """
        if prepare_simulation:
            time_step_index: int | None = 0
        else:
            time_step_index = None

        # Save FV P0 pressures.
        self.eval_glob_compl_pressure_on_domain(
            GLOBAL_PRESSURE, prepare_simulation=prepare_simulation
        )
        self.eval_glob_compl_pressure_on_domain(
            COMPLIMENTARY_PRESSURE, prepare_simulation=prepare_simulation
        )

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
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
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
