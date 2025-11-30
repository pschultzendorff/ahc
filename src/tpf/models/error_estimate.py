import functools
import itertools
import logging
import typing
from typing import Any, Literal

import numpy as np
import porepy as pp
import scipy as sp
from porepy.viz.exporter import DataInput

from tpf.models.protocol import EstimatesProtocol, ReconstructionProtocol, TPFProtocol
from tpf.models.reconstruction import (
    DataSavingRec,
    SolutionStrategyRec,
    TwoPhaseFlowReconstruction,
    evaluate_poly_at_points,
)
from tpf.numerics.quadrature import Integral, TriangleQuadrature
from tpf.utils.constants_and_typing import (
    COMPLEMENTARY_PRESSURE,
    FLUX_NAME,
    GLOBAL_PRESSURE,
    PRESSURE_KEY,
    TOTAL_FLUX,
    WETTING_FLUX,
)

logger = logging.getLogger(__name__)

# If the local estimators are very small (~1e-160), taking a square during their
# computation will result in an underflow error. These errors should NOT be raised.
# Treating the local estimators as zero is fine.
np.seterr(under="ignore")


class ErrorEstimateMixin(ReconstructionProtocol, TPFProtocol):
    """Methods to equilibrate fluxes during the Newton iteration.

    Note: If the grid is updated during the simulation, the cellwise Poincare constants
    need to be updatedas well. This is not done by the current implementation.

    """

    DEFAULT_EST_QUAD_DEGREE_EST = 4

    def setup_estimates(self) -> None:
        """Degree of quadrature rule for the error estimators."""
        self.quadrature_est = TriangleQuadrature(self.DEFAULT_EST_QUAD_DEGREE_EST)
        """Quadrature rule for the error estimators.

        Note: For efficiency, :attr:`self.quadrature_rec` could be used.
            However, we may want to have the flexibility to use different degrees for
            estimates and reconstruction.

        """
        self.poincare_constants: np.ndarray = self.poincare_constant(self.g)

    @staticmethod
    def poincare_constant(g: pp.Grid) -> np.ndarray:
        return g.cell_diameters() / np.pi

    def local_residual_est(self, flux_name: FLUX_NAME) -> None:
        r"""Calculate and store the local residual estimator for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,
        .. math::
            \|q_t - \nabla \cdot \theta_t\|_K^2, \\
            \|\varphi \partial_t s_w - q_w+ \nabla \cdot \theta_w\|_K^2.

        """
        # To get the average source term for each element, we divide by element volumes.
        # Integrating gives the element source again.
        source: np.ndarray = (
            self.phase_fluid_source(self.g, self.wetting)
            if flux_name == "wetting"
            else self.total_fluid_source(self.g)
        ) / self.g.cell_volumes

        equilibrated_flux_coeffs: np.ndarray = pp.get_solution_values(
            f"{flux_name}_equil_RT0_coeffs",
            self.g_data,
            iterate_index=0,
        )
        # Divergence of a Raviart-Thomas basis function is given by twice the linear
        # coefficient:
        # :math:`\nabla \cdot \begin{pmatrix} ax + b \\ ay + c\end{pmatrix} = 2 a`
        div_equilibrated_flux: np.ndarray = 2 * equilibrated_flux_coeffs[:, 0]

        # Integrate elementwise and store the result.
        # NOTE The saturation term, source term, and divergence of the flux
        # reconstruction are all elementwise constant. Therefore, we can integrate
        # explicitely by multiplying with the element volume.
        if flux_name == TOTAL_FLUX:
            integral = Integral(
                (
                    self.g.cell_volumes
                    * self.poincare_constants
                    * (div_equilibrated_flux - source) ** 2
                )[..., None]
            )
        elif flux_name == WETTING_FLUX:
            dt = pp.ad.Scalar(self.time_manager.dt)
            dt_s_ad = pp.ad.time_derivatives.dt(self.wetting.s, dt)
            dt_s: np.ndarray = (dt_s_ad).value(self.equation_system)  # type: ignore
            porosity: np.ndarray = self.porosity(self.g)
            integral: Integral = Integral(
                (
                    self.g.cell_volumes
                    * self.poincare_constants
                    * (porosity * dt_s + div_equilibrated_flux - source) ** 2
                )[..., None]
            )

        pp.set_solution_values(
            f"{flux_name}_R_estimator",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def local_flux_est(self, flux_name: FLUX_NAME) -> None:
        r"""Calculate and store the local flux estimator for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,

        .. math::
            \|\mathbf{u}_{\alpha,h,\tau}(t) - \theta_\alpha(t)^{n,i,k}\|_K^2,

        where:
        - :math:`\mathbf{u}_{\alpha,h,\tau}(t)` is the FV flux at time :math:`t`,
            i.e., in :math:`\textbf{RTN}_0(\mathca{T}_h)` but not locally mass
            conservative
        - :math:`\theta_\alpha(t)^{n,i,k}` is the equilibrated flux at the current
            time step, continuation iteration, and Newton iteration, i.e., both in
            :math:`\textbf{RTN}_0(\mathca{T}_h)` and locally mass conservative.

        :math:`\mathbf{u}_{\alpha,h,\tau}(t)` is time dependent. To integrate in time,
        we use the trapezoidal rule. Thus, the local estimators have to be evaluated
        both for the current iteration and the previous time step. **Crucially**, it
        is insufficient to evaluate the current local estimators at each iteration and
        store after the time step converges. This is, because the equilibrated flux from
        the previous time step differs from the current one.

        """
        # Retrieve FV and equilibrated flux coefficients.
        fv_coeffs_new = pp.get_solution_values(
            f"{flux_name}_RT0_coeffs", self.g_data, iterate_index=0
        )
        fv_coeffs_old = pp.get_solution_values(
            f"{flux_name}_RT0_coeffs", self.g_data, time_step_index=0
        )
        equil_coeffs = pp.get_solution_values(
            f"{flux_name}_equil_RT0_coeffs", self.g_data, iterate_index=0
        )

        def integrand(x: np.ndarray, fv_coeffs: np.ndarray) -> np.ndarray:
            coeffs_diff = fv_coeffs - equil_coeffs
            flux_diff = self._evaluate_flux_at_points(coeffs_diff, x[..., 0], x[..., 1])
            return (flux_diff**2).sum(axis=-1)

        # Integrate in space at current and previous time step and store the result.
        for time_step in ["_new", "_old"]:
            integral: Integral = self.quadrature_est.integrate(
                functools.partial(
                    integrand,
                    fv_coeffs=fv_coeffs_new if time_step == "_new" else fv_coeffs_old,
                ),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{flux_name}_F_estimator{time_step}",
                integral.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    def local_pressure_potential(self, pressure_key: PRESSURE_KEY) -> None:
        def evaluate_potential_from_coeffs(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            """Calculate the total flux from reconstructed pressures and P0
            mobilities."""
            return self._evaluate_pressure_potential_at_points(
                pressure_coeffs,
                x[..., 0],
                x[..., 1],
            )

        pressure_coeffs_postproc = pp.get_solution_values(
            f"{pressure_key}_coeffs_postproc", self.g_data, iterate_index=0
        )
        pressure_coeffs_rec = pp.get_solution_values(
            f"{pressure_key}_coeffs_rec", self.g_data, iterate_index=0
        )

        def integrand(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            pressure_potential = evaluate_potential_from_coeffs(x, pressure_coeffs)
            return np.sqrt((pressure_potential**2).sum(axis=-1))

        # 5: Finally, integrate in space. Store norm at current time step and inner
        # product of previous and current time step.
        for specifier in ["_postproc", "_rec"]:
            pressure_coeffs = (
                pressure_coeffs_postproc
                if specifier == "_postproc"
                else pressure_coeffs_rec
            )
            integral: Integral = self.quadrature_est.integrate(
                functools.partial(integrand, pressure_coeffs=pressure_coeffs),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{pressure_key}{specifier}_potential",
                integral.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    def local_saturation_min_max(self, specifier: str) -> None:
        pressure_coeffs = pp.get_solution_values(
            f"{COMPLEMENTARY_PRESSURE}_coeffs_{specifier}", self.g_data, iterate_index=0
        )

        def integrand_min(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            pressure = evaluate_poly_at_points(pressure_coeffs, x[..., 0], x[..., 1])
            saturation = self.eval_saturation(pressure)
            return np.repeat(np.min(saturation, axis=0)[None, ...], 6, axis=0)

        integral: Integral = self.quadrature_est.integrate(
            functools.partial(integrand_min, pressure_coeffs=pressure_coeffs),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"s_w_min_{specifier}",
            integral.elementwise.squeeze() / self.quadrature_est.volumes,
            self.g_data,
            iterate_index=0,
        )

        def integrand_max(
            x: np.ndarray,
            pressure_coeffs: np.ndarray,
        ) -> np.ndarray:
            pressure = evaluate_poly_at_points(pressure_coeffs, x[..., 0], x[..., 1])
            saturation = self.eval_saturation(pressure)
            return np.repeat(np.max(saturation, axis=0)[None, ...], 6, axis=0)

        integral: Integral = self.quadrature_est.integrate(
            functools.partial(integrand_max, pressure_coeffs=pressure_coeffs),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"s_w_max_{specifier}",
            integral.elementwise.squeeze() / self.quadrature_est.volumes,
            self.g_data,
            iterate_index=0,
        )

    def local_flux(self, flux_name: FLUX_NAME) -> None:
        def integrand(
            x: np.ndarray,
        ) -> np.ndarray:
            fv_flux = self._evaluate_flux_at_points(
                pp.get_solution_values(
                    f"{flux_name}_RT0_coeffs", self.g_data, iterate_index=0
                ),
                x[..., 0],
                x[..., 1],
            )
            return np.sqrt((fv_flux**2).sum(axis=-1))

        integral: Integral = self.quadrature_est.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"{flux_name}_norm",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def local_darcy_est(self, flux_name: FLUX_NAME, flux_specifier: str = "") -> None:
        r"""Calculate and store the local Darcy estimator for each element at the
         current time and the inner product between current and previous time step.

        Note: The values stored are the squares of the elementwise estimators, i.e.,

        .. math::
            \|\mathbf{u}_\mathrm{t} 
                - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau}\|_K^2(t^n), \\
            \|\mathbf{u}_\mathrm{w} 
                - \kappa \left(\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}\right)\|_K^2(t^n).

        and 

        .. math::
            \left(
                (\mathbf{u}_\mathrm{t} - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau})(t^n),
                (\mathbf{u}_\mathrm{t} - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau})(t^{n-1})
            \right), \\
            \left(
                (\mathbf{u}_\mathrm{w} - \kappa (\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}))(t^n),
                (\mathbf{u}_\mathrm{w} - \kappa (\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}))(t^{n-1})
            \right).

        """
        # 1. Get a scalar permeability array.
        perm: np.ndarray | dict[str, np.ndarray] = self.permeability(self.g)
        if isinstance(perm, np.ndarray):
            perm_arr = perm
        elif len(perm) == 1:
            perm_arr = perm["kxx"]
        else:
            raise ValueError("Not implemented for tensor permeability.")

        # 2: Get reconstructed pressure coefficients and mobilities
        if flux_name == TOTAL_FLUX:
            pressure_keys = [GLOBAL_PRESSURE]
            phases = [self.wetting, self.nonwetting]
        elif flux_name == WETTING_FLUX:
            pressure_keys = (GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE)
            phases = [self.wetting]
        else:
            raise ValueError(f"Unknown flux name: {flux_name}")

        pressure_coeffs_new = {}
        pressure_coeffs_old = {}
        phase_mobilities_new = {}
        phase_mobilities_old = {}

        for pressure_key in pressure_keys:
            pressure_coeffs_new[pressure_key] = pp.get_solution_values(
                f"{pressure_key}_coeffs_rec", self.g_data, iterate_index=0
            )
            pressure_coeffs_old[pressure_key] = pp.get_solution_values(
                f"{pressure_key}_coeffs_rec", self.g_data, time_step_index=0
            )

        for phase in phases:
            # Non-upwinded mobilities.
            # TODO These are saved in reconstruction, just use those. Then HC can also
            # be changed.
            phase_mobilities_new[phase.name] = (
                self.rel_perm(  # type: ignore
                    self.wetting.s,
                    phase,
                ).value(self.equation_system)
                / phase.viscosity
            )
            phase_mobilities_old[phase.name] = (
                self.rel_perm(  # type: ignore
                    self.wetting.s.previous_timestep(),
                    phase,
                ).value(self.equation_system)
                / phase.viscosity
            )

        # 3: Define helper functions to evaluate fluxes from reconstructed pressures and
        # P0 mobilities.
        if flux_name == TOTAL_FLUX:

            def evaluate_flux_from_reconstructions(
                x: np.ndarray,
                pressure_coeffs: dict[str, np.ndarray],
                phase_mobilities: dict[str, np.ndarray],
            ) -> np.ndarray:
                """Calculate the total flux from reconstructed pressures and P0
                mobilities."""
                global_pressure_pot: np.ndarray = (
                    self._evaluate_pressure_potential_at_points(
                        pressure_coeffs[GLOBAL_PRESSURE], x[..., 0], x[..., 1]
                    )
                )
                total_mobility = (
                    phase_mobilities[self.wetting.name]
                    + phase_mobilities[self.nonwetting.name]
                )
                return (
                    -perm_arr[None, :, None]
                    * total_mobility[None, :, None]
                    * global_pressure_pot
                )

        elif flux_name == WETTING_FLUX:

            def evaluate_flux_from_reconstructions(
                x: np.ndarray,
                pressure_coeffs: dict[str, np.ndarray],
                phase_mobilities: dict[str, np.ndarray],
            ) -> np.ndarray:
                """Calculate the total flux from reconstructed pressures and P0
                mobilities."""
                global_pressure_pot = self._evaluate_pressure_potential_at_points(
                    pressure_coeffs[GLOBAL_PRESSURE], x[..., 0], x[..., 1]
                )
                complementary_pressure_pot = (
                    self._evaluate_pressure_potential_at_points(
                        pressure_coeffs[COMPLEMENTARY_PRESSURE], x[..., 0], x[..., 1]
                    )
                )
                wetting_mobility = phase_mobilities[self.wetting.name]

                return -perm_arr[None, :, None] * (
                    wetting_mobility[None, :, None] * global_pressure_pot
                    + complementary_pressure_pot
                )

        # 4: Define integrand that computes either the norm or the inner product of the
        # flux differences.
        def integrand(
            x: np.ndarray,
            specifier: Literal["_norm", "_inner_product"],
        ) -> np.ndarray:
            fv_flux_new = self._evaluate_flux_at_points(
                pp.get_solution_values(
                    f"{flux_name}{flux_specifier}_RT0_coeffs",
                    self.g_data,
                    iterate_index=0,
                ),
                x[..., 0],
                x[..., 1],
            )
            rec_flux_new = evaluate_flux_from_reconstructions(
                x, pressure_coeffs_new, phase_mobilities_new
            )

            flux_diff_new = fv_flux_new - rec_flux_new

            if specifier == "_norm":
                return (flux_diff_new**2).sum(axis=-1)

            elif specifier == "_inner_product":
                fv_flux_old = self._evaluate_flux_at_points(
                    pp.get_solution_values(
                        f"{flux_name}_RT0_coeffs", self.g_data, time_step_index=0
                    ),
                    x[..., 0],
                    x[..., 1],
                )
                rec_flux_old = evaluate_flux_from_reconstructions(
                    x, pressure_coeffs_old, phase_mobilities_old
                )
                flux_diff_old = fv_flux_old - rec_flux_old

                return (flux_diff_new * flux_diff_old).sum(axis=-1)

        # 5: Finally, integrate in space. Store norm at current time step and inner
        # product of previous and current time step.
        for specifier in ["_norm", "_inner_product"]:
            # Make mypy happy.
            specifier = typing.cast(Literal["_norm", "_inner_product"], specifier)
            integral: Integral = self.quadrature_est.integrate(
                functools.partial(integrand, specifier=specifier),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{flux_name}_D_estimator{specifier}",
                integral.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    def local_darcy_est_from_postproc(self, flux_name: FLUX_NAME) -> None:
        r"""Calculate and store the local Darcy estimator for each element at the
         current time and the inner product between current and previous time step.

        Note: The values stored are the squares of the elementwise estimators, i.e.,

        .. math::
            \|\mathbf{u}_\mathrm{t} 
                - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau}\|_K^2(t^n), \\
            \|\mathbf{u}_\mathrm{w} 
                - \kappa \left(\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}\right)\|_K^2(t^n).

        and 

        .. math::
            \left(
                (\mathbf{u}_\mathrm{t} - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau})(t^n),
                (\mathbf{u}_\mathrm{t} - \kappa \lambda_\mathrm{t} \nabla \tilde{P}_{h,\tau})(t^{n-1})
            \right), \\
            \left(
                (\mathbf{u}_\mathrm{w} - \kappa (\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}))(t^n),
                (\mathbf{u}_\mathrm{w} - \kappa (\lambda_\mathrm{w} \nabla \tilde{P}_{h,\tau}
                + \nabla \tilde{Q}_{h,\tau}))(t^{n-1})
            \right).

        """
        # 1. Get a scalar permeability array.
        perm: np.ndarray | dict[str, np.ndarray] = self.permeability(self.g)
        if isinstance(perm, np.ndarray):
            perm_arr = perm
        elif len(perm) == 1:
            perm_arr = perm["kxx"]
        else:
            raise ValueError("Not implemented for tensor permeability.")

        # 2: Get reconstructed pressure coefficients and mobilities.
        if flux_name == TOTAL_FLUX:
            pressure_keys = (GLOBAL_PRESSURE,)
            mobility_keys = ("total_mobility",)
        elif flux_name == WETTING_FLUX:
            pressure_keys = (GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE)
            mobility_keys = ("total_mobility", "fractional_flow")
        else:
            raise ValueError(f"Unknown flux name: {flux_name}")

        pressure_coeffs_new = {}
        pressure_coeffs_old = {}
        mobilities_new = {}
        mobilities_old = {}

        for pressure_key in pressure_keys:
            pressure_coeffs_new[pressure_key] = pp.get_solution_values(
                f"{pressure_key}_coeffs_postproc", self.g_data, iterate_index=0
            )
            pressure_coeffs_old[pressure_key] = pp.get_solution_values(
                f"{pressure_key}_coeffs_postproc", self.g_data, time_step_index=0
            )

        for mobility_key in mobility_keys:
            mobilities_new[mobility_key] = pp.get_solution_values(
                mobility_key, self.g_data, iterate_index=0
            )
            mobilities_old[mobility_key] = pp.get_solution_values(
                mobility_key, self.g_data, time_step_index=0
            )

        # 3: Define helper functions to evaluate fluxes from reconstructed pressures and
        # P0 mobilities.
        if flux_name == TOTAL_FLUX:

            def evaluate_flux_from_reconstructions(
                x: np.ndarray,
                pressure_coeffs: dict[str, np.ndarray],
                phase_mobilities: dict[str, np.ndarray],
            ) -> np.ndarray:
                """Calculate the total flux from reconstructed pressures and P0
                mobilities."""
                global_pressure_pot: np.ndarray = (
                    self._evaluate_pressure_potential_at_points(
                        pressure_coeffs[GLOBAL_PRESSURE], x[..., 0], x[..., 1]
                    )
                )
                total_mobility = phase_mobilities["total_mobility"]
                return (
                    -perm_arr[None, :, None]
                    * total_mobility[None, :, None]
                    * global_pressure_pot
                )

        elif flux_name == WETTING_FLUX:

            def evaluate_flux_from_reconstructions(
                x: np.ndarray,
                pressure_coeffs: dict[str, np.ndarray],
                phase_mobilities: dict[str, np.ndarray],
            ) -> np.ndarray:
                """Calculate the total flux from reconstructed pressures and P0
                mobilities."""
                global_pressure_pot = self._evaluate_pressure_potential_at_points(
                    pressure_coeffs[GLOBAL_PRESSURE], x[..., 0], x[..., 1]
                )
                complementary_pressure_pot = (
                    self._evaluate_pressure_potential_at_points(
                        pressure_coeffs[COMPLEMENTARY_PRESSURE], x[..., 0], x[..., 1]
                    )
                )
                wetting_mobility = (
                    phase_mobilities["fractional_flow"]
                    * phase_mobilities["total_mobility"]
                )

                return -perm_arr[None, :, None] * (
                    wetting_mobility[None, :, None] * global_pressure_pot
                    + complementary_pressure_pot
                )

        # 4: Define integrand that computes either the norm or the inner product of the
        # flux differences.
        def integrand(
            x: np.ndarray,
            specifier: Literal["_norm", "_inner_product"],
        ) -> np.ndarray:
            fv_flux_new = self._evaluate_flux_at_points(
                pp.get_solution_values(
                    f"{flux_name}_RT0_coeffs", self.g_data, iterate_index=0
                ),
                x[..., 0],
                x[..., 1],
            )
            rec_flux_new = evaluate_flux_from_reconstructions(
                x, pressure_coeffs_new, mobilities_new
            )

            # FIXME Not zero for wetting flux.
            flux_diff_new = fv_flux_new - rec_flux_new

            if specifier == "_norm":
                return (flux_diff_new**2).sum(axis=-1)

            elif specifier == "_inner_product":
                fv_flux_old = self._evaluate_flux_at_points(
                    pp.get_solution_values(
                        f"{flux_name}_RT0_coeffs", self.g_data, time_step_index=0
                    ),
                    x[..., 0],
                    x[..., 1],
                )
                rec_flux_old = evaluate_flux_from_reconstructions(
                    x, pressure_coeffs_old, mobilities_old
                )
                flux_diff_old = fv_flux_old - rec_flux_old

                return (flux_diff_new * flux_diff_old).sum(axis=-1)

        # 5: Finally, integrate in space. Store norm at current time step and inner
        # product of previous and current time step.
        for specifier in ["_norm", "_inner_product"]:
            # Make mypy happy.
            specifier = typing.cast(Literal["_norm", "_inner_product"], specifier)
            integral: Integral = self.quadrature_est.integrate(
                functools.partial(integrand, specifier=specifier),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{flux_name}_D_estimator_from_postproc{specifier}",
                integral.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    @staticmethod
    def _evaluate_pressure_potential_at_points(
        coeffs: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Helper function to evaluate the potential of a P2 pressure defined by the
        given coefficients at the specified points.

        """
        pressure_potential_x: np.ndarray = (
            2 * x * (coeffs[..., 0])[None, ...]
            + y * (coeffs[..., 1])[None, ...]
            + (coeffs[..., 2])[None, ...]
        )
        pressure_potential_y: np.ndarray = (
            2 * y * (coeffs[..., 3])[None, ...]
            + x * (coeffs[..., 1])[None, ...]
            + (coeffs[..., 4])[None, ...]
        )
        return np.stack([pressure_potential_x, pressure_potential_y], axis=-1)

    @staticmethod
    def _evaluate_flux_at_points(
        coeffs: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        r"""Helper function to evaluate the RT0 flux defined by the given coefficients
         at the specified points.

        Args:
            coeffs: Coefficients defining the flux.
            x: x-coordinates of evaluation points.
            y: y-coordinates of evaluation points.

        Returns:
            flux: Evaluated flux at the specified points.

        """
        flux_x: np.ndarray = coeffs[..., 0] * x + coeffs[..., 1]
        flux_y: np.ndarray = coeffs[..., 0] * y + coeffs[..., 2]
        return np.stack([flux_x, flux_y], axis=-1)

    def local_sp_est(self) -> None:
        coeffs: np.ndarray = pp.get_solution_values(
            COMPLEMENTARY_PRESSURE + "_coeffs_rec", self.g_data, iterate_index=0
        )
        s_p0: np.ndarray = self.wetting.s.value(self.equation_system)  # type: ignore

        def integrand(
            x: np.ndarray,
        ) -> np.ndarray:
            q_p2 = evaluate_poly_at_points(coeffs, x[..., 0], x[..., 1])
            s_p2 = self.eval_saturation(q_p2)

            # Time integral by diving by time step size. In :meth:`global_sp_est`, the
            # previous time step value is added.
            # ``s_p2`` has shape (num_quad_points_per_element, num_cells), while
            # ``s_p0`` has shape (num_cells,). Since the latter is cellwise
            # constant, we can broadcast.
            return (s_p0[None, ...] - s_p2) ** 2 / self.time_manager.dt

        integral: Integral = self.quadrature_est.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            "SP_estimator",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def global_res_and_flux_est(self) -> float:
        r"""Compute the global residual and flux error estimator by summing local
         contributions and integrating in time for both total and wetting fluxes.


        Mathematically, this corresponds to:

        .. math::
            \left\{
                \sum_{\alpha \in \{w, t\}}
                \int_{I^n}
                \sum_{K \in \mathcal{T}_h}
                    \left( \eta_{R,\alpha,K} + \eta_{F,\alpha,K}(t) \right)^2
                \, dt
            \right\}^{1/2}

        The time integrals are approximated by the trapezoidal rule.

        Note:
            The residual estimator math:`\eta_{R,\alpha,K}` is time-independent. To
            approximate its time integral, we reuse the same value at both :math:`t^n`
            and :math:`t^{n-1}`.


        Returns:
            estimator: The combined global residual and flux error estimator

        """
        estimators: list[float] = []
        for flux_name in (TOTAL_FLUX, WETTING_FLUX):
            self.local_residual_est(flux_name)
            self.local_flux_est(flux_name)

            # Load local spatial integrals from current and previous time step. NOTE The
            # stored values are squared, hence we first take the square root.
            residual = (
                pp.get_solution_values(
                    f"{flux_name}_R_estimator", self.g_data, iterate_index=0
                )
                ** 0.5
            )
            flux_new = (
                pp.get_solution_values(
                    f"{flux_name}_F_estimator_new", self.g_data, iterate_index=0
                )
                ** 0.5
            )
            flux_old = (
                pp.get_solution_values(
                    f"{flux_name}_F_estimator_old", self.g_data, iterate_index=0
                )
                ** 0.5
            )

            # Calculate global integrals at current and previous time step.
            global_integral_new: float = ((residual + flux_new) ** 2).sum()
            global_integral_old: float = ((residual + flux_old) ** 2).sum()

            # Integrate in time by trapezoidal rule.
            estimators.append(
                self.time_manager.dt / 2 * (global_integral_new + global_integral_old)
            )

        est = sum(estimators) ** 0.5
        logger.info(f"Global residual and flux error estimator: {est}")
        return est

    def global_darcy_est(self) -> tuple[float, float]:
        r"""Compute the global Darcy error estimator by summing local contributions and
        integrating in time.

        Mathematically, we approximate:

        .. math::
            \left\{
                \int_{I^n}
                    \sum_{K \in \mathcal{T}_h} \left( \eta_{\bullet,K}(t) \right)^2
                \, \mathrm{d}t
            \right\}^{1/2}

        We use the time integration scheme proposed by Vohralík and Wheeler
        (2013, doi:10.1007/s10596-013-9356-0), which is exact for estimators
        :math:`\eta_{\bullet,K}` that vary linearly in time. The quadrature rule is:

        .. math::
            \frac{\tau^n}{3} \sum_{K \in \mathcal{T}_h} \left[
                \left( \eta_{\bullet,K}(t^n) \right)^2
                + \left( q(t^n), q(t^{n-1}) \right)_K
                + \left( \eta_{\bullet,K}(t^{n-1}) \right)^2
            \right]

        where :math:`q` denotes the spatial integrand used to compute
        :math:`\eta_{\bullet,K}`.

        Returns:
            estimator: The global Darcy error estimator for each flux type.

        """
        estimators: dict[str, float] = {}
        for flux_name in (TOTAL_FLUX, WETTING_FLUX):
            self.local_darcy_est(flux_name)

            norm_key = f"{flux_name}_D_estimator_norm"
            inner_key = f"{flux_name}_D_estimator_inner_product"

            # Load local spatial integrals from current and previous time step and
            # combined inner-product.
            new = pp.get_solution_values(norm_key, self.g_data, iterate_index=0)
            inner = pp.get_solution_values(inner_key, self.g_data, iterate_index=0)
            old = pp.get_solution_values(norm_key, self.g_data, time_step_index=0)

            # Estimate time integral with quadrature rule for linear functions.
            # NOTE The stored local values are already squared.
            estimators[flux_name] = (
                self.time_manager.dt / 3 * (new + inner + old).sum()
            ) ** 0.5

            if estimators[flux_name] < 0:
                raise RuntimeError(
                    "Temporal integral of Darcy error estimate is negative."
                )

            logger.info(
                f"Global {flux_name} Darcy error estimator:"
                + f" {estimators[flux_name]}"
            )

        return estimators[TOTAL_FLUX], estimators[WETTING_FLUX]

    def global_sp_est(self) -> float:
        r"""Compute the global saturation-pressure estimator by summing local
         contributions and integrating in time.

        The temporal derivative of an (assumed) linear function is integrated.

        Returns:
            estimator: The global saturation-pressure error estimator.

        """
        # Calculate local spatial integrals at current time step.
        self.local_sp_est()

        new = pp.get_solution_values("SP_estimator", self.g_data, iterate_index=0)
        old = pp.get_solution_values("SP_estimator", self.g_data, time_step_index=0)

        # Integrate in time by multiplying constant value with time step size. NOTE The
        # stored values are already squared.
        est = (self.time_manager.dt * (new + old).sum()) ** 0.5

        if est < 0:
            raise RuntimeError("Temporal integral of SP error estimate is negative.")

        logger.info(f"Global saturation-pressure error estimator: {est}")

        return est

    def global_darcy_and_sp_est(self) -> float:
        r"""Compute the combined global Darcy and saturation-pressure error estimator."""
        est_total, est_wetting = self.global_darcy_est()
        est_saturation_pressure = self.global_sp_est()
        combined_est = (
            est_total**2 + est_wetting**2 + est_saturation_pressure**2
        ) ** 0.5
        logger.info(
            "Combined global Darcy and saturation-pressure error estimator:"
            + f" {combined_est}"
        )
        return combined_est

    def local_energy_norm(self) -> None:
        r"""Calculate the local in space and time energy norm of the numerical
        solution.

        Calculate the local in time and space terms that are summed and integrated in
        :meth:`global_energy_norm`.

        """
        for flux_name in (TOTAL_FLUX, WETTING_FLUX):
            fv_flux_coeffs: np.ndarray = pp.get_solution_values(
                f"{flux_name}_RT0_coeffs", self.g_data, iterate_index=0
            )

            def integrand(x: np.ndarray) -> np.ndarray:
                r"""
                Returns:
                    integrand: :math:`|\mathbf{u}_{\alpha,h,\tau}|_K^2 .

                """
                fv_flux = self._evaluate_flux_at_points(
                    fv_flux_coeffs, x[..., 0], x[..., 1]
                )
                return (fv_flux**2).sum(axis=-1)

            local_energy: Integral = self.quadrature_est.integrate(
                integrand,
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{flux_name}_energy_norm",
                local_energy.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    def global_energy_norm(self) -> float:
        r"""Calculate the global in space and local in time energy norm of the numerical
        solution.

        The energy norm is defined as the sum of the size of the total and the wetting
        flux.

        .. math::
            \mathcal{E}_{I^n,\Omega}(p_{n,h,\tau}, s_{w,h,\tau})
                := \int_{I^n} \int_\Omega |\bm{u}_{\alpha,h,\tau}|^2 \, dx \, dt

        """
        self.local_energy_norm()

        global_energies: list[float] = []
        for flux_name in (TOTAL_FLUX, WETTING_FLUX):
            local_energy_new: np.ndarray = pp.get_solution_values(
                f"{flux_name}_energy_norm", self.g_data, iterate_index=0
            )
            local_energy_old: np.ndarray = pp.get_solution_values(
                f"{flux_name}_energy_norm", self.g_data, time_step_index=0
            )
            global_energies.append(
                (self.time_manager.dt / 2 * (local_energy_new + local_energy_old).sum())
                ** (1 / 2)
            )

        global_energy: float = sum(global_energies)
        logger.info(f"Global energy norm: {global_energy}")
        return global_energy


# This could also be a mixin, but by subclassing ``SolutionStrategyReconstruction``, we
# avoid having to pay attention to the order of the different solution strategy classes.
#
# Protocols define different types for ``nonlinear_solver_statistics``, causing mypy
# errors. This is safe in practice, but ``nonlinear_solver_statistics`` must be used
# with care. We ignore the error.
class SolutionStrategyEst(  # type: ignore
    EstimatesProtocol,
    SolutionStrategyRec,
):
    def prepare_simulation(self) -> None:
        """Set up estimators after setting up the base simulation and
        reconstructions.

        """
        super().prepare_simulation()

        # Setup estimators.
        self.setup_estimates()
        self.set_initial_estimators()

    def set_initial_estimators(self) -> None:
        """Initialize iterate and time step values for error estimators.

        Approximating time integrals requires previous time step values. At the initial
        time step, the analytical solution is equal to the initial values (assuming
        elementwise constant initial values), hence all local estimators are zero.

        Note: This is basically doing the same as :meth:`postprocess_solution` and
        :meth:`after_hc_convergence`, but not everything.

        """
        initial_values: np.ndarray = np.zeros(self.g.num_cells)
        for flux_name, estimator_name in itertools.product(
            (TOTAL_FLUX, WETTING_FLUX),
            [
                "R_estimator",
                "F_estimator_new",
                "F_estimator_old",
                "D_estimator_norm",
                "energy_norm",
            ],
        ):
            key = f"{flux_name}_{estimator_name}"
            pp.set_solution_values(
                key, initial_values, self.g_data, time_step_index=0, iterate_index=0
            )

        pp.set_solution_values(
            "SP_estimator",
            initial_values,
            self.g_data,
            time_step_index=0,
            iterate_index=0,
        )

        initial_values = np.ones(self.g.num_cells)
        for mobility_key in ["total_mobility"]:
            pp.set_solution_values(
                mobility_key,
                initial_values,
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
        converged, diverged = super().check_convergence(
            nonlinear_increment, residual, reference_residual, nl_params
        )
        residual_and_flux_est = self.global_res_and_flux_est()
        total_darcy_est, wetting_darcy_est = self.global_darcy_est()
        sp_est = self.global_sp_est()
        global_energy_norm: float = self.global_energy_norm()

        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm=None,
            residual_norm=None,
            residual_and_flux_est=residual_and_flux_est,
            darcy_est={
                TOTAL_FLUX: total_darcy_est,
                WETTING_FLUX: wetting_darcy_est,
            },
            sp_est=sp_est,
            global_energy_norm=global_energy_norm,
        )
        return converged, diverged

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)
        # FIXME Remove all these methods again and any reference to stored values.
        # self.local_pressure_potential(GLOBAL_PRESSURE)
        # self.local_pressure_potential(COMPLEMENTARY_PRESSURE)
        # self.local_flux(TOTAL_FLUX)
        # self.local_flux(WETTING_FLUX)
        # self.local_saturation_min_max("rec")
        # self.local_saturation_min_max("postproc")
        # self.local_darcy_est_from_postproc(TOTAL_FLUX)
        # self.local_darcy_est_from_postproc(WETTING_FLUX)

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Update time step values for local in space and time estimators.

        # NOTE The local residual and flux error estimators are only needed at the
        # current iteration. We set the time step values for completeness and to avoid
        # extra code in :meth:`_data_to_export`.
        # NOTE The RT0 coeffs are needed to calculate the local flux estimators.
        for flux_name, specifier in itertools.product(
            (TOTAL_FLUX, WETTING_FLUX),
            [
                "R_estimator",
                "F_estimator_new",
                "F_estimator_old",
                "D_estimator_norm",
                # "D_estimator_from_postproc_norm",
                "RT0_coeffs",
                "energy_norm",
            ],
        ):
            key = f"{flux_name}_{specifier}"
            time_step_values: np.ndarray = pp.get_solution_values(
                key, self.g_data, iterate_index=0
            )
            pp.set_solution_values(
                key, time_step_values, self.g_data, time_step_index=0
            )

        time_step_values = pp.get_solution_values(
            "SP_estimator", self.g_data, iterate_index=0
        )
        pp.set_solution_values(
            "SP_estimator", time_step_values, self.g_data, time_step_index=0
        )

        # for pressure_key, specifier in itertools.product(
        #     (GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE), ("_postproc", "_rec")
        # ):
        #     time_step_values = pp.get_solution_values(
        #         f"{pressure_key}{specifier}_potential", self.g_data, iterate_index=0
        #     )
        #     pp.set_solution_values(
        #         f"{pressure_key}{specifier}_potential",
        #         time_step_values,
        #         self.g_data,
        #         time_step_index=0,
        #     )
        # for flux_name in (TOTAL_FLUX, WETTING_FLUX):
        #     time_step_values = pp.get_solution_values(
        #         f"{flux_name}_norm", self.g_data, iterate_index=0
        #     )
        #     pp.set_solution_values(
        #         f"{flux_name}_norm",
        #         time_step_values,
        #         self.g_data,
        #         time_step_index=0,
        #     )


class DataSavingEst(DataSavingRec):
    def _data_to_export(
        self, time_step_index: int | None = None, iterate_index: int | None = None
    ) -> list[DataInput]:
        """Append error estimators to the exported data."""
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index,
            iterate_index=iterate_index,
        )
        for flux_name, estimator_name in itertools.product(
            (TOTAL_FLUX, WETTING_FLUX),
            [
                "R_estimator",
                "F_estimator_new",
                "F_estimator_old",
                "D_estimator_norm",
                "D_estimator_from_postproc_norm",
                "energy_norm",
            ],
        ):
            # Before simulation, the estimators won't be set yet, due to the order of
            # calls in :meth:`prepare_simulation`. However, after the first time step,
            # :attr:`time_manager.time_step_index` won't be updated yet. Checking for
            # all of this is quite convoluted. Instead we just use try-except.

            try:
                key = f"{flux_name}_{estimator_name}"
                data.append(
                    (
                        self.g,
                        key,
                        pp.get_solution_values(
                            key,
                            self.g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            except KeyError:
                pass

            try:
                data.append(
                    (
                        self.g,
                        "SP_estimator",
                        pp.get_solution_values(
                            "SP_estimator",
                            self.g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            except KeyError:
                pass
        for pressure_key, specifier in itertools.product(
            (GLOBAL_PRESSURE, COMPLEMENTARY_PRESSURE), ("_postproc", "_rec")
        ):
            try:
                data.append(
                    (
                        self.g,
                        f"{pressure_key}{specifier}_potential",
                        pp.get_solution_values(
                            f"{pressure_key}{specifier}_potential",
                            self.g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            except KeyError:
                pass
        for flux_name in (TOTAL_FLUX, WETTING_FLUX):
            try:
                data.append(
                    (
                        self.g,
                        f"{flux_name}_norm",
                        pp.get_solution_values(
                            f"{flux_name}_norm",
                            self.g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            except KeyError:
                pass
        for specifier in ["max_postproc", "min_postproc", "max_rec", "min_rec"]:
            try:
                data.append(
                    (
                        self.g,
                        f"s_w_{specifier}",
                        pp.get_solution_values(
                            f"s_w_{specifier}", self.g_data, iterate_index=0
                        ),
                    )
                )
            except KeyError:
                pass

        return data


# Protocols define different types for ``nonlinear_solver_statistics``, causing mypy
# errors. This is safe in practice, but ``nonlinear_solver_statistics`` must be used
# with care. We ignore the error.
class TwoPhaseFlowErrorEstimate(  # type: ignore
    ErrorEstimateMixin,
    SolutionStrategyEst,
    DataSavingEst,
    TwoPhaseFlowReconstruction,
): ...  # type: ignore
