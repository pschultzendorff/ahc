import functools
import itertools
import logging
import typing
from typing import Any, Literal, Optional, cast

import numpy as np
import porepy as pp
from porepy.viz.exporter import DataInput
from tpf.models.protocol import EstimatesProtocol, ReconstructionProtocol, TPFProtocol
from tpf.models.reconstruction import (
    DataSavingReconstruction,
    SolutionStrategyReconstruction,
    TwoPhaseFlowReconstruction,
)
from tpf.numerics.quadrature import Integral, TriangleQuadrature
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    PHASENAME,
    PRESSURE_KEY,
)

logger = logging.getLogger(__name__)


class ErrorEstimateMixin(ReconstructionProtocol, TPFProtocol):

    def setup_estimates(self) -> None:
        self.quadrature_estimate_degree: int = 4
        """Degree of quadrature rule for the error estimators."""
        self.quadrature_estimate = TriangleQuadrature(self.quadrature_estimate_degree)
        """Quadrature rule for the error estimators.

        Note: For efficiency, :attr:`self.quadrature_reconstructions` could be used.
            However, we may want to have the flexibility to use different degrees for
            estimates and reconstruction.

        """

    def poincare_constant(self, g: pp.Grid) -> np.ndarray:
        return g.cell_diameters() / np.pi

    def local_residual_est(
        self, flux_name: Literal["total", "wetting_from_ff"]
    ) -> None:
        r"""Calculate and store the local residual estimate for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,
        .. math::
            \|q_t - \nabla \cdot \theta_t\|_K^2, \\
            \|\varphi \partial_t s_w - q_w+ \nabla \cdot \theta_w\|_K^2.

        """
        g, g_data = self.mdg.subdomains(return_data=True)[0]

        # Collect terms for the estimators as ``np.ndarray``.
        poincare_constant: np.ndarray = self.poincare_constant(g)

        # To get the average source term for each element, we divide by element volumes.
        # Integrating gives the element source again.
        source: np.ndarray = (
            self.phase_fluid_source(g, self.wetting)
            if flux_name == "wetting_from_ff"
            else self.total_fluid_source(g)
        ) / g.cell_volumes

        equilibrated_flux_coeffs: np.ndarray = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs",
            g_data,
            iterate_index=0,
        )
        # Divergence of a Raviart-Thomas basis function is given by twice the linear
        # coefficient:
        # :math:`\nabla \cdot \begin{pmatrix} ax + b \\ ay + c\end{pmatrix} = 2 a`
        div_equilibrated_flux: np.ndarray = 2 * equilibrated_flux_coeffs[:, 0]

        # Integrate elementwise, shift values, and store the result.
        # NOTE The saturation term, source term, and divergence of the flux
        # reconstruction, are all elementwise constant. Therefore, we can integrate
        # explicitely by multiplying with the element volume.
        if flux_name == "wetting_from_ff":
            dt = pp.ad.Scalar(self.time_manager.dt)
            dt_s_ad = pp.ad.time_derivatives.dt(self.wetting.s, dt)
            dt_s: np.ndarray = (dt_s_ad).value(self.equation_system)
            porosity: np.ndarray = self.porosity(g)
            integral: Integral = Integral(
                g.cell_volumes
                * poincare_constant
                * (porosity * dt_s + div_equilibrated_flux - source) ** 2
            )
        elif flux_name == "total":
            integral = Integral(
                g.cell_volumes
                * poincare_constant
                * (div_equilibrated_flux - source) ** 2
            )

        pp.shift_solution_values(
            f"{flux_name}_R_estimate",
            g_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_R_estimate",
            integral.elementwise,
            g_data,
            iterate_index=0,
        )

    def local_flux_est(self, flux_name: Literal["total", "wetting_from_ff"]) -> None:
        r"""Calculate and store the local flux estimate for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,

        .. math::
            \|\mathbf{u}_\alpha - \theta_\alpha\|_K^2,

        where :math:`\mathbf{u}_\alpha` is the FV flux at the current time step,
        continuation iteration, and Newton iteration, i.e., in
        :math:`\textbf{RTN}_0(\mathca{T}_h)` but not locally mass conservative, and
        :math:`\theta_\alpha` is the reconstructed flux, i.e., both in
        :math:`\textbf{RTN}_0(\mathca{T}_h)` and locally mass conservative.

        """
        _, g_data = self.mdg.subdomains(return_data=True)[0]
        # First, we calculate the spatial integral of the difference between FV and
        # reconstructed flux elementwise at the current time step.

        # Retrieve FV and equilibrated flux coefficients.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", g_data, iterate_index=0
        )
        equilibrated_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs", g_data, iterate_index=0
        )

        def integrand(x: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs[..., 0] - equilibrated_coeffs[..., 0]
            ) * x[..., 0] + (fv_coeffs[..., 1] - equilibrated_coeffs[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs[..., 0] - equilibrated_coeffs[..., 0]
            ) * x[..., 1] + (fv_coeffs[..., 2] - equilibrated_coeffs[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise, shift values, and store the result.
        integral: Integral = self.quadrature_estimate.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{flux_name}_F_estimate",
            g_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_F_estimate",
            integral.elementwise,
            g_data,
            iterate_index=0,
        )

    def local_nonconformity_est(
        self,
        pressure_key: PRESSURE_KEY,
    ) -> None:
        r"""Calculate and store the local nonconformity estimate for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,

        .. math::
            \|\mathbf{u}_\alpha - \theta_\alpha\|_K^2,

        """
        sd, g_data = self.mdg.subdomains(return_data=True)[0]

        # First, calculate the three different spatial integrals (norm at current time
        # step, inner product current-previous time step, norm at previous time step) of
        # the difference between post-processed and reconstructed pressure potential.
        # The latter was stored at the previous time step and is not recalculated.
        perm: np.ndarray | dict[str, np.ndarray] = self.permeability(sd)
        # Calculate a not upwinded total mobility.
        if pressure_key == GLOBAL_PRESSURE:
            rel_perms: dict[str, np.ndarray] = {}
            for phase in self.phases.values():
                rel_perms[phase.name] = self.rel_perm(self.wetting.s, phase).value(
                    self.equation_system
                )
            total_mobility: np.ndarray = (
                rel_perms[self.wetting.name] / self.wetting.viscosity
                + rel_perms[self.nonwetting.name] / self.nonwetting.viscosity
            )

        # Get postprocessed pressure coefficients and reconstructed pressure points
        # values/coefficients from the current and previous time step.
        # NOTE If ``self._permeability`` is a scalar, ``postprocessed_coeffs[...,1]``
        # and ``reconstructed_coeffs[...,1]`` are zero.
        postprocessed_coeffs: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_postprocessed_coeffs", g_data, iterate_index=0
        )
        reconstructed_coeffs: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_reconstructed_coeffs", g_data, iterate_index=0
        )

        postprocessed_coeffs_old: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_postprocessed_coeffs", g_data, time_step_index=0
        )
        reconstructed_coeffs_old: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_reconstructed_coeffs", g_data, time_step_index=0
        )

        def integrand(
            x: np.ndarray,
            values: Literal["L2_new", "inner_product_new_old"],
        ) -> np.ndarray:
            r"""

            The integrand can take two different forms, exemplarily shown for
            complimentary pressure:
            .. math::
                (\kappa \nabla (\tilde{\mathfrac{q}^n} - \mathfrac{q}^n))^2, \\
                \kappa \nabla (\tilde{\mathfrac{q}}^n - \mathfrac{q}^n) \cdot
                \kappa \nabla (\tilde{\mathfrac{q}}^{n-1} - \mathfrac{q}^{n-1}).

            """
            nonlocal reconstructed_coeffs, reconstructed_coeffs_old, postprocessed_coeffs_old, total_mobility, perm
            # Calculate the pressure potential at the current time step and at the
            # previous time step (if needed).
            fluxes: list[np.ndarray] = []
            for i, (pp_coeffs, rc_coeffs) in enumerate(
                [
                    (postprocessed_coeffs, reconstructed_coeffs),
                    (postprocessed_coeffs_old, reconstructed_coeffs_old),
                ]
            ):
                # We do not need the previous time step values to evaluate the norm.
                if values == "L2_new" and i == 1:
                    break
                # Calculate the potential directly from the coefficients.
                pressure_potential_x: np.ndarray = (
                    2 * x[..., 0] * (pp_coeffs[..., 0] - rc_coeffs[..., 0])[None, ...]
                    + x[..., 1] * (pp_coeffs[..., 1] - rc_coeffs[..., 1])[None, ...]
                    + (pp_coeffs[..., 2] - rc_coeffs[..., 2])[None, ...]
                )
                pressure_potential_y: np.ndarray = (
                    2 * x[..., 1] * (pp_coeffs[..., 3] - rc_coeffs[..., 3])[None, ...]
                    + x[..., 0] * (pp_coeffs[..., 1] - rc_coeffs[..., 1])[None, ...]
                    + (pp_coeffs[..., 4] - rc_coeffs[..., 4])[None, ...]
                )
                pressure_potential: np.ndarray = np.stack(
                    [pressure_potential_x, pressure_potential_y], axis=-1
                )
                # NOTE For now, perm is just a scalar, so we do not have to use
                # matrix multiplication.
                # Different treatment for scalar and tensor permeabilities.
                if isinstance(perm, np.ndarray):
                    fluxes.append(
                        perm[None, :, None]
                        * total_mobility[None, :, None]
                        * pressure_potential
                        if pressure_key == GLOBAL_PRESSURE
                        else perm[None, :, None] * pressure_potential
                    )
                elif len(perm) == 1:
                    fluxes.append(
                        perm["kxx"][None, :, None]
                        * total_mobility[None, :, None]
                        * pressure_potential
                        if pressure_key == GLOBAL_PRESSURE
                        else perm["kxx"][None, :, None] * pressure_potential
                    )
                elif len(perm) == 2:
                    # TODO Fix this for tensor permeabilities.
                    # Perm has form ``{"kxx": np.ndarray, "kyy": np.ndarray, ...}``.
                    # pressure_potential_times_mobility: np.ndarray = (
                    #     total_mobility[None, :, None] * pressure_potential
                    #     if pressure_key == GLOBAL_PRESSURE
                    #     else pressure_potential
                    # )
                    # fluxes.append()
                    ...
                else:
                    raise ValueError(
                        "Permeability must be scalar or tensor with zero"
                        + " values off the diagonal."
                    )
            if values == "L2_new":
                fluxes.append(fluxes[0])
            return (
                fluxes[0][..., 0] * fluxes[1][..., 0]
                + fluxes[0][..., 1] * fluxes[1][..., 1]
            )

        # Integrate :math:`(\kappa \nabla (\tilde{\mathfrac{q}^n} -
        # \mathfrac{q}^n))^2` elementwise, shift values, and store the result.
        integral_L2_new: Integral = self.quadrature_estimate.integrate(
            functools.partial(integrand, values="L2_new"),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{pressure_key}_NC_estimate",
            g_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{pressure_key}_NC_estimate",
            integral_L2_new.elementwise,
            g_data,
            iterate_index=0,
        )
        # Integrate :math:`\kappa \nabla (\tilde{\mathfrac{q}}^n - \mathfrac{q}^n)
        # \cdot \kappa \nabla (\tilde{\mathfrac{q}}^{n-1} - \mathfrac{q}^{n-1})`
        # elementwise.
        integral_inner_product_new_old: Integral = self.quadrature_estimate.integrate(
            functools.partial(integrand, values="inner_product_new_old"),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"{pressure_key}_NC_estimate_inner_product_new_old",
            integral_inner_product_new_old.elementwise,
            g_data,
            iterate_index=0,
        )

    def global_res_and_flux_est(self) -> float:
        r"""Sum local flux and residual estimators, integrate in time, and sum total and
         wetting estimators.

        .. math::
            \left\{
                \sum_{\alpha \in \{w,t\}}
                    \sum_{n = 1}^N
                        \int_{I_n}
                            \sum_{K \in \mathcal{T}_h}
                                (\eta_{R,\alpha,K} + \eta_{F,\alpha,K}(t))^2
                        dt
            \right\}^{\frac{1}{2}}

        The time integrals are approximated by the trapezoidal rule.

        """
        _, g_data = self.mdg.subdomains(return_data=True)[0]

        estimators: dict[str, float] = {}
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

            # Calculate local estimates.
            self.local_residual_est(flux_name)
            self.local_flux_est(flux_name)
            # Load spatial integrals from current time step.
            integral_R_new: np.ndarray = pp.get_solution_values(
                f"{flux_name}_R_estimate", g_data, iterate_index=0
            )
            integral_F_new: np.ndarray = pp.get_solution_values(
                f"{flux_name}_F_estimate", g_data, iterate_index=0
            )
            # Load spatial integrals from previous time step.
            integral_R_old: np.ndarray = pp.get_solution_values(
                f"{flux_name}_R_estimate", g_data, time_step_index=0
            )
            integral_F_old: np.ndarray = pp.get_solution_values(
                f"{flux_name}_F_estimate", g_data, time_step_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # take the square root first
            global_integral_new: float = (
                (integral_R_new ** (1 / 2) + integral_F_new ** (1 / 2)) ** 2
            ).sum()
            global_integral_old: float = (
                (integral_R_old ** (1 / 2) + integral_F_old ** (1 / 2)) ** 2
            ).sum()
            # Integrate in time by trapezoidal rule.
            estimators[flux_name] = (
                self.time_manager.dt / 2 * (global_integral_new + global_integral_old)
            ) ** (1 / 2)
        logger.info(
            f"Global residual and flux error estimate: {sum(estimators.values())}"
        )
        return sum(estimators.values())

    def global_nonconformity_est(self) -> tuple[float, float]:
        r"""Sum local nonconformity estimators and integrate in time.

        .. math::
            \left\{
                \sum_{n = 1}^N
                    \int_{I_n}
                        \sum_{K \in \mathcal{T}_h} (\eta_{NC,1,K}(t))^2
                    dt
            \right\}^{\frac{1}{2}}

        .. math::
            \left\{
                \sum_{n = 1}^N
                    \int_{I_n}
                        \sum_{K \in \mathcal{T}_h} (\eta_{NC,2,K}(t))^2
                    dt
            \right\}^{\frac{1}{2}}

        We follow the construction of Vohralík and M. F. Wheeler, “A posteriori error
        estimates, stopping criteria, and adaptivity for two-phase flows,” 2013,
        doi:10.1007/s10596-013-9356-0. The time integrals are approximated by
        (exemplarily for complimentary pressure):
        .. math::
            \frac{|t_n - t_{n-1}|}{3} \sum_{K \in \mathcal{T}_h}
            \left[
                (\eta_{NC,2,K}(t_n))^2
                + \left(\kappa \nabla (Q_h^{n,i,k} -\hat{Q}_h^{n,i,k}),
                    \kappa \nabla (Q_h^{n-1} -\hat{Q}_h^{n-1})\right)_K
                + (\eta_{NC,2,K}(t_{n - 1}))^2
            \right]

        """
        _, g_data = self.mdg.subdomains(return_data=True)[0]

        estimators: dict[str, float] = {}
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            # Satisfy mypy.
            pressure_key = typing.cast(PRESSURE_KEY, pressure_key)
            # Calculate local estimates.
            self.local_nonconformity_est(pressure_key)
            # Load spatial integrals from current time step.
            integral_NC_new: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_NC_estimate", g_data, iterate_index=0
            )
            integral_NC_inner_product_new_old: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_NC_estimate_inner_product_new_old",
                g_data,
                iterate_index=0,
            )
            # Load spatial integrals from previous time step.
            integral_NC_old: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_NC_estimate", g_data, time_step_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # do not need to square here.
            global_integral_new: float = (integral_NC_new).sum()
            global_integral_inner_product_new_old: float = (
                integral_NC_inner_product_new_old**2
            ).sum()
            global_integral_old: float = (integral_NC_old).sum()
            # Finally, estimate the time integral.
            estimators[pressure_key] = (
                self.time_manager.dt
                / 3
                * (
                    global_integral_new
                    + global_integral_inner_product_new_old
                    + global_integral_old
                )
            ) ** (1 / 2)
            logger.info(
                f"Global {pressure_key} nonconformity error estimate: {estimators[pressure_key]}"
            )

        return estimators[GLOBAL_PRESSURE], estimators[COMPLIMENTARY_PRESSURE]

    def local_energy_norm(self) -> None:
        r"""Calculate the local in space and time energy norm of the numerical
        solution.

        Calculate the local in time and space terms that are summed and integrated in
        :meth:`global_energy_norm`.

        """
        g, g_data = self.mdg.subdomains(return_data=True)[0]

        # Saturation term:
        # FIXME This has to be evaluated by solving a local FEM problem.
        # dt_saturation: np.ndarray = pp.get_solution_values(
        #     "s_w", g_data, iterate_index=0
        # ) - pp.get_solution_values("s_w", g_data, time_step_index=0)
        saturation_term: Integral = Integral(np.zeros(g.num_cells))

        # Global pressure term:
        # Note that the postprocessing of the pressure is done
        # s.t. :math:`\kappa \lambda_t \nabla P = \bm{F}_t` at the edges. Instead of
        # integrating the pressure potential times total (upwinded) mobility times
        # permeability, we integrate the flux directly.
        # total_flux_coeffs: np.ndarray = pp.get_solution_values(
        #     f"total_flux_RT0_coeffs", g_data, iterate_index=0
        # )
        # global_pressure_term: Integral = Integral(
        #     2 * total_flux_coeffs[..., 0] * g.cell_volumes
        # )
        global_pressure_coeffs: np.ndarray = pp.get_solution_values(
            f"{GLOBAL_PRESSURE}_postprocessed_coeffs", g_data, iterate_index=0
        )
        perm: np.ndarray | dict[str, np.ndarray] = self.permeability(g)
        rel_perms: dict[str, np.ndarray] = {}
        for phase in self.phases.values():
            rel_perms[phase.name] = self.rel_perm(self.wetting.s, phase).value(
                self.equation_system
            )
        total_mobility: np.ndarray = (
            rel_perms[self.wetting.name] / self.wetting.viscosity
            + rel_perms[self.nonwetting.name] / self.nonwetting.viscosity
        )

        def integrand_1(
            x: np.ndarray,
        ) -> np.ndarray:
            r"""
            Returns:
                integrand: :math:`|\kappa \nabla P(x)|^2.

            """
            nonlocal global_pressure_coeffs, perm
            # Calculate the potential directly from the coefficients.
            pressure_potential_x: np.ndarray = (
                2 * x[..., 0] * global_pressure_coeffs[..., 0][None, ...]
                + x[..., 1] * global_pressure_coeffs[..., 1][None, ...]
                + global_pressure_coeffs[..., 2][None, ...]
            )
            pressure_potential_y: np.ndarray = (
                2 * x[..., 1] * global_pressure_coeffs[..., 3][None, ...]
                + x[..., 0] * global_pressure_coeffs[..., 1][None, ...]
                + global_pressure_coeffs[..., 4][None, ...]
            )
            pressure_potential: np.ndarray = np.stack(
                [pressure_potential_x, pressure_potential_y], axis=-1
            )
            # Different treatment for scalar and tensor permeabilities.
            if isinstance(perm, np.ndarray):
                pressure_potential *= perm[None, :, None]  # ** (1 / 2)
            elif len(perm) == 1:
                pressure_potential *= perm["kxx"][None, :, None]  # ** (1 / 2)
            elif len(perm) == 2:
                # TODO Fix this for tensor permeabilities.
                # Perm has form ``{"kxx": np.ndarray, "kyy": np.ndarray, ...}``.
                # pressure_potential_times_mobility: np.ndarray = (
                #     total_mobility[None, :, None] * pressure_potential
                #     if pressure_key == GLOBAL_PRESSURE
                #     else pressure_potential
                # )
                # fluxes.append()
                ...
            else:
                raise ValueError(
                    "Permeability must be scalar or tensor with zero"
                    + " values off the diagonal."
                )
            pressure_potential *= total_mobility[None, :, None]  # ** (1 / 2)
            return pressure_potential[..., 0] ** 2 + pressure_potential[..., 1] ** 2

        global_pressure_term: Integral = self.quadrature_estimate.integrate(
            integrand_1,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )

        # Complimentary pressure term.
        # complimentary_pressure_coeffs: np.ndarray = pp.get_solution_values(
        #     f"{COMPLIMENTARY_PRESSURE}_postprocessed_coeffs", g_data, iterate_index=0
        # )

        # def integrand_2(
        #     x: np.ndarray,
        # ) -> np.ndarray:
        #     r"""
        #     Returns:
        #         integrand: :math:`Q(x)^2`.

        #     """
        #     nonlocal complimentary_pressure_coeffs
        #     integrand_squareroot: np.ndarray = (
        #         complimentary_pressure_coeffs[..., 0] * x[..., 0] ** 2
        #         + complimentary_pressure_coeffs[..., 1] * x[..., 0] * x[..., 1]
        #         + complimentary_pressure_coeffs[..., 2] * x[..., 0]
        #         + complimentary_pressure_coeffs[..., 3] * x[..., 1] ** 2
        #         + complimentary_pressure_coeffs[..., 4] * x[..., 1]
        #         + complimentary_pressure_coeffs[..., 5]
        #     )
        #     return integrand_squareroot**2

        # complimentary_pressure_term: Integral = self.quadrature_estimate.integrate(
        #     integrand_2,
        #     self.quadpy_elements,
        #     recalc_points=False,
        #     recalc_volumes=False,
        # )

        # def integrand_3(
        #     x: np.ndarray,
        # ) -> np.ndarray:
        #     r"""
        #     Returns:
        #         integrand: :math:`Q(x)^2`.

        #     """
        #     nonlocal complimentary_pressure_coeffs
        #     integrand_squared: np.ndarray = (
        #         complimentary_pressure_coeffs[..., 0] * x[..., 0] ** 2
        #         + complimentary_pressure_coeffs[..., 1] * x[..., 0] * x[..., 1]
        #         + complimentary_pressure_coeffs[..., 2] * x[..., 0]
        #         + complimentary_pressure_coeffs[..., 3] * x[..., 1] ** 2
        #         + complimentary_pressure_coeffs[..., 4] * x[..., 1]
        #         + complimentary_pressure_coeffs[..., 5]
        #     )
        #     return integrand_squared

        # complimentary_pressure_term_not_squared: Integral = (
        #     self.quadrature_estimate.integrate(
        #         integrand_3,
        #         self.quadpy_elements,
        #         recalc_points=False,
        #         recalc_volumes=False,
        #     )
        # )
        # pass
        # # Complimentary pressure term. The pressure is postprocessed from an elementwise
        # # constant form s.t. its cellwise integral equals the constant term.
        # complimentary_pressure: np.ndarray = pp.get_solution_values(
        #     COMPLIMENTARY_PRESSURE, g_data, iterate_index=0
        # )
        # complimentary_pressure_term: Integral = Integral(complimentary_pressure ** 2)

        # FIXME This is just while the term is too high.
        complimentary_pressure_term = Integral(np.zeros(g.num_cells))

        for name, term in zip(
            [
                "energy_norm_saturation_part",
                "energy_norm_global_pressure_part",
                "energy_norm_complimentary_pressure_part",
            ],
            [saturation_term, global_pressure_term, complimentary_pressure_term],
        ):
            pp.shift_solution_values(
                name,
                g_data,
                pp.ITERATE_SOLUTIONS,
                len(self.iterate_indices),
            )
            pp.set_solution_values(
                name,
                term.elementwise,
                g_data,
                iterate_index=0,
            )

    def global_energy_norm(self) -> float:
        r"""Calculate the global in space and local in time energy norm of the numerical
        solution.

        The energy norm is defined similar to [Cancès, C., Pop, I. & Vohralík, M. An a
        posteriori  error estimate for vertex-centered finite volume discretizations of
        immiscible incompressible two-phase flow. Math. Comp. 83, 153–188 (2014).]

        .. math::
            \mathcal{E}_{I_n,\Omega}(p_{n,h,\tau}, s_{w,h,\tau}) :=
                \|s_{w,h,\tau}\|_{L^2(t_{n-1},t_n;H^{-1}(\Omega))}^2
                + \|P(p_{n,h,\tau},s_{w,h,\tau}\|_{L^2(t_{n-1},t_n;H^1_0(\Omega))}^2
                + \|Q(s_{w,h,\tau})\|_{L^2(Q_{t_{n-1},t_n})}^2.

        Here,
            .. math::
                \|v\|_{H^1_0(\Omega)} :=
                \left{\int_\Omega |\kappa^{1/2} \lambda_t \nabla v|^2\right}^{1/2}

        is the energy norm equivalent to the :math:`H^1` norm due to homogeneous
        Dirichlet boundary conditions and positive-definiteness and symmetry of
        :math:`\kappa`.

        """
        self.local_energy_norm()

        g_data = self.mdg.subdomains(return_data=True)[0][1]
        global_terms: list[float] = []
        for energy_norm_term in [
            "energy_norm_saturation_part",
            "energy_norm_global_pressure_part",
            "energy_norm_complimentary_pressure_part",
        ]:
            local_term_new: np.ndarray = pp.get_solution_values(
                energy_norm_term, g_data, iterate_index=0
            )
            local_term_old: np.ndarray = pp.get_solution_values(
                energy_norm_term, g_data, time_step_index=0
            )
            global_terms.append(
                (self.time_manager.dt / 2 * (local_term_new + local_term_old).sum())
                ** (1 / 2)
            )
        global_energy_norm: float = sum(global_terms)
        logger.info(f"Global energy norm: {global_energy_norm}")
        return global_energy_norm


# This could also be a mixin, but by subclassing ``SolutionStrategyReconstruction``, we
# avoid having to pay attention to the order of the different solution strategy classes.
# EstimatesProtocol and SolutionStrategyReconstruction define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SolutionStrategyEst(  # type: ignore
    EstimatesProtocol,
    SolutionStrategyReconstruction,
):

    def prepare_simulation(self) -> None:
        """Set up estimators after setting up the base simulation and
        reconstructions.

        """
        super().prepare_simulation()

        # Setup estimators.
        self.setup_estimates()
        self.initialize_estimate_vals()

    def initialize_estimate_vals(self) -> None:
        # FIXME Get rid of this and instead pass `prepare_simulation` to the local
        # estimate methods.
        """Initialize time step values estimates.

        To calculate the integrals in time, we need to access previous time step values.
        At the initial time step, the continuous solution is equal to the initial
        values (assuming elementwise constant initial values), hence both the residual
        and flux estimators are zero in each cell.

        Note: This is basically doing the same as ``postprocess_solution`` and
        ``after_hc_convergence``, but not everything.

        """
        sd, g_data = self.mdg.subdomains(return_data=True)[0]

        # Initialize time step and iterate values for local estimators.
        for flux_name in ["total", "wetting_from_ff"]:
            pp.set_solution_values(
                f"{flux_name}_R_estimate",
                np.zeros(sd.num_cells),
                g_data,
                time_step_index=0,
                iterate_index=0,
            )
            pp.set_solution_values(
                f"{flux_name}_F_estimate",
                np.zeros(sd.num_cells),
                g_data,
                time_step_index=0,
                iterate_index=0,
            )
        # FIXME This should not be zero!
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            pp.set_solution_values(
                f"{pressure_key}_NC_estimate",
                np.zeros(sd.num_cells),
                g_data,
                time_step_index=0,
                iterate_index=0,
            )
        # Initialize energies with one s.t. we do not encounter a divide by zero error
        # when calculating relative errors.
        for energy_norm_term in [
            "energy_norm_saturation_part",
            "energy_norm_global_pressure_part",
            "energy_norm_complimentary_pressure_part",
        ]:
            pp.set_solution_values(
                energy_norm_term,
                np.ones(sd.num_cells),
                g_data,
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
        residual_and_flux_est: float = self.global_res_and_flux_est()
        global_nonconformity_est, complimentary_nonconfonformity_est = (
            self.global_nonconformity_est()
        )
        global_energy_norm: float = self.global_energy_norm()
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm=None,
            residual_norm=None,
            residual_and_flux_est=residual_and_flux_est,
            nonconformity_est={
                GLOBAL_PRESSURE: global_nonconformity_est,
                COMPLIMENTARY_PRESSURE: complimentary_nonconfonformity_est,
            },
            global_energy_norm=global_energy_norm,
        )
        return converged, diverged

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Update time step values for local in space and time estimates.
        # NOTE In theory, we do not need the shifts! However, for completeness, it does
        # not hurt leaving them in here in case someone wants to store multiple time
        # step values for whatever reason.
        g_data = self.mdg.subdomains(return_data=True)[0][1]
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            pp.shift_solution_values(
                f"{pressure_key}_NC_estimate",
                g_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            pressure_values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_NC_estimate", g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_NC_estimate",
                pressure_values,
                g_data,
                time_step_index=0,
            )
        for flux_name, key in itertools.product(
            ["total", "wetting_from_ff"],
            ["R_estimate", "F_estimate"],
        ):
            pp.shift_solution_values(
                f"{flux_name}_{key}",
                g_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            flux_values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{key}", g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{flux_name}_{key}", flux_values, g_data, time_step_index=0
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
                energy_norm_term, g_data, iterate_index=0
            )
            pp.set_solution_values(
                energy_norm_term, local_term, g_data, time_step_index=0
            )


class DataSavingEst(DataSavingReconstruction):

    def _data_to_export(
        self, time_step_index: Optional[int] = None, iterate_index: Optional[int] = None
    ) -> list[DataInput]:
        """Append error estimates to the exported data."""
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
                ["total", "wetting_from_ff"], ["R_estimate", "F_estimate"]
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
            for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
                data.append(
                    (
                        g,
                        f"{pressure_key}_NC_estimate",
                        pp.get_solution_values(
                            f"{pressure_key}_NC_estimate",
                            g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            for energy_norm_term in [
                "energy_norm_saturation_part",
                "energy_norm_global_pressure_part",
                "energy_norm_complimentary_pressure_part",
            ]:
                data.append(
                    (
                        g,
                        energy_norm_term,
                        pp.get_solution_values(
                            energy_norm_term,
                            g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )

        return data


# EstimatesProtocol and SolutionStrategyReconstruction define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class TwoPhaseFlowErrorEstimate(  # type: ignore
    ErrorEstimateMixin,
    SolutionStrategyEst,
    DataSavingEst,
    TwoPhaseFlowReconstruction,
): ...  # type: ignore
