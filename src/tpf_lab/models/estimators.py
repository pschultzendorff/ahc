import functools
import itertools
import logging
from typing import Callable, Literal

import numpy as np
import porepy as pp
from tpf_lab.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    OperatorType,
)
from tpf_lab.models.flow_and_transport import SolutionStrategyTPF
from tpf_lab.models.phase import Phase
from tpf_lab.models.reconstructions import SolutionStrategyReconstructions
from tpf_lab.numerics.quadrature import (
    GaussLegendreQuadrature1D,
    Integral,
    TriangleQuadrature,
    get_quadpy_elements,
)

logger = logging.getLogger(__name__)

# TODO:


class Estimates:

    phases: list[Phase]
    wetting: Phase
    nonwetting: Phase
    time_manager: pp.TimeManager
    equation_system: pp.ad.EquationSystem

    iterate_indices: list[int]
    time_step_indices: list[int]

    mdg: pp.MixedDimensionalGrid

    _porosity: Callable[[pp.Grid], np.ndarray]
    _permeability: Callable[[pp.Grid], np.ndarray]

    phase_fluid_source: Callable[[pp.Grid, Phase], np.ndarray]
    total_fluid_source: Callable[[pp.Grid], np.ndarray]

    rel_perm: Callable[[pp.ad.Operator, Phase], pp.ad.Operator]

    quadpy_elements: np.ndarray

    def setup_estimates(self):
        # TODO Not really necessary, since we have a :class`TriangleQuadrature` instance
        # from the reconstructions anyways!
        self.estimate_quadrature_degree: int = 4
        self.estimate_quadrature = TriangleQuadrature(self.estimate_quadrature_degree)

    def poincare_constant(self, g: pp.Grid) -> np.ndarray:
        return g.cell_diameters() / np.pi

    def residual_est(self, flux_name: Literal["total", "wetting"]) -> None:
        sd, sd_data = self.mdg.subdomains(return_data=True)[0]

        # Collect terms for the estimators as ``np.ndarray``.
        poincare_constant: np.ndarray = self.poincare_constant(sd)

        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s_ad = pp.ad.time_derivatives.dt(self.wetting.s, dt)
        dt_s: np.ndarray = (dt_s_ad).value(self.equation_system)

        porosity: np.ndarray = self._porosity(sd)
        source: np.ndarray = (
            self.phase_fluid_source(sd, self.wetting)
            if flux_name == "wetting"
            else self.total_fluid_source(sd)
        )

        equilibrated_flux_coeffs: np.ndarray = pp.get_solution_values(
            f"{flux_name}_equilibrated_flux_RT0_coeffs",
            sd_data,
            iterate_index=0,
        )
        # Divergence of a Raviart-Thomas basis function is given by twice the linear
        # coefficient:
        # :math:`\nabla \cdot \begin{pmatrix} ax + b \\ ay + c\end{pmatrix} = 2 a`
        div_equilibrated_flux: np.ndarray = 2 * equilibrated_flux_coeffs[:, 0]

        # Integrate elementwise and store the result.
        # NOTE The saturation term, source term, and divergence of the flux
        # reconstruction, are all elementwise constant. Therefore, we can integrate
        # explicitely by multiplying with the element volume.
        if flux_name == "wetting":
            integral_L2_new: Integral = Integral(
                poincare_constant
                * (porosity * dt_s + div_equilibrated_flux - source) ** 2
            )
        elif flux_name == "total":
            integral_L2_new = Integral(
                poincare_constant * (div_equilibrated_flux - source) ** 2
            )

        # TODO Instead of storing the values at every iterate index and setting the
        # time step values after nonlinear convergence, we can also just set the
        # time step values here and overwrite after each nonlinear iteration. This
        # loses some clarity, but is probably more efficient.
        pp.shift_solution_values(
            f"{flux_name}_R_integral",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_R_integral",
            integral_L2_new.elementwise,
            sd_data,
            iterate_index=0,
        )

        # Load spatial integral from previous time step.
        integral_L2_old: Integral = Integral(
            pp.get_solution_values(
                f"{flux_name}_R_integral", sd_data, time_step_index=0
            )
        )

        # Integrate in time by trapezoidal rule.
        estimator: Integral = (
            self.time_manager.dt / 2 * (integral_L2_new + integral_L2_old)
        )

        logger.info(f"Global residual estimate {flux_name}: {estimator.sum()}")

    def flux_est(self, flux_name: Literal["total", "wetting"]) -> None:
        r"""

        .. math::
            \|\mathbf{u}_t - \theta_\alpha\|_K,

        where :math:`\mathbf{u}_\alpha` is the FV flux at the current time step,
        continuation iteration, and Newton iteration, i.e., in
        :math:`\textbf{RTN}_0(\mathca{T}_h)` but no locally mass conservative, and
        :math:`\theta_\alpha` is a reconstructed flux, i.e., both in
        :math:`\textbf{RTN}_0(\mathca{T}_h)` and locally mass conservative.

        """
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        # First, we calculate the spatial integral of the difference between FV and
        # reconstructed flux elementwise at the current time step.

        # Retrieve FV and equilibrated flux coefficients.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", sd_data, iterate_index=0
        )
        equilibrated_coeffs = pp.get_solution_values(
            f"{flux_name}_equilibrated_flux_RT0_coeffs", sd_data, iterate_index=0
        )

        def integrand(x: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs[..., 0] - equilibrated_coeffs[..., 0]
            ) * x[..., 0] + (fv_coeffs[..., 1] - equilibrated_coeffs[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs[..., 0] - equilibrated_coeffs[..., 0]
            ) * x[..., 1] + (fv_coeffs[..., 2] - equilibrated_coeffs[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise and store the result.
        integral_L2_new: Integral = self.estimate_quadrature.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        # TODO Instead of storing the values at every iterate index and setting the
        # time step values after nonlinear convergence, we can also just set the
        # time step values here and overwrite after each nonlinear iteration. This
        # loses some clarity, but is probably more efficient.
        pp.shift_solution_values(
            f"{flux_name}_F_integral",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_F_integral",
            integral_L2_new.elementwise,
            sd_data,
            iterate_index=0,
        )

        # Load spatial integral from previous time step.
        integral_L2_old: Integral = Integral(
            pp.get_solution_values(
                f"{flux_name}_F_integral", sd_data, time_step_index=0
            )
        )

        # Integrate in time by trapezoidal rule.
        estimator: Integral = (
            self.time_manager.dt / 2 * (integral_L2_new + integral_L2_old)
        )

        logger.info(f"Global flux estimate {flux_name}: {estimator.sum()}")

    def nonconformity_est(
        self,
        pressure_key: Literal[GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
    ) -> None:
        r"""

        We follow the construction of Vohralík and M. F. Wheeler, “A posteriori error
        estimates, stopping criteria, and adaptivity for two-phase flows,” 2013,
        doi:10.1007/s10596-013-9356-0.
        Exemplarily, the time integral of the complimentary pressure nonconformity
        estimator is approximated by
        .. math::
            \int_{t_{n-1}}^{t_n} \eta_{NC,K}(t) \, dt
            \approx \frac{|t_n - t_{n-1}}{3}
                \left(
                    \eta_{NC,K}(t_n)
                    + \eta_{NC,K}(t_{n-1})
                    + (\kappa \nabla (\tilde{\mathfrac{q}}^n - {\mathfrac{q}}^n),
                        \tilde{\mathfrac{q}}^{n-1} - {\mathfrac{q}}^{n-1}))_K
                \right),

        where :math:`\tilde{\mathfrac{q}}` is the reconstructed complimentary pressure
        and :math:`\mathfrac{q}` is the post-processed complimentary pressure.

        """
        sd, sd_data = self.mdg.subdomains(return_data=True)[0]

        # First, calculate the three different spatial integrals (norm at current time
        # step, inner product current-previous time step, norm at previous time step) of
        # the difference between post-processed and reconstructed pressure potential.
        # The latter was stored at the previous time step and is not recalculated.
        perm: np.ndarray = self._permeability(sd)
        # Calculate a not upwinded total mobility.
        if pressure_key == GLOBAL_PRESSURE:
            rel_perms: dict[str, np.ndarray] = {}
            for phase in self.phases:
                rel_perms[phase.name] = self.rel_perm(self.wetting.s, phase).value(
                    self.equation_system
                )
            total_mobility: np.ndarray = (
                rel_perms[self.wetting.name] / self.wetting.constants.viscosity()
                + rel_perms[self.nonwetting.name]
                / self.nonwetting.constants.viscosity()
            )

        # Get postprocessed pressure coefficients and reconstructed pressure points
        # values/coefficients from the current and previous time step.
        # NOTE If ``self._permeability`` is a scalar, ``postprocessed_coeffs[...,1]``
        # and ``reconstructed_coeffs[...,1]`` are zero.
        postprocessed_coeffs: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_postprocessed_coeffs", sd_data, iterate_index=0
        )
        reconstructed_coeffs: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_reconstructed_coeffs", sd_data, iterate_index=0
        )

        postprocessed_coeffs_old: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_postprocessed_coeffs", sd_data, time_step_index=0
        )
        reconstructed_coeffs_old: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_reconstructed_coeffs", sd_data, time_step_index=0
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
            nonlocal reconstructed_coeffs, reconstructed_coeffs_old, postprocessed_coeffs_old, postprocessed_coeffs, perm, total_mobility
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
                fluxes.append(
                    perm[None, :, None]
                    * total_mobility[None, :, None]
                    * pressure_potential
                    if pressure_key == GLOBAL_PRESSURE
                    else perm[None, :, None] * pressure_potential
                )
            if values == "L2_new":
                fluxes.append(fluxes[0])
            return (
                fluxes[0][..., 0] * fluxes[1][..., 0]
                + fluxes[0][..., 1] * fluxes[1][..., 1]
            )

        # Integrate :math:`(\kappa \nabla (\tilde{\mathfrac{q}^n} -
        # \mathfrac{q}^n))^2` elementwise and store the result.
        integral_L2_new: Integral = self.estimate_quadrature.integrate(
            functools.partial(integrand, values="L2_new"),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        # TODO Instead of storing the values at every iterate index and setting the
        # time step values after nonlinear convergence, we can also just set the
        # time step values here and overwrite after each nonlinear iteration. This
        # loses some clarity, but is probably more efficient.
        pp.shift_solution_values(
            f"{pressure_key}_NC_integral",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{pressure_key}_NC_integral",
            integral_L2_new.elementwise,
            sd_data,
            iterate_index=0,
        )
        # Integrate :math:`\kappa \nabla (\tilde{\mathfrac{q}}^n - \mathfrac{q}^n)
        # \cdot \kappa \nabla (\tilde{\mathfrac{q}}^{n-1} - \mathfrac{q}^{n-1})`
        # elementwise. This needs to be recalculated at each time step, hence we do
        # not store the result.
        integral_inner_product_new_old: Integral = self.estimate_quadrature.integrate(
            functools.partial(integrand, values="inner_product_new_old"),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        # Load :math:`(\kappa \nabla (\tilde{\mathfrac{q}^{n-1}} -
        # \mathfrac{q}^{n-1}))^2`.
        integral_L2_old: Integral = Integral(
            pp.get_solution_values(
                f"{pressure_key}_NC_integral", sd_data, time_step_index=0
            )
        )
        # Finally, estimate the time integral.
        estimator: Integral = (
            self.time_manager.dt
            / 3
            * (integral_L2_new + integral_L2_old + integral_inner_product_new_old)
        )

        logger.info(f"Global nonconformity estimate {pressure_key}: {estimator.sum()}")

    def integrate_res_and_flux_est_in_time(self) -> None:
        pass

    def integrate_nonconformity_est_in_time(self) -> None:
        pass

    def integrate_discretization_est_in_time(self) -> None:
        pass

    def integrate_continuation_est_in_time(self) -> None:
        pass

    def integrate_linearization_est_in_time(self) -> None:
        pass


class SolutionStrategyEst(SolutionStrategyReconstructions):

    setup_estimates: Callable[[], None]
    residual_est: Callable[[str], tuple[float, float]]
    flux_est: Callable[[str], tuple[float, float]]
    nonconformity_est: Callable[[str], None]

    # TODO Make sure that we don't need this anymore. The current time step values do
    # not get updated until AFTER convergence. Hence to obtain the previous time step
    # values, one just simply calls ``pp.get_solution_values(..., time_step_index=0)``.
    # @property
    # def time_step_indices(self) -> np.ndarray:
    #     """Indices for storing solutions at previous timesteps. To integrate estimators
    #     in time, we need data from previous time steps."""
    #     return np.array([0, 1])

    def prepare_simulation(self) -> None:
        super().prepare_simulation()
        self.setup_estimates()
        self.initialize_estimators()

    def initialize_estimators(self) -> None:
        """Initialize time step values for reconstructed pressures and equilibrated
        fluxes.

        To calculate the integrals in time, we need to access previous time step values.
        At the initial time step, the continuous solution is equal to the initial
        values (assuming elementwise constant initial values), hence both the residual
        and flux estimators are zero in each cell.

        Note: This is basically doing the same as ``postprocess_solution`` and
        ``after_hc_convergence``, but not everything.

        """
        sd, sd_data = self.mdg.subdomains(return_data=True)[0]

        for flux_name in ["total", "wetting"]:
            pp.set_solution_values(
                f"{flux_name}_R_integral",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
            )
            pp.set_solution_values(
                f"{flux_name}_F_integral",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
            )

        # Calculate values.
        # NOTE The following fluxes are only used to post-process the global and
        # complimentary pressures.
        # for flux_name in [
        #     "inverse_mobility_times_total",
        #     "inverse_mobility_times_complimentary",
        # ]:
        for flux_name in [
            "total",
            self.wetting.name,
            self.nonwetting.name,
        ]:
            self.extend_fv_fluxes(flux_name)
        # for pressure_key, flux_name in zip(
        #     [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
        #     ["inverse_mobility_times_total", "inverse_mobility_times_complimentary"],
        # ):
        #     self.reconstruct_pressure_vohralik(pressure_key, flux_name)
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            self.reconstruct_pressure_vohralik(pressure_key)

        # Initialize time step values.
        for pressure_key, key in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            [
                "postprocessed_coeffs",
                "reconstructed_coeffs",
            ],
        ):
            values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_{key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_{key}", values, sd_data, time_step_index=0
            )
        # FIXME This is not zero!
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            pp.set_solution_values(
                f"{pressure_key}_NC_integral",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
            )

    def after_nonlinear_iteration(self, nonlinear_increment: np.ndarray) -> None:
        super().after_nonlinear_iteration(nonlinear_increment)
        if (
            self.time_manager.time_index > 1
            or self.nonlinear_solver_statistics.num_iteration > 1
        ):
            self.nonconformity_est(GLOBAL_PRESSURE)
            self.nonconformity_est(COMPLIMENTARY_PRESSURE)
            self.residual_est("total")
            self.residual_est("wetting")
            self.flux_est("total")
            self.flux_est("wetting")

    def after_nonlinear_convergence(self) -> None:
        # TODO Once HC is fully implemented, delete this. The work is done by
        # ``after_hc_convergence``.
        super().after_nonlinear_convergence()
        # Shift reconstructed pressure coefficients and spatial integrals of the
        # estimators and set new time step values.
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        for pressure_key, key in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["NC_integral"],
        ):
            # FIXME The shifting is not needed!
            pp.shift_solution_values(
                f"{pressure_key}_{key}",
                sd_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_{key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_{key}", values, sd_data, time_step_index=0
            )
        for flux_name, key in itertools.product(
            ["total", "wetting"],
            ["R_integral", "F_integral"],
        ):
            pp.shift_solution_values(
                f"{flux_name}_{key}",
                sd_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{flux_name}_{key}", values, sd_data, time_step_index=0
            )

    def after_hc_convergence(self) -> None:
        # Shift reconstructed pressure coefficients and spatial integrals of the
        # estimators and set new time step values.
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        for pressure_key, key in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["NC_integral"],
        ):
            pp.shift_solution_values(
                f"{pressure_key}_{key}",
                sd_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_{key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_{key}", values, sd_data, time_step_index=0
            )
        for flux_name, key in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["R_integral", "F_integral"],
        ):
            pp.shift_solution_values(
                f"{flux_name}_{key}",
                sd_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{flux_name}_{key}", values, sd_data, time_step_index=0
            )
