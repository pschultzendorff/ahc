import functools
import itertools
import logging
import typing
from typing import Any, Literal

import numpy as np
import porepy as pp
from porepy.viz.exporter import DataInput

from tpf.models.protocol import EstimatesProtocol, ReconstructionProtocol, TPFProtocol
from tpf.models.reconstruction import (
    DataSavingRec,
    SolutionStrategyRec,
    TwoPhaseFlowReconstruction,
)
from tpf.numerics.quadrature import Integral, TriangleQuadrature
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


class ErrorEstimateMixin(ReconstructionProtocol, TPFProtocol):
    """Methods to equilibrate fluxes during the Newton iteration.

    Note: If the grid is updated during the simulation, the cellwise Poincare constants
    need to be updatedas well. This is not done by the current implementation.

    """

    def setup_estimates(self) -> None:
        self.quadrature_estimate_degree: int = 4
        """Degree of quadrature rule for the error estimators."""
        self.quadrature_estimate = TriangleQuadrature(self.quadrature_estimate_degree)
        """Quadrature rule for the error estimators.

        Note: For efficiency, :attr:`self.quadrature_reconstructions` could be used.
            However, we may want to have the flexibility to use different degrees for
            estimates and reconstruction.

        """
        self.poincare_constants: np.ndarray = self.poincare_constant(self.g)

    @staticmethod
    def poincare_constant(g: pp.Grid) -> np.ndarray:
        return g.cell_diameters() / np.pi

    def local_residual_est(
        self, flux_name: Literal["total", "wetting_from_ff"]
    ) -> None:
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
            if flux_name == "wetting_from_ff"
            else self.total_fluid_source(self.g)
        ) / self.g.cell_volumes

        equilibrated_flux_coeffs: np.ndarray = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs",
            self.g_data,
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
        if flux_name == "wetting_from_ff":
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
        elif flux_name == "total":
            integral = Integral(
                (
                    self.g.cell_volumes
                    * self.poincare_constants
                    * (div_equilibrated_flux - source) ** 2
                )[..., None]
            )

        pp.set_solution_values(
            f"{flux_name}_R_estimator",
            integral.elementwise.squeeze(),
            self.g_data,
            iterate_index=0,
        )

    def local_flux_est(self, flux_name: Literal["total", "wetting_from_ff"]) -> None:
        r"""Calculate and store the local flux estimator for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,

        .. math::
            \|\mathbf{u}_{\alpha,h,\tau}(t) - \theta_\alpha(t)^{n,i,k}\|_K^2,

        where :math:`\mathbf{u}_{\alpha,h,\tau}(t)` is the FV flux at time :math:`t`,
        i.e., in :math:`\textbf{RTN}_0(\mathca{T}_h)` but not locally mass conservative,
        and :math:`\theta_\alpha(t)^{n,i,k}` is the reconstructed flux at the current
        time step, continuation iteration, and Newton iteration, i.e., both in
        :math:`\textbf{RTN}_0(\mathca{T}_h)` and locally mass conservative.

        :math:`\mathbf{u}_{\alpha,h,\tau}(t)` is time dependent. To integrate in time,
        we use the trapezoidal rule. Thus, the local estimators have to be evaluated
        both for the current iteration and the previous time step. **Crucially**, it
        does not suffice to evaluate the current local estimators at each iteration and
        store after the time step converges. This is, because the equilibrated flux at
        the previous time step is not the same as the current equilibrated flux.

        """
        # First, we calculate the spatial integral of the difference between FV and
        # reconstructed flux elementwise at the current time step.

        # Retrieve FV and equilibrated flux coefficients.
        fv_coeffs_new = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", self.g_data, iterate_index=0
        )
        fv_coeffs_old = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", self.g_data, time_step_index=0
        )
        equilibrated_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs", self.g_data, iterate_index=0
        )

        def integrand(x: np.ndarray, fv_coeffs: np.ndarray) -> np.ndarray:
            integrand_x: np.ndarray = (
                fv_coeffs[..., 0] - equilibrated_coeffs[..., 0]
            ) * x[..., 0] + (fv_coeffs[..., 1] - equilibrated_coeffs[..., 1])
            integrand_y: np.ndarray = (
                fv_coeffs[..., 0] - equilibrated_coeffs[..., 0]
            ) * x[..., 1] + (fv_coeffs[..., 2] - equilibrated_coeffs[..., 2])
            return integrand_x**2 + integrand_y**2

        # Integrate elementwise and store the result.
        for specifier in ["", "_old"]:
            integral: Integral = self.quadrature_estimate.integrate(
                functools.partial(
                    integrand,
                    fv_coeffs=fv_coeffs_new if specifier == "" else fv_coeffs_old,
                ),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{flux_name}_F_estimator{specifier}",
                integral.elementwise.squeeze(),
                self.g_data,
                iterate_index=0,
            )

    def local_nonconformity_est(
        self,
        pressure_key: PRESSURE_KEY,
    ) -> None:
        r"""Calculate and store the local nonconformity estimator for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,

        .. math::
            \|\mathbf{u}_\alpha - \theta_\alpha\|_K^2,

        """

        # First, calculate the three different spatial integrals (norm at current time
        # step, inner product current-previous time step, norm at previous time step) of
        # the difference between post-processed and reconstructed pressure potential.
        # The latter was stored at the previous time step and is not recalculated.
        perm: np.ndarray | dict[str, np.ndarray] = self.permeability(self.g)
        # Calculate a not upwinded total mobility.
        if pressure_key == GLOBAL_PRESSURE:
            rel_perms: dict[str, np.ndarray] = {}
            for phase in self.phases.values():
                rel_perms[phase.name] = self.rel_perm(self.wetting.s, phase).value(  # type: ignore
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
            f"{pressure_key}_postprocessed_coeffs", self.g_data, iterate_index=0
        )
        reconstructed_coeffs: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_reconstructed_coeffs", self.g_data, iterate_index=0
        )

        postprocessed_coeffs_old: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_postprocessed_coeffs", self.g_data, time_step_index=0
        )
        reconstructed_coeffs_old: np.ndarray = pp.get_solution_values(
            f"{pressure_key}_reconstructed_coeffs", self.g_data, time_step_index=0
        )

        def integrand(
            x: np.ndarray,
            values: Literal["", "_inner_product_new_old"],
        ) -> np.ndarray:
            r"""

            The integrand can take two different forms, exemplarily shown for
            complimentary pressure:
            .. math::
                (\kappa \nabla (\tilde{\mathfrac{q}^n} - \mathfrac{q}^n))^2, \\
                \kappa \nabla (\tilde{\mathfrac{q}}^n - \mathfrac{q}^n) \cdot
                \kappa \nabla (\tilde{\mathfrac{q}}^{n-1} - \mathfrac{q}^{n-1}).

            """
            nonlocal \
                reconstructed_coeffs, \
                reconstructed_coeffs_old, \
                postprocessed_coeffs_old, \
                total_mobility, \
                perm
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
                if values == "" and i == 1:
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
            if values == "":
                fluxes.append(fluxes[0])
            return (
                fluxes[0][..., 0] * fluxes[1][..., 0]
                + fluxes[0][..., 1] * fluxes[1][..., 1]
            )

        # Integrate
        # :math:`(\kappa \nabla (\tilde{\mathfrac{q}^n} - \mathfrac{q}^n))^2`
        # and
        # :math:`\kappa \nabla (\tilde{\mathfrac{q}}^n - \mathfrac{q}^n)
        # \cdot \kappa \nabla (\tilde{\mathfrac{q}}^{n-1} - \mathfrac{q}^{n-1})`
        # elementwise and store the result.
        for specifier in ["", "_inner_product_new_old"]:
            # Make mypy happy.
            specifier = typing.cast(Literal["", "_inner_product_new_old"], specifier)
            integral: Integral = self.quadrature_estimate.integrate(
                functools.partial(integrand, values=specifier),
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"{pressure_key}_NC_estimator{specifier}",
                integral.elementwise.squeeze(),
                self.g_data,
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

        Note: The residual estimator is not time dependent, hence we use the same value
        for :math:`t_n` and :math:`t_{n-1}`. Alternatively, we could just multiply the
        value at :math:`t_n` by :math:`\Delta t` to get the time integral. Since the
        estimator is summed with the flux integrator before temporal integration, we use
        the former.

        Returns:
            estimator: The global residual and flux estimator.

        """
        estimators: dict[str, float] = {}
        for flux_name in ["total", "wetting_from_ff"]:
            # Satisfy mypy.
            flux_name = typing.cast(Literal["total", "wetting_from_ff"], flux_name)

            # Calculate local estimatorss.
            self.local_residual_est(flux_name)
            self.local_flux_est(flux_name)
            # Load spatial integrals from current time step.
            local_integral_R: np.ndarray = pp.get_solution_values(
                f"{flux_name}_R_estimator", self.g_data, iterate_index=0
            )
            local_integral_F_new: np.ndarray = pp.get_solution_values(
                f"{flux_name}_F_estimator", self.g_data, iterate_index=0
            )
            # Load spatial integrals from previous time step.
            local_integral_F_old: np.ndarray = pp.get_solution_values(
                f"{flux_name}_F_estimator_old", self.g_data, iterate_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # take the square root first
            global_integral_new: float = (
                (local_integral_R ** (1 / 2) + local_integral_F_new ** (1 / 2)) ** 2
            ).sum()
            global_integral_old: float = (
                (local_integral_R ** (1 / 2) + local_integral_F_old ** (1 / 2)) ** 2
            ).sum()
            # Integrate in time by trapezoidal rule.
            estimators[flux_name] = (
                self.time_manager.dt / 2 * (global_integral_new + global_integral_old)
            ) ** (1 / 2)
        logger.info(
            f"Global residual and flux error estimator: {sum(estimators.values())}"
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

        Returns:
            estimator: The sum of both global nonconformity estimators.

        """
        estimators: dict[str, float] = {}
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            # Satisfy mypy.
            pressure_key = typing.cast(PRESSURE_KEY, pressure_key)
            # Calculate local estimatorss.
            self.local_nonconformity_est(pressure_key)
            # Load spatial integrals from current time step.
            local_integral_NC_new: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_NC_estimator", self.g_data, iterate_index=0
            )
            local_integral_NC_inner_product_new_old: np.ndarray = (
                pp.get_solution_values(
                    f"{pressure_key}_NC_estimator_inner_product_new_old",
                    self.g_data,
                    iterate_index=0,
                )
            )
            # Load spatial integrals from previous time step.
            local_integral_NC_old: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_NC_estimator", self.g_data, time_step_index=0
            )
            # Calculate global values at current and previous time step.
            # NOTE The values stored were the squares of the elementwise norms, hence we
            # do not need to square here.
            global_integral_new: float = (local_integral_NC_new).sum()
            global_integral_inner_product_new_old: float = (
                local_integral_NC_inner_product_new_old**2
            ).sum()
            global_integral_old: float = (local_integral_NC_old).sum()
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
                f"Global {pressure_key} nonconformity error estimator:"
                + f" {estimators[pressure_key]}"
            )

        return estimators[GLOBAL_PRESSURE], estimators[COMPLIMENTARY_PRESSURE]

    def local_energy_norm(self) -> None:
        r"""Calculate the local in space and time energy norm of the numerical
        solution.

        Calculate the local in time and space terms that are summed and integrated in
        :meth:`global_energy_norm`.

        """
        for flux_name in ["total", "wetting_from_ff"]:
            flux_coeffs: np.ndarray = pp.get_solution_values(
                f"{flux_name}_flux_RT0_coeffs", self.g_data, iterate_index=0
            )

            def integrand(
                x: np.ndarray,
            ) -> np.ndarray:
                r"""
                Returns:
                    integrand: :math:`|\bm{u}|^2 = .

                """
                nonlocal flux_coeffs
                # Calculate the potential directly from the coefficients.
                integrand_x: np.ndarray = (
                    x[..., 0] * flux_coeffs[..., 0] + flux_coeffs[..., 1]
                )
                integrand_y: np.ndarray = (
                    x[..., 1] * flux_coeffs[..., 0] + flux_coeffs[..., 2]
                )
                return integrand_x**2 + integrand_y**2

            local_energy: Integral = self.quadrature_estimate.integrate(
                integrand,
                self.quadpy_elements,
                recalc_points=False,
                recalc_volumes=False,
            )
            pp.set_solution_values(
                f"energy_norm_{flux_name}_flux_part",
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
            \mathcal{E}_{I_n,\Omega}(p_{n,h,\tau}, s_{w,h,\tau})
                := \int_{I_n} \int_\Omega |\bm{u}_{\alpha,h,\tau}|^2 \, dx \, dt

        """
        self.local_energy_norm()

        global_energies: list[float] = []
        for flux_name in ["total", "wetting_from_ff"]:
            local_energy_new: np.ndarray = pp.get_solution_values(
                f"energy_norm_{flux_name}_flux_part", self.g_data, iterate_index=0
            )
            local_energy_old: np.ndarray = pp.get_solution_values(
                f"energy_norm_{flux_name}_flux_part", self.g_data, time_step_index=0
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
# EstimatesProtocol and SolutionStrategyReconstruction define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
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

        To calculate the integrals in time, we need to access previous time step values.
        At the initial time step, the continuous solution is equal to the initial
        values (assuming elementwise constant initial values), hence both the residual
        and flux estimators are zero in each cell.

        Note: This is basically doing the same as ``postprocess_solution`` and
        ``after_hc_convergence``, but not everything.

        """
        for flux_name, specifier in itertools.product(
            ["total", "wetting_from_ff"],
            ["R_estimator", "F_estimator", "F_estimator_old", "energy_norm_flux_part"],
        ):
            if specifier.endswith("estimator"):
                name: str = f"{flux_name}_{specifier}"
                initial_values: np.ndarray = np.zeros(self.g.num_cells)
            else:
                name = f"energy_norm_{flux_name}_flux_part"
                # Initialize energies with one s.t. we do not encounter a divide by zero
                # error when calculating relative errors.
                initial_values = np.ones(self.g.num_cells)
            pp.set_solution_values(
                name,
                initial_values,
                self.g_data,
                time_step_index=0,
                iterate_index=0,
            )
        # FIXME This should not be zero!
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            pp.set_solution_values(
                f"{pressure_key}_NC_estimator",
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
        converged, diverged = super().check_convergence(
            nonlinear_increment, residual, reference_residual, nl_params
        )
        residual_and_flux_est: float = self.global_res_and_flux_est()
        global_pressure_nc_est, complimentary_pressure_nc_est = (
            self.global_nonconformity_est()
        )
        global_energy_norm: float = self.global_energy_norm()
        self.nonlinear_solver_statistics.log_error(
            nonlinear_increment_norm=None,
            residual_norm=None,
            residual_and_flux_est=residual_and_flux_est,
            nonconformity_est={
                GLOBAL_PRESSURE: global_pressure_nc_est,
                COMPLIMENTARY_PRESSURE: complimentary_pressure_nc_est,
            },
            global_energy_norm=global_energy_norm,
        )
        return converged, diverged

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()

        # Update time step values for local in space and time estimators.
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            pressure_values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_NC_estimator", self.g_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_NC_estimator",
                pressure_values,
                self.g_data,
                time_step_index=0,
            )

        # NOTE The local residual error estimators are only needed at the current
        # iteration. We set the time step values for completeness and to avoid extra
        # code in :meth:`_data_to_export`.
        # NOTE The RT0 coeffs are needed to calculate the local flux estimators.
        for flux_name, specifier in itertools.product(
            ["total", "wetting_from_ff"],
            ["R_estimator", "F_estimator", "energy_norm_flux_part", "flux_RT0_coeffs"],
        ):
            if specifier.startswith("energy"):
                name: str = f"energy_norm_{flux_name}_flux_part"
            else:
                name = f"{flux_name}_{specifier}"
            flux_values: np.ndarray = pp.get_solution_values(
                name, self.g_data, iterate_index=0
            )
            pp.set_solution_values(name, flux_values, self.g_data, time_step_index=0)


class DataSavingEst(DataSavingRec):
    def _data_to_export(
        self, time_step_index: int | None = None, iterate_index: int | None = None
    ) -> list[DataInput]:
        """Append error estimators to the exported data."""
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index,
            iterate_index=iterate_index,
        )
        for flux_name, key in itertools.product(
            ["total", "wetting_from_ff"],
            ["R_estimator", "F_estimator", "energy_norm_flux_part"],
        ):
            # Before simulation, the estimators won't be set yet, due to the order of
            # calls in :meth:`prepare_simulation`. However, after the first time step,
            # :attr:`time_manager.time_step_index` won't be updated yet. Checking for
            # all of this is quite convoluted. Instead we just use try-except.

            try:
                if key.endswith("estimator"):
                    name: str = f"{flux_name}_{key}"
                else:
                    name = f"energy_norm_{flux_name}_flux_part"
                data.append(
                    (
                        self.g,
                        name,
                        pp.get_solution_values(
                            name,
                            self.g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            except KeyError:
                pass
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            try:
                data.append(
                    (
                        self.g,
                        f"{pressure_key}_NC_estimator",
                        pp.get_solution_values(
                            f"{pressure_key}_NC_estimator",
                            self.g_data,
                            time_step_index=time_step_index,
                            iterate_index=iterate_index,
                        ),
                    )
                )
            except KeyError:
                pass
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
