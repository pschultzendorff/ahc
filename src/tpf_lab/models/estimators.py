import functools
import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np
import porepy as pp
from tpf_lab.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    OperatorType,
)
from tpf_lab.models.flow_and_transport import SolutionStrategyTPF
from tpf_lab.models.phase import Phase
from tpf_lab.models.reconstructions import SolutionStrategyReconstructionsMixin
from tpf_lab.numerics.quadrature import (
    GaussLegendreQuadrature1D,
    Integral,
    TriangleQuadrature,
    get_quadpy_elements,
)

logger = logging.getLogger(__name__)


@dataclass
class SolverStatisticsEstMixin:
    residual_and_flux_est: list[float] = field(default_factory=list)
    """List of residual and flux error estimates for each non-linear iteration."""
    nonconformity_est: list[dict[str, float]] = field(default_factory=list)
    """List of nonconformity error estimates for each non-linear iteration."""

    def log_error(
        self,
        nonlinear_increment_norm: Optional[float] = None,
        residual_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        if not (nonlinear_increment_norm is None or residual_norm is None):
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)
        else:
            self.residual_and_flux_est.append(kwargs.get("residual_and_flux_est", 0.0))
            self.nonconformity_est.append(
                kwargs.get(
                    "nonconformity_est",
                    {GLOBAL_PRESSURE: 0.0, COMPLIMENTARY_PRESSURE: 0.0},
                )
            )

    def reset(self) -> None:
        """Reset the estimator lists."""
        super().reset()
        self.residual_and_flux_est.clear()
        self.nonconformity_est.clear()

    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        # This calls ``pp.SolverStatistics.save``, which adds a new entry to the
        # ``data`` dictionary that is found at ``self.path``.
        super().save()
        # Instead of creating a new entry, we load the already created entry and append.
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data.
            ind = len(data)
            # Since data was stored and loaded as json, the keys have turned to strings.
            data[str(ind)].update(
                {
                    "residual_and_flux_est": self.residual_and_flux_est,
                    "nonconformity_est": self.nonconformity_est,
                }
            )

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


@dataclass
class SolverStatisticsEst(SolverStatisticsEstMixin, pp.SolverStatistics): ...


class EstimatesMixin:

    phases: dict[str, Phase]
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

    def local_residual_est(self, flux_name: Literal["total", "wetting"]) -> None:
        r"""Calculate and store the local residual estimate for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,

        .. math::
            \|q_t - \nabla \cdot \theta_t\|_K^2, \\
            \|\varphi \partial_t s_w - q_w+ \nabla \cdot \theta_w\|_K^2.

        """
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
            f"{flux_name}_flux_equilibrated_RT0_coeffs",
            sd_data,
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
        if flux_name == "wetting":
            integral: Integral = Integral(
                poincare_constant
                * (porosity * dt_s + div_equilibrated_flux - source) ** 2
            )
        elif flux_name == "total":
            integral = Integral(
                poincare_constant * (div_equilibrated_flux - source) ** 2
            )

        pp.shift_solution_values(
            f"{flux_name}_R_estimate",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_R_estimate",
            integral.elementwise,
            sd_data,
            iterate_index=0,
        )

    def local_flux_est(self, flux_name: Literal["total", "wetting"]) -> None:
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
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        # First, we calculate the spatial integral of the difference between FV and
        # reconstructed flux elementwise at the current time step.

        # Retrieve FV and equilibrated flux coefficients.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", sd_data, iterate_index=0
        )
        equilibrated_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs", sd_data, iterate_index=0
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
        integral: Integral = self.estimate_quadrature.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{flux_name}_F_estimate",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_F_estimate",
            integral.elementwise,
            sd_data,
            iterate_index=0,
        )

    def local_nonconformity_est(
        self,
        pressure_key: Literal[GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
    ) -> None:
        r"""Calculate and store the local nonconformity estimate for each element.


        Note: The values stored are actually the squares of the elementwise norms, i.e.,

        .. math::
            \|\mathbf{u}_\alpha - \theta_\alpha\|_K^2,

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
            for phase in self.phases.values():
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
            nonlocal reconstructed_coeffs, reconstructed_coeffs_old, postprocessed_coeffs_old, total_mobility
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
        # \mathfrac{q}^n))^2` elementwise, shift values, and store the result.
        integral_L2_new: Integral = self.estimate_quadrature.integrate(
            functools.partial(integrand, values="L2_new"),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{pressure_key}_NC_estimate",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{pressure_key}_NC_estimate",
            integral_L2_new.elementwise,
            sd_data,
            iterate_index=0,
        )
        # Integrate :math:`\kappa \nabla (\tilde{\mathfrac{q}}^n - \mathfrac{q}^n)
        # \cdot \kappa \nabla (\tilde{\mathfrac{q}}^{n-1} - \mathfrac{q}^{n-1})`
        # elementwise.
        integral_inner_product_new_old: Integral = self.estimate_quadrature.integrate(
            functools.partial(integrand, values="inner_product_new_old"),
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.set_solution_values(
            f"{pressure_key}_NC_estimate_inner_product_new_old",
            integral_inner_product_new_old.elementwise,
            sd_data,
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
        _, sd_data = self.mdg.subdomains(return_data=True)[0]

        estimators: dict[str, float] = {}
        for flux_name in ["total", self.wetting.name]:
            # Calculate local estimates.
            self.local_residual_est(flux_name)
            self.local_flux_est(flux_name)
            # Load spatial integrals from current time step.
            integral_R_new: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_R_estimate", sd_data, iterate_index=0
                )
            )
            integral_F_new: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_F_estimate", sd_data, iterate_index=0
                )
            )
            # Load spatial integrals from previous time step.
            integral_R_old: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_R_estimate", sd_data, time_step_index=0
                )
            )
            integral_F_old: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_F_estimate", sd_data, time_step_index=0
                )
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
        _, sd_data = self.mdg.subdomains(return_data=True)[0]

        estimators: dict[str, float] = {}
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            # Calculate local estimates.
            self.local_nonconformity_est(pressure_key)
            # Load spatial integrals from current time step.
            integral_NC_new: Integral = Integral(
                pp.get_solution_values(
                    f"{pressure_key}_NC_estimate", sd_data, iterate_index=0
                )
            )
            integral_NC_inner_product_new_old: Integral = Integral(
                pp.get_solution_values(
                    f"{pressure_key}_NC_estimate_inner_product_new_old",
                    sd_data,
                    iterate_index=0,
                )
            )
            # Load spatial integrals from previous time step.
            integral_NC_old: Integral = Integral(
                pp.get_solution_values(
                    f"{pressure_key}_NC_estimate", sd_data, time_step_index=0
                )
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


# TODO Does this need to be a subclass?
class SolutionStrategyEstMixin(SolutionStrategyReconstructionsMixin):

    setup_estimates: Callable[[], None]
    global_res_and_flux_est: Callable[[], float]
    global_nonconformity_est: Callable[[], tuple[float, float]]

    mdg: pp.MixedDimensionalGrid
    wetting: Phase
    nonwetting: Phase
    extend_fv_fluxes: Callable[[str], None]
    reconstruct_pressure_vohralik: Callable[[str], None]
    postprocess_solution: Callable[[], None]
    time_manager: pp.TimeManager
    nonlinear_solver_statistics: SolverStatisticsEst
    time_step_indices: list[int]

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

        # Extend and equilibrate fluxes. Reconstruct pressures.
        # self.postprocess_solution()
        # Calculate flux coefficients. The flux values were set by
        # ``SolutionStrategyReconstructionsMixin.eval_additional_vars``.
        for flux_name in [
            "total",
            self.wetting.name,
            self.nonwetting.name,
        ]:
            self.extend_fv_fluxes(flux_name)
        #     self.reconstruct_pressure_vohralik(pressure_key, flux_name)
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            self.reconstruct_pressure_vohralik(pressure_key)

        # Initialize time step values. The iterate values were set by
        # ``reconstruct_pressure_vohralik``.
        for pressure_key, coeffs_key in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            [
                "postprocessed_coeffs",
                "reconstructed_coeffs",
            ],
        ):
            values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_{coeffs_key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_{coeffs_key}", values, sd_data, time_step_index=0
            )

        # Initialize time step values for local estimators.
        for flux_name in ["total", "wetting"]:
            pp.set_solution_values(
                f"{flux_name}_R_estimate",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
            )
            pp.set_solution_values(
                f"{flux_name}_F_estimate",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
            )
        # FIXME This should not be zero!
        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            pp.set_solution_values(
                f"{pressure_key}_NC_estimate",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
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
        # if (
        #     self.time_manager.time_index > 1
        #     or self.nonlinear_solver_statistics.num_iteration > 1
        # ):
        residual_and_flux_est: float = self.global_res_and_flux_est()
        global_nonconformity_est, complimentary_nonconfonformity_est = (
            self.global_nonconformity_est()
        )
        self.nonlinear_solver_statistics.log_error(
            residual_and_flux_est=residual_and_flux_est,
            nonconformity_est={
                GLOBAL_PRESSURE: global_nonconformity_est,
                COMPLIMENTARY_PRESSURE: complimentary_nonconfonformity_est,
            },
        )
        return converged, diverged

    def after_nonlinear_convergence(self) -> None:
        # TODO Once HC is fully implemented, delete this. The work is done by
        # ``after_hc_convergence``.
        super().after_nonlinear_convergence()

        # Update time step values for local in space and time estimates.
        # NOTE In theory, we do not need the shifts! However, for completeness, it does
        # not hurt leaving them in here in case someone wants to store multiple time
        # step values for whatever reason.
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        for pressure_key, key in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["NC_estimate"],
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
            ["total", "wetting"],
            ["R_estimate", "F_estimate"],
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
