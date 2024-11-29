import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import porepy as pp
import sympy as sym
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from tpf_lab.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    GLOBAL_PRESSURE,
    OperatorType,
)
from tpf_lab.models.constitutive_laws_tpf import RelativePermeability, RelPermConstants
from tpf_lab.models.flow_and_transport import SolutionStrategyTPF
from tpf_lab.models.phase import Phase
from tpf_lab.numerics.quadrature import Integral
from tqdm.auto import trange

logger = logging.getLogger(__name__)


@dataclass
class SolverStatisticsHC(pp.SolverStatistics):
    hc_lambda_fl: float = 1.0
    """Homotopy continuation lambda."""
    hc_lambda_ad: pp.ad.Scalar = pp.ad.Scalar(1.0)
    """Homotopy continuation lambda for automatic differentiation."""
    hc_lambdas: list[float] = field(default_factory=list)
    """List of homotopy continuation lambda values for the current time step."""
    hc_num_iteration: int = 0
    """Number of homotopy continuation iterations performed for current time step."""
    num_iteration: int = 0
    """Number of non-linear iterations performed for current homotopy continuation step.

    """
    nums_iteration: list[int] = field(default_factory=list)
    """Number of non-linear iterations performed for current homotopy continuation step.

    """
    nonlinear_increment_norms_hc: list[list[float]] = field(default_factory=list)
    """List of list of increment magnitudes for each non-linear iteration. Outer list
    are HC iterations, inner list are non-linear iterations.

    """
    residual_norms_hc: list[list[float]] = field(default_factory=list)
    """List of list of residual norms. Outer list are HC iterations, inner list are
    non-linear iterations.

    """
    discretization_est: list[list[float]] = field(default_factory=list)
    """List of list of discretization error estimates. Outer list are HC iterations,
    inner list are non-linear iterations.

    """
    hc_est: list[list[float]] = field(default_factory=list)
    """List of list of homotopy continuation error estimates. Outer list are HC
    iterations, inner list are non-linear iterations.

    """
    linearization_est: list[list[float]] = field(default_factory=list)
    """List of list of linearization error estimates. Outer list are HC iterations,
    inner list are non-linear iterations.

    """

    def log_error(
        self,
        nonlinear_increment_norm: Optional[float] = None,
        residual_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        if not (nonlinear_increment_norm is None or residual_norm is None):
            super().log_error(nonlinear_increment_norm, residual_norm)
        else:
            self.discretization_est[-1].append(kwargs.get("discretization_est", 0.0))
            self.hc_est[-1].append(kwargs.get("hc_est", 0.0))
            self.linearization_est[-1].append(kwargs.get("linearization_est", 0.0))

    def reset(self) -> None:
        """Reset the homotopy continuation statistics object at the start of a new
        Newton loop.

        """
        self.nums_iteration.append(self.num_iteration)
        self.nonlinear_increment_norms_hc.append(self.nonlinear_increment_norms)
        self.residual_norms_hc.append(self.residual_norms)
        super().reset()
        self.discretization_est.append([])
        self.hc_est.append([])
        self.linearization_est.append([])

    def hc_reset(self) -> None:
        """Reset the homotopy continuation statistics object at the start of a new
        continuation loop.

        """
        self.nums_iteration.clear()
        self.nonlinear_increment_norms_hc.clear()
        self.residual_norms_hc.clear()
        self.discretization_est.clear()
        self.hc_est.clear()
        self.linearization_est.clear()

        self.hc_num_iteration = 0
        self.hc_lambda_fl = 1.0
        self.hc_lambda_ad.set_value(1.0)
        self.hc_lambdas.clear()
        self.hc_lambdas.append(1.0)

    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data: dict[int, Any] = json.load(file)
            else:
                data = {}

            # Append data.
            ind: int = len(data) + 1
            # The data is organized into dictionaries for each hc step. Each hc step
            # contains lists with values for all Newton steps.
            data[ind] = {
                i: {
                    "num_iteration": n,
                    "nonlinear_increment_norms": nin,
                    "residual_norms": rn,
                    "discretization_error_estimates": de,
                    "hc_error_estimates": hce,
                    "linearization_error_estimates": le,
                }
                for i, (n, nin, rn, de, hce, le) in enumerate(
                    zip(
                        self.nums_iteration,
                        self.nonlinear_increment_norms_hc,
                        self.residual_norms_hc,
                        self.discretization_est,
                        self.hc_est,
                        self.linearization_est,
                    )
                )
            }
            data[ind].update({"hc_num_iterations": self.hc_num_iteration})
            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


class RelativePermeabilityHC(RelativePermeability):
    """

    FIXME: Have two different fluxes and add them (weighted) together instead of a
    permeability that changes all the time. -> The system does not need to be assembled
    at each continuation iteration.

    """

    nonlinear_solver_statistics: SolverStatisticsHC
    """Homotopy continuation parameters. Normally provided by a mixin of instance
    :class:`SolutionStrategyHC`."""

    _rel_perm_constants_1: RelPermConstants
    _rel_perm_constants_2: RelPermConstants

    hc_rel_perm_toggle: bool = True
    """Toggle between homotopy continuation and goal relative permeabilities.

    The default value is ``True`` to use homotopy continuation relative permeabilities.
    Any method that changes this value, is expected to change it back to ``True`` after
    the call.

    """

    def set_rel_perm_constants(self) -> None:
        rel_perm_constants: dict[str, dict] = self.params.get("rel_perm_constants", {})
        rel_perm_1_constants: dict[str, Any] = rel_perm_constants.get("model_1", {})
        rel_perm_2_constants: dict[str, Any] = rel_perm_constants.get("model_2", {})

        self._rel_perm_constants_1 = RelPermConstants.from_dict(rel_perm_1_constants)
        self._rel_perm_constants_2 = RelPermConstants.from_dict(rel_perm_2_constants)

    def rel_perm(self, saturation_w: OperatorType, phase: Phase) -> OperatorType:
        # Return homotopy continuation relative permeability.
        if self.hc_rel_perm_toggle:
            rel_perm_1: OperatorType = super().rel_perm(
                saturation_w, phase, rel_perm_constants=self._rel_perm_constants_1
            )
            rel_perm_2: OperatorType = super().rel_perm(
                saturation_w, phase, rel_perm_constants=self._rel_perm_constants_2
            )
            return (
                self.nonlinear_solver_statistics.hc_lambda_ad * rel_perm_1
                + (pp.ad.Scalar(1) - self.nonlinear_solver_statistics.hc_lambda_ad)
                * rel_perm_2
            )

        # Return goal relative permeability.
        else:
            return super().rel_perm(
                saturation_w, phase, rel_perm_constants=self._rel_perm_constants_2
            )


class EstimatesHCMixin:

    global_res_and_flux_est: Callable[..., float]
    global_nonconformity_est: Callable[..., tuple[float, float]]

    def local_hc_est(self, flux_name: Literal["total", "wetting"]) -> None:
        """

        We assume the following sub-dictionaries to be present in the data dictionary:
            iterate_dictionary, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in iterate_dictionary will be shifted and updated:
            - {flux_name}_C_estimate, storing the local in time and space continuation
                error estimate.

        """
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        # Retrieve flux w.r.t. goal rel. perm. and nonequilbirated flux coeffs.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", sd_data, iterate_index=0
        )
        fv_goal_rel_perm_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_wrt_goal_rel_perm_RT0_coeffs", sd_data, iterate_index=0
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
        integral: Integral = self.estimate_quadrature.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{flux_name}_C_estimate",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_C_estimate",
            integral.elementwise,
            sd_data,
            iterate_index=0,
        )

    def local_linearization_est(self, flux_name: Literal["total", "wetting"]) -> None:
        """

        We assume the following sub-dictionaries to be present in the data dictionary:
            iterate_dictionary, storing all parameters.
                Stored in ``data[pp.ITERATE_SOLUTIONS]``.

        The following entries in iterate_dictionary will be shifted and updated:
            - {flux_name}_L_estimate, storing the local in time and space linearization
                error estimate.

        """
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        # Retrieve flux w.r.t. goal rel. perm. and nonequilbirated flux coeffs.
        fv_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_RT0_coeffs", sd_data, iterate_index=0
        )
        fv_equilibrated_coeffs = pp.get_solution_values(
            f"{flux_name}_flux_equilibrated_RT0_coeffs", sd_data, iterate_index=0
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
        integral: Integral = self.estimate_quadrature.integrate(
            integrand,
            self.quadpy_elements,
            recalc_points=False,
            recalc_volumes=False,
        )
        pp.shift_solution_values(
            f"{flux_name}_L_estimate",
            sd_data,
            pp.ITERATE_SOLUTIONS,
            len(self.iterate_indices),
        )
        pp.set_solution_values(
            f"{flux_name}_L_estimate",
            integral.elementwise,
            sd_data,
            iterate_index=0,
        )

    def global_discretization_est(self) -> float:
        residual_estimate: float = self.global_res_and_flux_est()
        nonconformity_estimates: tuple[float, float] = self.global_nonconformity_est()
        logger.info(
            "Global discretization error estimate:"
            + f" {3 * residual_estimate + sum(nonconformity_estimates)}"
        )
        return 3 * residual_estimate + sum(nonconformity_estimates)

    def global_hc_est(self) -> float:
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        estimators: dict[str, float] = {}
        for flux_name in ["total", self.wetting.name]:
            # Calculate local estimate.
            self.local_hc_est(flux_name)
            # Load spatial integral from current time step.
            integral_C_new: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_C_estimate", sd_data, iterate_index=0
                )
            )
            # Load spatial integral from previous time step.
            integral_C_old: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_C_estimate", sd_data, time_step_index=0
                )
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
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        estimators: dict[str, float] = {}
        for flux_name in ["total", self.wetting.name]:
            # Calculate local estimate.
            self.local_linearization_est(flux_name)
            # Load spatial integral from current time step.
            integral_L_new: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_L_estimate", sd_data, iterate_index=0
                )
            )
            # Load spatial integral from previous time step.
            integral_L_old: Integral = Integral(
                pp.get_solution_values(
                    f"{flux_name}_L_estimate", sd_data, time_step_index=0
                )
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

    def total_est(self) -> float:
        """Return total error estimate, consisting of discretization, homotopy
        continuation, and linearization components.

        """
        return (
            self.global_discretization_est()
            + self.global_hc_est()
            + self.global_linearization_est()
        )


# TODO Subclass of SolutionStrategyEstMixin or not?
class SolutionStrategyEstHCMixin:
    hc_rel_perm_toggle: bool
    """Toggle between homotopy continuation and goal relative permeabilities. Normally
    provided by a mixin of instance  :class:`RelativePermeabilityHC`."""
    mdg: pp.MixedDimensionalGrid
    wetting: Phase
    nonwetting: Phase
    extend_fv_fluxes: Callable[[str], None]
    equilibrate_flux_during_Newton: Callable[[str], None]
    phase_flux: Callable[[pp.Grid, Phase], pp.ad.Scalar]
    phases: dict[str, Phase]

    def initialize_estimators(self) -> None:
        super().initialize_estimators()
        sd, sd_data = self.mdg.subdomains(return_data=True)[0]
        # Initialize time step values for local estimators.
        for flux_name in ["total", "wetting"]:
            pp.set_solution_values(
                f"{flux_name}_C_estimate",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
            )
            pp.set_solution_values(
                f"{flux_name}_L_estimate",
                np.zeros(sd.num_cells),
                sd_data,
                time_step_index=0,
            )

    def eval_additional_vars(self) -> None:
        """Calculate numerical fluxes w.r.t. the goal relative permeabilities."""
        super().eval_additional_vars()

        sd, d = self.mdg.subdomains(return_data=True)[0]
        # Switch rel. perm. to goal rel. perm, calculate fluxes, and switch back.
        self.hc_rel_perm_toggle = False
        for flux_name in ["total", self.wetting.name, self.nonwetting.name]:
            if flux_name == "total":
                flux: np.ndarray = self.total_flux(sd).value(self.equation_system)
            else:
                flux = self.phase_flux(sd, self.phases[flux_name]).value(
                    self.equation_system
                )
            pp.shift_solution_values(
                f"{flux_name}_flux_wrt_goal_rel_perm",
                d,
                pp.ITERATE_SOLUTIONS,
                max_index=len(self.iterate_indices),
            )
            pp.set_solution_values(
                f"{flux_name}_flux_wrt_goal_rel_perm", flux, d, iterate_index=0
            )
        self.hc_rel_perm_toggle = True

    def postprocess_solution(self) -> None:
        for flux_name in ["total", self.wetting.name]:
            # Calculate RT0 coefficients for the FV fluxes.
            self.extend_fv_fluxes(flux_name)
            # Calculate RT0 coefficients for the equilibrated fluxes.
            self.equilibrate_flux_during_Newton(flux_name)
            self.extend_fv_fluxes(flux_name, flux_specifier="_equilibrated")

        # NOTE The fluxes w.r.t. goal rel. perm are only used to post-process the global
        # and complimentary pressures and do NOT need to be equilibrated.
        for flux_name in [
            "total",
            self.wetting.name,
            self.nonwetting.name,
        ]:
            self.extend_fv_fluxes(flux_name, flux_specifier="_wrt_goal_rel_perm")

        for pressure_key in [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE]:
            self.reconstruct_pressure_vohralik(
                pressure_key, flux_specifier="_wrt_goal_rel_perm"
            )

        self.divergence_mismatch()


class SolutionStrategyHCMixin:

    # transfer_solution: Callable
    # """Provided by a mixin of type :class:`Postprocessing`."""
    # extend_normal_fluxes: Callable
    # """Provided by a mixin of type :class:`Postprocessing`."""
    # reconstruct_potentials: Callable
    # """Provided by a mixin of type :class:`Postprocessing`."""
    solid: pp.SolidConstants
    permeability: Callable

    wetting: Phase
    nonwetting: Phase

    mdg: pp.MixedDimensionalGrid

    nonlinear_solver_statistics: SolverStatisticsHC
    """Solver statistics for the homotopy continuation and nonlinear solver."""

    equation_system: pp.EquationSystem
    time_manager: pp.TimeManager
    time_step_indices: list[int]

    global_discretization_est: Callable[[], float]
    global_hc_est: Callable[[], float]
    global_linearization_est: Callable[[], float]

    save_data_time_step: Callable[[], None]

    @property
    def hc_indices(self) -> list[int]:
        """Return the indices of the homotopy continuation variables."""
        return [0]

    def prepare_simulation(self) -> None:
        super().prepare_simulation()
        self.setup_hc(self.params)

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

        # Set the solution from the previous time step as a first guess.
        assembled_variables = self.equation_system.get_variable_values(
            time_step_index=0
        )
        self.equation_system.set_variable_values(
            assembled_variables, hc_index=0, additive=False
        )

    def before_hc_iteration(self) -> None:
        self.decay_lambda()
        self.increase_hc_index()

    def after_hc_iteration(self) -> None:
        self.compute_hc_decay(
            nl_iterations=self.nonlinear_solver_statistics.num_iteration
        )

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
        # TODO Adjust this based on HC iterations?
        # if not self.time_manager.is_constant:
        #     self.time_manager.compute_time_step(
        #         iterations=self.nonlinear_solver_statistics.num_iteration
        #     )

        # Shift estimates to the next time step.
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
                f"{pressure_key}_{key}", sd_data, hc_index=0
            )
            pp.set_solution_values(
                f"{pressure_key}_{key}", values, sd_data, time_step_index=0
            )
        for flux_name, key in itertools.product(
            ["total", "wetting"],
            ["R_estimate", "C_estimate", "L_estimate"],
        ):
            pp.shift_solution_values(
                f"{flux_name}_{key}",
                sd_data,
                pp.TIME_STEP_SOLUTIONS,
                len(self.time_step_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{key}", sd_data, hc_index=0
            )
            pp.set_solution_values(
                f"{flux_name}_{key}", values, sd_data, time_step_index=0
            )

    def after_hc_failure(self) -> None:
        self.save_data_time_step()

        if self.time_manager.is_constant:
            # We cannot change the constant hc decay.
            logger.info("Homotopy continuation did not converge.")
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
        hc_params: dict[str, Any],
    ) -> bool:
        """Stop if homotopy continuation error is sufficiently small in comparison to
        the discretization error.

        """
        # Check if the HC error is smaller than the discretization error.
        hc_est: float = self.nonlinear_solver_statistics.hc_est[-1][-1]
        discr_error: float = self.nonlinear_solver_statistics.discretization_est[-1][-1]
        if hc_est <= hc_params["hc_error_ratio"] * discr_error:
            logger.info(
                f"HC error {hc_est} smaller than {hc_params['hc_error_ratio']}"
                + f" * discretization error {discr_error}."
                + " Stopping HC loop."
            )
            return True
        # Check all other convergence criteria.
        elif (
            self.nonlinear_solver_statistics.hc_num_iteration
            >= hc_params["hc_max_iterations"]
        ):
            logger.info(
                f"Reached maximum number of homotopy continuation iterations "
                f"{hc_params["hc_max_iterations"]}. Stopping homotopy continuation."
            )
            return True
        elif (
            self.nonlinear_solver_statistics.hc_lambda_fl <= hc_params["hc_lambda_min"]
        ):
            logger.info(
                f"HC parameter decreased below minimal value"
                f" {hc_params["hc_lambda_min"]}. Stopping homotopy continuation."
            )
            return True
        else:
            return False

    # TODO Write an HomotopyContinuationManager that takes care of the following
    # methods.
    def setup_hc(self, hc_params: dict[str, Any]) -> None:
        self.hc_constant_decay: bool = hc_params["hc_constant_decay"]
        self.hc_init_decay: float = hc_params["hc_lambda_decay"]
        self.hc_decay: float = self.hc_init_decay
        self.hc_decay_min_max: tuple[float, float] = hc_params["hc_decay_min_max"]
        self.nl_iter_optimal_range: tuple[int, int] = hc_params["nl_iter_optimal_range"]
        self.nl_iter_relax_factors: tuple[float, float] = hc_params[
            "nl_iter_relax_factors"
        ]
        # FIXME Implement a mechanism that avoids recomputing the decay if the maximum
        # is reached.
        self.hc_decay_recomp_max: int = hc_params["hc_decay_recomp_max"]
        self.hc_decay_recomp_counter: int = 0

    def decay_lambda(self) -> None:
        """Decay lambda by current decay rate."""
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

    def increase_hc_index(self) -> None:
        self.nonlinear_solver_statistics.hc_num_iteration += 1

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
                raise ValueError("Cannot recompute decay.")

    def _hc_adaptation_based_on_recomputation(self) -> None:
        """Same as ``pp.TimeManager._adaptation_based_on_recomputation`` but for
        homotopy continuation.

        """
        self.hc_decay *= self.nl_iter_relax_factors[1]
        # TODO Stop recomputing the decay if the maximum is reached.
        self.hc_decay = min(self.hc_decay, self.hc_decay_min_max[1])
        self.hc_decay_recomp_counter += 1
        logger.info(f"Slowing HC decay. Next decay = {self.hc_decay}.")

    def _hc_adaptation_based_on_iterations(self, iterations: int) -> None:
        """Same as ``pp.TimeManager._adaptation_based_on_iterations`` but for
        homotopy continuation.

        """
        if iterations < self.nl_iter_optimal_range[0]:
            self.hc_decay *= self.nl_iter_relax_factors[0]
            # TODO Stop recomputing the decay if the maximum is reached.
            self.hc_decay = max(self.hc_decay, self.hc_decay_min_max[0])
            logger.info(f"Speeding up HC decay. Next decay = {self.hc_decay}.")
        elif iterations >= self.nl_iter_optimal_range[1]:
            self.hc_decay *= self.nl_iter_relax_factors[1]
            # TODO Stop recomputing the decay if the minimum is reached.
            self.hc_decay = min(self.hc_decay, self.hc_decay_min_max[1])
            logger.info(f"Slowing HC decay. Next decay = {self.hc_decay}.")
        self.hc_decay_recomp_counter = 0

    # endregion

    # region NONLINEAR LOOP
    def before_nonlinear_loop(self) -> None:
        """Set the starting estimate to the solution from the previous continuation
        step."""
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
        # Distribute primary variables.
        # TODO At the moment this sets **all** iterate variables to the time step
        # solution (also secondary variables). This should be changed.
        # Secondary variables are not updated yet, so this will make things confusing.
        self.equation_system.shift_hc_values(max_index=len(self.hc_indices))
        hc_solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.set_variable_values(
            hc_solution, hc_index=0, additive=False
        )

        # Shift estimates to the next hc step.
        _, sd_data = self.mdg.subdomains(return_data=True)[0]
        for pressure_key, key in itertools.product(
            [GLOBAL_PRESSURE, COMPLIMENTARY_PRESSURE],
            ["NC_estimate"],
        ):
            pp.shift_solution_values(
                f"{pressure_key}_{key}",
                sd_data,
                pp.HC_ITERATE_SOLUTIONS,
                len(self.hc_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{pressure_key}_{key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(f"{pressure_key}_{key}", values, sd_data, hc_index=0)
        for flux_name, key in itertools.product(
            ["total", "wetting"],
            ["R_estimate", "C_estimate", "L_estimate"],
        ):
            pp.shift_solution_values(
                f"{flux_name}_{key}",
                sd_data,
                pp.HC_ITERATE_SOLUTIONS,
                len(self.hc_indices),
            )
            values: np.ndarray = pp.get_solution_values(
                f"{flux_name}_{key}", sd_data, iterate_index=0
            )
            pp.set_solution_values(f"{flux_name}_{key}", values, sd_data, hc_index=0)

    def after_nonlinear_failure(self) -> None:
        """Method to be called if the non-linear solver fails to converge."""
        self.save_data_time_step()

        if self.hc_constant_decay:
            # We cannot change the constant HC decay.
            raise ValueError("Nonlinear iterations did not converge.")
        else:
            # Reset lambda.
            self.nonlinear_solver_statistics.hc_lambda_fl /= self.hc_decay
            self.compute_hc_decay(recompute_decay=True)

            # Reset the iterate values. This ensures that the initial guess for an
            # unknown time step equals the known time step.
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
        # Set ``converged`` to True if the linearization error is smaller than the
        # HC error.
        hc_est: float = self.global_hc_est()
        linearization_est: float = self.global_linearization_est()
        if linearization_est <= nl_params["nl_error_ratio"] * hc_est:
            logger.info(
                f"Linearization error {linearization_est} smaller than"
                + f" {nl_params['nl_error_ratio']} * HC error {hc_est}."
                + " Stopping Newton loop."
            )
            converged = True

        self.nonlinear_solver_statistics.log_error(
            # NOTE The discretization error estimate does not need to be calculated
            # at this point. After HC convergence is sufficient if we want the code
            # to be more efficient.
            discretization_est=self.global_discretization_est(),
            hc_est=hc_est,
            linearization_est=linearization_est,
        )
        return converged, diverged

    # endregion


class HCSolver:
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}

        default_params: dict[str, Any] = {
            "hc_max_iterations": 20,
            "hc_lambda_min": 0.0,
            # HC decay parameters.
            "hc_constant_decay": True,
            "hc_lambda_decay": 0.9,
            "hc_decay_min_max": (0.1, 0.9),
            "nl_iter_optimal_range": (4, 7),
            "nl_iter_relax_factors": (0.7, 1.3),
            # Adaptivity parameters.
            "hc_error_ratio": 0.1,
            "nl_error_ratio": 0.1,
        }
        """Defines the ratio between the homotopy continuation error and the
        discretization errror at which the Newton homotopy continuation is stopped.

        FIXME Shift this to estimators?

        """
        default_params.update(params)
        self.params = default_params
        self.progress_bar_position: int = self.params.setdefault(
            "progress_bar_position", 0
        )

        self.params["progress_bar_position"] += 1
        self.nonlinear_solver = pp.NewtonSolver(self.params)

    def solve(self, model) -> bool:
        """Solve the nonlinaer problem using the homotopy continuation (HC) algorithm.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            bool: ``True`` if the HC algorithm is converged.

        """
        hc_is_converged: bool = False
        model.before_hc_loop()

        def hc_step() -> None:
            nonlocal hc_is_converged
            model.before_hc_iteration()
            nl_is_converged: bool = self.nonlinear_solver.solve(model)
            if nl_is_converged:
                model.after_hc_iteration()
                hc_is_converged = model.hc_check_convergence(self.params)

        # Redirect the root logger, s.t. no logger interferes with the progressbars.
        with logging_redirect_tqdm([logging.root]):
            # Initialize a progress bar. Length is the number of maximal Newton
            # iterations.
            hc_progressbar = trange(  # type: ignore
                self.params["hc_max_iterations"],
                desc="HC loop",
                position=self.progress_bar_position,
                leave=False,
                dynamic_ncols=True,
            )

            while (
                model.nonlinear_solver_statistics.hc_num_iteration
                <= self.params["hc_max_iterations"]
                and not hc_is_converged
            ):
                hc_progressbar.set_description_str(
                    "HC iteration number "
                    + f"{model.nonlinear_solver_statistics.hc_num_iteration + 1} of"
                    + f" {self.params['hc_max_iterations']}"
                )
                hc_progressbar.set_postfix_str(
                    f"lambda = {model.nonlinear_solver_statistics.hc_lambda_fl:.2f}"
                )
                hc_step()
                hc_progressbar.update(n=1)

            if hc_is_converged:
                model.after_hc_convergence()
            else:
                model.after_hc_failure()
        hc_progressbar.close()
        return hc_is_converged
        return hc_is_converged
