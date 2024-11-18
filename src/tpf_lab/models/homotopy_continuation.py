import json
import logging
from dataclasses import dataclass, field
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
class SolverStatisticsHCMixin:
    hc_lambda_fl: float = 1.0
    hc_lambda_ad: pp.ad.Scalar = pp.ad.Scalar(1.0)
    hc_lambdas: list[float] = field(default_factory=list)
    hc_num_iteration: int = 0

    discretization_est: list[float] = field(default_factory=list)
    """List of discretization error estimates for each non-linear iteration."""
    hc_est: list[float] = field(default_factory=list)
    """List of homotopy continuation error estimates for each non-linear iteration."""
    linearization_est: list[float] = field(default_factory=list)
    """List of linearization error estimates for each non-linear iteration."""

    def log_error(
        self,
        nonlinear_increment_norm: Optional[float] = None,
        residual_norm: Optional[float] = None,
        **kwargs,
    ) -> None:
        if not (nonlinear_increment_norm is None or residual_norm is None):
            super().log_error(nonlinear_increment_norm, residual_norm, **kwargs)
        else:
            self.discretization_est.append(kwargs.get("discretization_est", 0.0))
            self.hc_est.append(kwargs.get("hc_est", 0.0))
            self.linearization_est.append(kwargs.get("linearization_est", 0.0))

    def reset(self) -> None:
        """Reset the homotopy continuation statistics object at the start of a new
        Newton loop.

        """
        super().reset()
        self.discretization_est.clear()
        self.hc_est.clear()
        self.linearization_est.clear()

    def hc_reset(self) -> None:
        """Reset the homotopy continuation statistics object at the start of a new
        continuation loop.

        """
        self.hc_num_iteration = 0
        self.hc_lambda_fl = 1.0
        self.hc_lambda_ad.set_value(1.0)
        self.hc_lambdas.clear()
        self.hc_lambdas.append(1.0)

    def save(self) -> None:
        """Save the estimator statistics to a JSON file."""
        # This calls ``pp.SolverStatistics.save``, which adds a new entry to the
        # ``data`` dictionary that is found at ``self.path``.
        # TODO Do not call super here! Instead, at each time step create another
        # dictionary for all the hc steps.
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
            # FIXME Organize the data structure s.t. it takes into account the hc
            # iterations.
            # Since data was stored and loaded as json, the keys have turned to strings.
            data[str(ind)].update(
                {
                    "hc_lambdas": self.hc_lambdas,
                    "hc_num_iterations": self.hc_num_iteration,
                    "discretization_est": self.discretization_est,
                    "hc_est": self.hc_est,
                    "linearization_est": self.linearization_est,
                }
            )

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


@dataclass
class SolverStatisticsHC(SolverStatisticsHCMixin, pp.SolverStatistics): ...


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

    def global_linearization_est(self) -> None:
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
            self.extend_fv_fluxes(flux_name, flux_specifier="_wrt_goal_rel_perm")
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

    nonlinear_solver_statistics: SolverStatisticsHC
    """Solver statistics for the homotopy continuation and nonlinear solver."""

    equation_system: pp.EquationSystem
    time_manager: pp.TimeManager
    time_step_indices: list[int]

    global_discretization_est: Callable[[], float]
    global_hc_est: Callable[[], float]
    global_linearization_est: Callable[[], float]

    @property
    def hc_indices(self) -> list[int]:
        """Return the indices of the homotopy continuation variables."""
        return [0]

    # region HC LOOP
    def before_hc_loop(self) -> None:
        """Reset HC parameter and residuals."""
        self.nonlinear_solver_statistics.hc_reset()
        # Set the solution from the previous time step as a first guess.
        assembled_variables = self.equation_system.get_variable_values(
            time_step_index=0
        )
        self.equation_system.set_variable_values(
            assembled_variables, hc_index=0, additive=False
        )

    def before_hc_iteration(self) -> None:
        pass

    def after_hc_iteration(self, decay: float) -> None:
        """Decay HC parameter and compute residuals."""
        self.nonlinear_solver_statistics.hc_lambda_fl *= decay
        self.nonlinear_solver_statistics.hc_lambda_ad.set_value(
            self.nonlinear_solver_statistics.hc_lambda_fl
        )
        self.nonlinear_solver_statistics.hc_num_iteration += 1
        self.nonlinear_solver_statistics.hc_lambdas.append(
            self.nonlinear_solver_statistics.hc_lambda_fl
        )
        logger.info(
            f"Decayed hc_lambda to"
            + f" {self.nonlinear_solver_statistics.hc_lambda_fl:.2f}"
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
        # Save data (and calculate L2 error only after the time step solution was
        # distributed).
        self.save_data_time_step(
            # self.nonlinear_solver_statistics.residual_norms,
            # self.nonlinear_solver_statistics.num_iteration,
        )
        # Update the time step magnitude if the dynamic scheme is used.
        # TODO Adjust this based on HC iterations?
        # if not self.time_manager.is_constant:
        #     self.time_manager.compute_time_step(
        #         iterations=self.nonlinear_solver_statistics.num_iteration
        #     )

    def hc_check_convergence(
        self,
        hc_params: dict[str, Any],
    ) -> bool:
        """Stop if homotopy continuation error is sufficiently small in comparison to
        the discretization error.

        """
        # Check if the HC error is smaller than the discretization error.
        hc_est: float = self.nonlinear_solver_statistics.hc_est[-1]
        discr_error: float = self.nonlinear_solver_statistics.discretization_est[-1]
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

    # endregion

    # region NONLINEAR LOOP
    def before_nonlinear_loop(self) -> None:
        """Set the starting estimate to the solution from the previous continuation
        step."""
        self.time_manager._recomp_sol = False
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
        self.nonlinear_solver_statistics.save()

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
        # TODO Remove this weird if statement.
        # if (
        #     self.time_manager.time_index >= 1
        #     or self.nonlinear_solver_statistics.num_iteration > 1
        # ):
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

    def total_error(self) -> None:
        """Calculate total error by comparing with the exact solution."""
        # TODO: Write this function.
        pass


class HCSolver:
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}

        default_params: dict[str, Any] = {
            "hc_max_iterations": 20,
            "hc_lambda_min": 0.0,
            "hc_lambda_decay": 0.9,
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
            self.nonlinear_solver.solve(model)
            model.after_hc_iteration(self.params["hc_lambda_decay"])
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

        hc_progressbar.close()
        return hc_is_converged
