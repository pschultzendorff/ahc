import json
import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, Optional

import numpy as np
import porepy as pp
import sympy as sym
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from tpf_lab.constants_and_typing import REL_PERM_MODEL, OperatorType
from tpf_lab.models.constitutive_laws_tpf import RelativePermeability, RelPermConstants
from tpf_lab.models.flow_and_transport import RelativePermeability, SolutionStrategyTPF
from tpf_lab.models.phase import Phase
from tqdm.auto import trange

logger = logging.getLogger(__name__)


class HCSolverStatistics(pp.SolverStatistics):
    hc_lambda_fl: float = 1.0
    hc_lambda_ad: pp.ad.Scalar = pp.ad.Scalar(1.0)
    hc_num_iteration: int = 0

    def reset_hc(self) -> None:
        """Reset the homotopy continuation statistics object."""
        self.hc_num_iteration = 0
        self.hc_lambda_fl = 1.0
        self.hc_lambda_ad.set_value(1.0)

    def save(self) -> None:
        """Save the statistics object to a JSON file."""
        if self.path is not None:
            # Check if object exists and append to it
            if self.path.exists():
                with self.path.open("r") as file:
                    data = json.load(file)
            else:
                data = {}

            # Append data - assume the index corresponds to time step
            ind = len(data) + 1
            data[ind] = {
                "nl_num_iteration": self.num_iteration,
                "nl_increment_norms": self.nonlinear_increment_norms,
                "nl_residual_norms": self.residual_norms,
                "hc_num_iteration": self.hc_num_iteration,
                "hc_lambda": self.hc_lambda_fl,
            }

            # Save to file
            with self.path.open("w") as file:
                json.dump(data, file, indent=4)


class RelativePermeabilityHC(RelativePermeability):
    """

    FIXME: Have two different fluxes and add them (weighted) together instead of a
    permeability that changes all the time. -> The system does not need to be assembled
    at each continuation iteration.

    """

    nonlinear_solver_statistics: HCSolverStatistics
    """Homotopy continuation parameters. Normally provided by a mixin of instance
    :class:`SolutionStrategyHC`."""

    _rel_perm_constants_1: RelPermConstants
    _rel_perm_constants_2: RelPermConstants

    def set_rel_perm_constants(self) -> None:
        rel_perm_constants: dict[str, dict] = self.params.get("rel_perm_constants", {})
        rel_perm_1_constants: dict[str, Any] = rel_perm_constants.get("model_1", {})
        rel_perm_2_constants: dict[str, Any] = rel_perm_constants.get("model_2", {})

        self._rel_perm_constants_1 = RelPermConstants.from_dict(rel_perm_1_constants)
        self._rel_perm_constants_2 = RelPermConstants.from_dict(rel_perm_2_constants)

    def rel_perm(self, saturation_w: OperatorType, phase: Phase) -> OperatorType:
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


class SolutionStrategyHC(SolutionStrategyTPF):

    transfer_solution: Callable
    """Provided by a mixin of type :class:`Postprocessing`."""
    extend_normal_fluxes: Callable
    """Provided by a mixin of type :class:`Postprocessing`."""
    reconstruct_potentials: Callable
    """Provided by a mixin of type :class:`Postprocessing`."""
    solid: pp.SolidConstants
    permeability: Callable

    nonlinear_solver_statistics: HCSolverStatistics
    """Solver statistics for the homotopy continuation and nonlinear solver."""

    # FIXME We need to store the global estimators as well. Should they be stored for
    # each nonlinear iteration, each HC iteration or whatsoever?
    residuals_wrt_homotopy: list[float] = []
    """Store the residuals of the equation w.r.t. the current homotopy iteration."""
    residuals_wrt_goal_function: list[float] = []
    r"""Store the residuals of the equation w.r.t. the goal system, i.e., w.r.t.
    :math:`\lambda = 0`."""

    @property
    def hc_indices(self) -> list[int]:
        """Return the indices of the homotopy continuation variables."""
        return [0]

    def set_solver_statistics(self) -> None:
        """Copied code from
        :meth:`pp.solution_strategy.SolutionStrategy.set_solver_statistics` with changes
        to ensure that :attr:`self.statistics` is of type :class:`HCSolverStatistics`.

        """
        # Retrieve the value with a default of pp.SolverStatistics
        statistics = self.params.get("nonlinear_solver_statistics", HCSolverStatistics)
        # Explicitly check if the retrieved value is a class and a subclass of
        # pp.SolverStatistics for type checking.
        if isinstance(statistics, type) and issubclass(statistics, HCSolverStatistics):
            self.nonlinear_solver_statistics = statistics()
        else:
            raise ValueError(
                f"Expected a subclass of HCSolverStatistics, got {statistics}."
            )

    # region HC LOOP
    def before_hc_loop(self) -> None:
        """Reset HC parameter and residuals."""
        self.nonlinear_solver_statistics.reset_hc()
        # Reset residuals arrays.
        self.residuals_wrt_homotopy = []
        self.residuals_wrt_goal_function = []
        # Set the starting estimate to the solution from the previous time step.
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
        # Compute residual (PorePys residual is actually the norm of the solution).
        b = self.linear_system[1]
        self.residuals_wrt_homotopy.append(float(np.linalg.norm(b)) / np.sqrt(b.size))

        # Set homotopy continuation param to zero, compute the residual and reset.
        self.nonlinear_solver_statistics.hc_lambda_ad.set_value(0)
        self.assemble_linear_system()
        b = self.linear_system[1]
        self.residuals_wrt_goal_function.append(
            float(np.linalg.norm(b)) / np.sqrt(b.size)
        )

        # Decay HC parameter.
        self.nonlinear_solver_statistics.hc_lambda_fl *= decay
        self.nonlinear_solver_statistics.hc_lambda_ad.set_value(
            self.nonlinear_solver_statistics.hc_lambda_fl
        )
        self.nonlinear_solver_statistics.hc_num_iteration += 1
        logger.info(
            f"Decayed hc_lambda to"
            + f" {self.nonlinear_solver_statistics.hc_lambda_fl:.2f}"
        )

    def after_hc_convergence(self) -> None:
        """Move to the next time step."""
        self.equation_system.shift_time_step_values(
            max_index=len(self.time_step_indices)
        )
        time_step_solution = self.equation_system.get_variable_values(hc_index=0)
        self.equation_system.set_variable_values(
            time_step_solution, time_step_index=0, additive=False
        )

        self.convergence_status = True
        self._export()
        logger.debug(f'{{"converged": {"true"}}}')
        # Save data (and calculate L2 error only after the time step solution was
        # distributed).
        self.save_data_time_step(
            # self.nonlinear_solver_statistics.residual_norms,
            # self.nonlinear_solver_statistics.num_iteration,
        )

    def hc_check_convergence(
        self,
        hc_params: dict[str, Any],
    ) -> bool:
        """Calculate error estimators and stop if NHC error is small enough."""
        if (
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
        # self.transfer_solution()
        # self.extend_normal_fluxes()
        # self.reconstruct_potentials()
        # hc_error: float = self.HC_error_est()
        # discr_error: float = self.discr_error_est()
        # if hc_error <= self.hc.error_ratio * discr_error:
        #     logger.info(
        #         f"HC error {hc_error} smaller than {self.hc.error_ratio} * "
        #         f"discr_error {discr_error}. Stopping homotopy continuation."
        #     )
        #     self.hc_is_converged = True
        # else:
        #     self.hc_is_converged = False
        # return self.hc_is_converged

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
        # FIXME
        # if self._limit_saturation_change:
        #     self._prev_saturation: np.ndarray = (
        #         self.equation_system.get_variable_values(
        #             variables=[self.primary_saturation_var], time_step_index=0
        #         )
        #     )

    def after_nonlinear_convergence(self) -> None:  # type: ignore
        """Export and move to the next homotopy continuation step.

        When ``self._limit_saturation_change == True``, check if the wetting saturation
        has changed too much

        """
        # FIXME Call ``super().after_nonlinear_convergence()``?
        # FIXME Limit saturation change.
        # If the saturation changes to much, decrease the time step and calculate again.
        # if self._limit_saturation_change:
        #     new_saturation: np.ndarray = self.equation_system.get_variable_values(
        #         variables=[self.primary_saturation_var], iterate_index=0
        #     )
        #     if (
        #         np.max(np.abs(new_saturation - self._prev_saturation))
        #         > self._max_saturation_change
        #     ):
        #         # This is set to false again in ``before_nonlinear_loop``.
        #         # NOTE This is not a very nice solution, however, as of now I didn't
        #         # find a way to pass ``recompute_solution`` to
        #         # ``time_manager.compute_time_step()`` in ``run_time_dependent_model``
        #         # without the code getting really messy.
        #         self.time_manager._recomp_sol = True
        #         self.convergence_status = False
        #         logger.debug(
        #             "Saturation grew to quickly. Trying again with a smaller time step."
        #         )
        #         return None

        # Distribute primary variables.
        # TODO At the moment this sets **all** iterate variables to the time step
        # solution (also secondary variables). This should be changed.
        # Secondary variables are not updated yet, so this will make things confusing.
        self.equation_system.shift_hc_values(max_index=len(self.hc_indices))
        hc_solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.set_variable_values(
            hc_solution, hc_index=0, additive=False
        )

    # endregion
    def HC_error_est(self) -> None:
        """Calculate homotopy continuation error estimator.

        Note: For now, we assume homogeneous Dirichlet and Neuman bc.

        """
        # TODO: Write this function.
        pass

    def discr_error_est(self) -> None:
        """Calculate discretization error estimator.

        Note: For now, we assume homogeneous Dirichlet and Neuman bc. The error is then
        bounded by the sum of the residual estimator, the flux estimator, and the
        nonconformity estimator. As we use cell-centered finite volumes, the flux
        estimator is zero.

        """
        # TODO: Write this function.
        pass

    def residual_est(self) -> None:
        """
        TODO: Is this always zero for CCFV as suggested by chapter 8.4.2 of M. Vohralík,
        “A posteriori error estimates for efficiency and error control in numerical
        simulations”?

        """
        # TODO: Write this function.
        pass

    def total_error(self) -> None:
        """Calculate total error by comparing with the exact solution."""
        # TODO: Write this function.
        pass


class HCSolver:
    def __init__(self, params=None) -> None:
        if params is None:
            params = {}

        default_params: dict[str, Any] = {
            "hc_max_iterations": 10,
            "hc_lambda_min": 0.0,
            "hc_lambda_decay": 0.9,
            # Adaptivity parameters.
            "hc_error_ratio": 0.05,
        }
        """Defines the ratio between the homotopy continuation error and the
        discretization errror at which the Newton homotopy continuation is stopped.

        FIXME Shift this to estimators?

        """
        default_params.update(params)
        self.params = default_params
        self.progress_bar_position: int = params.setdefault("progress_bar_position", 0)

        params["progress_bar_position"] += 1
        self.nonlinear_solver = pp.NewtonSolver(params)

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
