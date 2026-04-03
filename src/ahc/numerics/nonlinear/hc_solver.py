import logging
from typing import Any

import porepy as pp
from ahc.models.protocol import HCProtocol
from porepy.utils.ui_and_logging import (
    logging_redirect_tqdm_with_level as logging_redirect_tqdm,
)
from tqdm.auto import trange  # type: ignore [import-untyped]


class HCSolver:
    """A solver for nonlinear problems using the homotopy continuation (HC) algorithm."""

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
            "hc_adaptive": False,
            "hc_error_ratio": 0.1,
            "nl_error_ratio": 0.1,
        }
        default_params.update(params)
        self.params = default_params
        self.progress_bar_position: int = self.params.setdefault(
            "progress_bar_position", 0
        )

        self.params["progress_bar_position"] += 1
        self.nonlinear_solver = pp.NewtonSolver(self.params)

    def solve(self, model: HCProtocol) -> tuple[bool, bool]:
        """Solve the nonlinaer problem using the homotopy continuation (HC) algorithm.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            is_converged: ``True`` if the HC algorithm is converged.
            is_diverged: ``True`` if the HC algorithm is diverged.

        """
        model.hc_is_converged = False
        model.hc_is_diverged = False
        model.before_hc_loop()

        def hc_step() -> None:
            model.before_hc_iteration()
            nl_is_converged, nl_is_diverged = self.nonlinear_solver.solve(model)
            model.after_hc_iteration()
            model.hc_check_convergence(nl_is_converged, nl_is_diverged, self.params)

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
                and not model.hc_is_converged
                and not model.hc_is_diverged
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

            if model.hc_is_converged:
                model.after_hc_convergence()
            else:
                model.after_hc_failure()
        hc_progressbar.close()
        return model.hc_is_converged, model.hc_is_diverged
