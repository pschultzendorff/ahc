import logging

from porepy.numerics.nonlinear.nonlinear_solvers import NewtonSolver

from ahc.models.protocol import TPFProtocol

# ``tqdm`` is not a dependency. Up to the user to install it.
try:
    # Avoid some mypy trouble.
    from porepy.utils.ui_and_logging import (
        logging_redirect_tqdm_with_level as logging_redirect_tqdm,
    )
    from tqdm.autonotebook import trange  # type: ignore

except ImportError:
    _IS_TQDM_AVAILABLE: bool = False
else:
    _IS_TQDM_AVAILABLE = True


# Module-wide logger
logger = logging.getLogger(__name__)


class ModifiedNewtonSolver(NewtonSolver):
    """A modified version of the NewtonSolver class from porepy.

    Overrides the solve problem to pass reference_increment to the model's
    check_convergence method.

    """

    def solve(self, model: TPFProtocol) -> tuple[bool, bool]:
        """Solve the nonlinear problem.

        Parameters:
            model: The model instance specifying the problem to be solved.

        Returns:
            A 2-tuple containing:

            bool:
                True if the solution is converged.

        """
        # NOTE This code is mostly copied from the original NewtonSolver.solve() method
        # from PorePy, with the exception of passing reference_increment to the model's
        # check_convergence method.
        model.before_nonlinear_loop()

        is_converged = False
        is_diverged = False
        nonlinear_increment = model.equation_system.get_variable_values(
            time_step_index=0
        )

        # Initialize the reference residual and increment to check for relative
        # convergence.
        # reference_residual is the residual of the initial guess. It does not change
        # during the nonlinear loop.
        reference_residual = model.assemble_residual()
        # reference_increment is the current nonlinear solution. It is updated after
        # each Newton iteration.
        reference_increment = nonlinear_increment.copy()

        # Define a function that does all the work during one Newton iteration, except
        # for everything ``tqdm`` related.
        def newton_step() -> None:
            # Bind to variables in the outer function.
            nonlocal nonlinear_increment
            nonlocal reference_increment
            nonlocal is_converged
            nonlocal is_diverged

            # Logging.
            logger.info(
                "Newton iteration number "
                + f"{model.nonlinear_solver_statistics.num_iteration}"
                + f" of {self.params['nl_max_iterations']}"
            )

            # Re-discretize the nonlinear term
            try:
                model.before_nonlinear_iteration()

                nonlinear_increment = self.iteration(model)

                model.after_nonlinear_iteration(nonlinear_increment)

                # NOTE The residual is extracted after the solution has been updated by
                # the after_nonlinear_iteration() method.
                residual = model.assemble_residual()
                reference_increment = model.equation_system.get_variable_values(
                    variables=[model.wetting.s, model.nonwetting.p], iterate_index=0
                )

                is_converged, is_diverged = model.check_convergence(
                    nonlinear_increment,
                    residual,
                    reference_increment,
                    reference_residual,
                    self.params,
                )
            # Catch overflows due to divergence.
            except FloatingPointError as error:
                logger.error(f"Newton iteration failed: {error}")
                is_converged = False
                is_diverged = True

        # Progressbars turned off or tqdm not installed:
        if not self.progress_bar or not _IS_TQDM_AVAILABLE:
            while (
                model.nonlinear_solver_statistics.num_iteration
                <= self.params["nl_max_iterations"]
                and not is_converged
            ):
                newton_step()

                if is_diverged:
                    # The nonlinear solver failure is handled after the loop.
                    break
                elif is_converged:
                    model.after_nonlinear_convergence()
                    break

        # Progressbars turned on:
        else:
            # Redirect the root logger, s.t. no logger interferes with the progressbars.
            with logging_redirect_tqdm([logging.root]):
                # Initialize a progress bar. Length is the number of maximal Newton
                # iterations.
                solver_progressbar = trange(  # type: ignore
                    self.params["nl_max_iterations"],
                    desc="Newton loop",
                    position=self.progress_bar_position,
                    leave=False,
                    dynamic_ncols=True,
                )

                while (
                    model.nonlinear_solver_statistics.num_iteration
                    <= self.params["nl_max_iterations"]
                    and not is_converged
                ):
                    solver_progressbar.set_description_str(
                        "Newton iteration number "
                        + f"{model.nonlinear_solver_statistics.num_iteration + 1} of"
                        + f" {self.params['nl_max_iterations']}"
                    )
                    newton_step()

                    # Do not update the progress bar if something failed during a Newton
                    # iteration, because
                    # ``model.nonlinear_solver_statistics.nonlinear_increment_norms``
                    # might be empty.
                    if not is_diverged:
                        solver_progressbar.update(n=1)
                        # Ignore line being too long, because we would need an
                        # additional variable to fix this.
                        solver_progressbar.set_postfix_str(
                            f"increments: s={model.nonlinear_solver_statistics.nl_increment_sat_norms[-1]:.2e}"
                            f" p={model.nonlinear_solver_statistics.nl_increment_press_norms[-1]:.2e}"
                        )

                    if is_diverged:
                        # If the process finishes early, the tqdm bar needs to be
                        # manually closed. See https://stackoverflow.com/a/73175351.
                        solver_progressbar.close()
                        # The nonlinear solver failure is handled after the loop.
                        break
                    elif is_converged:
                        solver_progressbar.close()
                        model.after_nonlinear_convergence()
                        break

        if not is_converged:
            model.after_nonlinear_failure()

        return is_converged, is_diverged
