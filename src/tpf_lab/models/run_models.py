""" This module contains functions to run stationary and time-dependent models.

This is mostly just a copy of the PorePy module, but with tqdm functionality for
displaying the time step.

"""

import logging
from typing import Union

import porepy as pp

from src.tpf_lab.numerics.nonlinear.nonlinear_solvers import NewtonSolver
from src.tpf_lab.utils import is_notebook

if is_notebook():
    import tqdm.notebook as tqdm
else:
    import tqdm

logger = logging.getLogger(__name__)


def run_stationary_model(model, params: dict) -> None:
    """Run a stationary model.

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate model for documentation.
        params: Parameters related to the solution procedure. # Why not just set these
            as e.g. model.solution_parameters.

    """

    model.prepare_simulation()

    solver: Union[pp.LinearSolver, NewtonSolver]
    if model._is_nonlinear_problem():
        solver = NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    solver.solve(model)

    model.after_simulation()


def run_time_dependent_model(model, params: dict) -> None:
    """Run a time dependent model.

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure. # Why not just set these
            as e.g. model.solution_parameters.

    """
    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Assign a solver
    solver: Union[pp.LinearSolver, NewtonSolver]
    if model._is_nonlinear_problem():
        solver = NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    # Time loop
    expected_timesteps: int = (
        int((model._schedule[-1] - model._schedule[0]) / model._time_step)
    ) + 1
    time_bar = tqdm.trange(
        expected_timesteps,
        desc="time loop",
        position=0,
    )

    while model.time_manager.time < model.time_manager.time_final:
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        time_bar.set_description_str(
            f"Time step {model.time_manager.time_index}"
            + f" at time {model.time_manager.time:.1e}"
        )
        time_bar.update(n=1)
        logger.debug(
            f'{{"time index": {model.time_manager.time_index},'
            + f' "time": {model.time_manager.time:.1e},'
            + f' "time step": {model.time_manager.dt:.1e}}}'
        )
        solver.solve(model)
        model.time_manager.compute_time_step()

    model.after_simulation()


def _run_iterative_model(model, params: dict) -> None:
    """Run an iterative model.

    The intended use is for multi-step models with iterative couplings. Only known instance
    so far is the combination of fracture deformation and propagation.

    Parameters:
        model: Model class containing all information on parameters, variables,
            discretization, geometry. Various methods such as those relating to solving
            the system, see the appropriate solver for documentation.
        params: Parameters related to the solution procedure. # Why not just set these
            as e.g. model.solution_parameters.

    """

    # Assign parameters, variables and discretizations. Discretize time-indepedent terms
    if params.get("prepare_simulation", True):
        model.prepare_simulation()

    # Assign a solver
    solver: Union[pp.LinearSolver, NewtonSolver]
    if model._is_nonlinear_problem():
        solver = NewtonSolver(params)
    else:
        solver = pp.LinearSolver(params)

    # Time loop
    while model.time_manager.time < model.time_manager.time_final:
        model.propagation_index = 0
        model.time_manager.increase_time()
        model.time_manager.increase_time_index()
        model.before_propagation_loop()
        logger.debug(
            f"\nTime step {model.time_manager.time_index} at time"
            + f" {model.time_manager.time:.1e} of {model.time_manager.time_final:.1e}"
            + f" with time step {model.time_manager.dt:.1e}"
        )
        while model.keep_propagating():
            model.propagation_index += 1
            solver.solve(model)
        model.after_propagation_loop()

    model.after_simulation()
