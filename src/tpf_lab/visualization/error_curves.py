"""This module contains functions to read logs and plot error curves."""

import json

import matplotlib.pyplot as plt
import numpy as np


def read_errors_from_log(filename: str) -> np.ndarray:
    with open(filename, "r") as f:
        errors: list[list[float]] = []
        errors_per_timestep: list[float] = []
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # The line contains no or incorrect JSON. Ignore it.
                continue
            if "time" in data:
                if errors_per_timestep:
                    errors.append(errors_per_timestep)
                errors_per_timestep = []
            elif "error norm" in data:
                errors_per_timestep.append(data["error norm"])
        # Append the errors in the last time step.
        if errors_per_timestep:
            errors.append(errors_per_timestep)
    # The inner lists may be of different length, hence we use this method to convert to
    # an ``np.ndarray`` of floats.
    return np.array([np.array(errors_per_timestep) for errors_per_timestep in errors])


def plot_error_curves(filename: str, errors: np.ndarray) -> None:
    # Colors to differentiate time steps.
    colors = ["red", "blue"]
    color = colors[0]
    # Keep track of the iteration number over all timesteps.
    total_iteration = 0
    plt.figure()
    for errors_per_timestep in errors:
        plt.plot(
            range(total_iteration, total_iteration + errors_per_timestep.shape[0]),
            errors_per_timestep,
            color=color,
        )
        total_iteration += errors_per_timestep.shape[0]
        # Change colors
        color = colors[colors.index(color) - 1]
    plt.xlabel("iteration")
    plt.ylabel("error norm")
    plt.yscale("log")
    plt.savefig(filename)


def analyse_convergence(errors: np.ndarray) -> tuple[float, float]:
    """_summary_



    Parameters:
        errors: _description_

    Returns:
        order: Order of convergence.
        rate: Rate of convergece.

    """
    pass


def plot_error_position(filename: str, failure_time_indices: list[list[float]]) -> None:
    """Plot the failure time step in dependence of position of the source or something
    else.

    NOTE: So far only implemented in 2d.

    TODO: Improve this.

    """
    plt.imshow(np.array(failure_time_indices), extent=(0, 2, 0, 2), origin="lower")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()
    plt.savefig(filename)
