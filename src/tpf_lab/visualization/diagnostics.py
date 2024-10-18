"""Improve the ``pp.DiagnosticsMixin`` by allowing the ability to save the figure."""

import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from porepy.viz.data_saving_model_mixin import VerificationDataSaving

try:
    # MyPy is not happy with Seaborn since it's not typed. We silence this warning.
    import seaborn as sns  # type: ignore[import]
except ImportError:
    _IS_SEABORN_AVAILABLE: bool = False
else:
    _IS_SEABORN_AVAILABLE = True


@dataclass
class TwoPhaseFlowSaveData:
    solution_norms: list[float]
    residuals_wrt_homotopy: list[float]
    residuals_wrt_goal_function: list[float]
    """Only used for models with homotopy continuation."""

    iteration_counter: int

    time: float
    time_index: int


@dataclass
class BuckleyLeverettSaveData(TwoPhaseFlowSaveData):
    l2_error: float


class DiagnosticsMixinExtended(pp.DiagnosticsMixin):
    # MyPy doesn't like that the signature is different.
    def plot_diagnostics(  # type: ignore
        self, diagnostics_data, key: str, filename: str, **kwargs
    ) -> None:
        if _IS_SEABORN_AVAILABLE:
            plt.figure()
            super().plot_diagnostics(diagnostics_data, key)
            plt.savefig(filename)


class DataSavingTwoPhaseFlow(VerificationDataSaving):
    """Model class mixin to save residual and number of Newton iterations for each time
    step."""

    results: list
    """List of objects containing the results of the verification."""
    residuals_wrt_homotopy: list[float]
    residuals_wrt_goal_function: list[float]

    # Ignore signature incompatible with supertype.
    def save_data_time_step(  # type: ignore
        self, solution_norms: list[float], iteration_counter: int
    ) -> None:
        """Save data to the `results` list."""
        collected_data = self.collect_data(solution_norms, iteration_counter)
        self.results.append(collected_data)

    # Ignore signature incompatible with supertype.
    def collect_data(self, solution_norms: list[float], iteration_counter: int):  # type: ignore
        """Collect residual and number of Newton iterations."""
        # Collect residuals w.r.t. to the goal function for models with homotopy
        # continuation.
        if hasattr(self, "residuals_wrt_goal_function"):
            residuals_wrt_homotopy: list[float] = self.residuals_wrt_homotopy
            residuals_wrt_goal_function: list[float] = self.residuals_wrt_goal_function
        else:
            residuals_wrt_homotopy = [0] * iteration_counter
            residuals_wrt_goal_function = [0] * iteration_counter
        return TwoPhaseFlowSaveData(
            solution_norms=solution_norms,
            residuals_wrt_goal_function=residuals_wrt_goal_function,
            residuals_wrt_homotopy=residuals_wrt_homotopy,
            iteration_counter=iteration_counter,
            time=self.time_manager.time,
            time_index=self.time_manager.time_index,
        )


class BuckleyLeverettDataSaving(DataSavingTwoPhaseFlow):
    """Model class mixin to save L2-error, residual and number of Newton iterations each
    time step.

    Requires an analytical solution mixin with the method ``true_solution``.

    """

    saturation_var: str
    """Saturation variable. Provided by ``BuckleyLeverett``."""
    _exact_solution: np.ndarray
    """Exact solution array. Provided by a mixin of type
    ``BuckleyLeverettSemiAnalyticalSolution``."""
    relative_l2_error: Callable
    """Relative l2 error function. Provided by ``porepy.applications.building_blocks.
    verification_utils.VerificationUtils``."""
    domain: pp.Domain

    # Ignore signature incompatible with supertype.
    def collect_data(self, solution_norms: list[float], iteration_counter: int):  # type: ignore
        """Collect L2-error, residual and number of Newton iterations."""
        true_solution = self._exact_solution
        approx_solution = self.equation_system.get_variable_values(
            [self.saturation_var], time_step_index=0
        )
        # Calculate l2_error only if the last time step was reached.
        if np.isclose(self.time_manager.time, self.time_manager.time_final):
            # Map solution
            adjusted_cell_centers = (
                self.mdg.subdomains()[0].cell_centers[0, :]
                + self.domain.bounding_box["xmin"]
            )
            fig = plt.figure()
            plt.plot(
                adjusted_cell_centers, true_solution, label="semi-analytical solution"
            )
            plt.plot(adjusted_cell_centers, approx_solution, label="numerical solution")
            plt.title("Compare solutions at t=1")
            plt.xlabel("x")
            plt.ylabel("S_w")
            plt.legend()
            fig.subplots_adjust(left=0.2, bottom=0.2)
            plt.savefig(
                os.path.join(
                    self.params["folder_name"],
                    self.params["file_name"] + "_compare_solutions.png",
                )
            )
            # Calculate error
            l2_error = float(
                self.relative_l2_error(
                    self.mdg.subdomains()[0],
                    true_solution,
                    approx_solution,
                    is_scalar=True,
                    is_cc=True,
                )
            )
        else:
            l2_error = 0.0
        # Collect residuals w.r.t. to the goal function for models with homotopy
        # continuation.
        if hasattr(self, "residuals_wrt_goal_function"):
            residuals_wrt_homotopy: list[float] = self.residuals_wrt_homotopy
            residuals_wrt_goal_function: list[float] = self.residuals_wrt_goal_function
        else:
            residuals_wrt_homotopy = [0] * iteration_counter
            residuals_wrt_goal_function = [0] * iteration_counter
        return BuckleyLeverettSaveData(
            solution_norms=solution_norms,
            residuals_wrt_goal_function=residuals_wrt_goal_function,
            residuals_wrt_homotopy=residuals_wrt_homotopy,
            l2_error=l2_error,
            iteration_counter=iteration_counter,
            time=self.time_manager.time,
            time_index=self.time_manager.time_index,
        )
