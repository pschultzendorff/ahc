import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import cast

import numpy as np
import porepy as pp

from ahc.models.protocol import TPFProtocol


# No attributes are added dynamically -> use slots=True for memory efficiency.
@dataclass(slots=True)
class SolutionVals:
    pressure: np.ndarray = field(default_factory=lambda: np.array([]))
    saturation: np.ndarray = field(default_factory=lambda: np.array([]))
    total_flux: np.ndarray = field(default_factory=lambda: np.array([]))
    wetting_flux: np.ndarray = field(default_factory=lambda: np.array([]))
    flow_residual: np.ndarray = field(default_factory=lambda: np.array([]))
    transport_residual: np.ndarray = field(default_factory=lambda: np.array([]))


# No attributes are added dynamically -> use slots=True for memory efficiency.
@dataclass(slots=True)
class ComparisonStats:
    """Norms of differences between approximation solution and reference solution of
    different quantities.

    Note: The default value of -1.0 is used to indicate that the comparison statistics
        have not been calculated yet. While min and max values might be negative, the
        norms of true differences are always non-negative.

    """

    pressure_diff_norm: float = -1.0
    pressure_diff_max: float = -1.0
    pressure_diff_min: float = -1.0

    saturation_diff_norm: float = -1.0
    saturation_diff_max: float = -1.0
    saturation_diff_min: float = -1.0

    total_flux_diff_norm: float = -1.0
    total_flux_diff_max: float = -1.0
    total_flux_diff_min: float = -1.0

    wetting_flux_diff_norm: float = -1.0
    wetting_flux_diff_max: float = -1.0
    wetting_flux_diff_min: float = -1.0

    flow_residual_norm: float = -1.0
    transport_residual_norm: float = -1.0


def _difference_stats(
    current: np.ndarray, reference: np.ndarray
) -> tuple[float, float, float]:
    diff = current - reference
    return (
        np.linalg.norm(diff).item(),
        np.max(diff).item(),
        np.min(diff).item(),
    )


class ComparisonMixin(TPFProtocol):
    """_summary_"""

    def save_solution(self, time_step_index: int = 0) -> None:
        """Save the current variable values as numpy arrays."""
        solution = self.equation_system.get_variable_values(
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            time_step_index=time_step_index,
        )
        np.save(
            self.params["folder_name"] / f"solution_{time_step_index}.npy", solution
        )

    def compare_with_reference(
        self,
        current_reference_solution: np.ndarray,
        previous_reference_solution: np.ndarray,
        previous_dt: float,
        hc_parameter: float = 0.0,
    ) -> tuple[ComparisonStats, ComparisonStats]:
        """Compare current approximation with a reference solution and return statistics
        about the differences.

        Note: This method is supposed to be called after convergence of a time step and
            after
            :meth:`~porepy.models.solution_strategy.SolutionStrategy.after_nonlinear_convergence`
            has been called. Calling it at any other time may lead to incorrect results.

        Parameters:
            current_reference_solution: Reference saturation and pressure values (in
                that order) for the same problem.
            previous_reference_solution: Reference saturation and pressure values (in
                that order) for the previous time step.
            previous_dt: Time step size of the solved time step. Assumed to be equal for
                the current and reference solution. Required for the transport residual
                to be calculated correctly.
            hc_parameter: Homotopy continuation parameter at which the
                reference solution was computed. The fluxes and residuals of the current
                and reference solutions are computed at this value. Defaults to 0.0,
                i.e., the target problem.

        Returns:
            A tuple of two ComparisonStats objects, representing the absolute and
            relative differences between the current and reference solutions.

        """
        # Adjust the model state to correctly evaluate the residual of the solved time
        # step.

        # This method is called after after_nonlinear_convergence. The current
        # converged time step solution is stored at time_step_index=0 and equal to the
        # solution at iterate_index=0. The previous time step solution is stored at
        # time_step_index=1. pp.ad.time_derivatives.dt computes the difference between
        # iterate_index=0 and time_step_index=0. Therefore we shift as follows:
        # iterate_index=0 -> iterate_index=0
        # time_step_index=1 -> time_step_index=0
        current_solution = self.equation_system.get_variable_values(
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )
        previous_solution = self.equation_system.get_variable_values(
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            time_step_index=1,
        )
        saved_hc_parameter = (
            # nonlinear_solver_statistics will be an instance of HCSolverStatistics and
            # have the attribute hc_lambda_fl. Ignore mypy.
            self.nonlinear_solver_statistics.hc_lambda_fl if self.uses_hc else 0.0  # type: ignore
        )
        saved_dt = self.time_manager.dt

        self._set_model_state(
            current_solution,
            previous_solution,
            dt=previous_dt,
            hc_parameter=hc_parameter,
        )
        solution_stats = self.collect_solution_values()

        self._set_model_state(
            current_reference_solution,
            previous_reference_solution,
            dt=previous_dt,
            hc_parameter=hc_parameter,
        )
        reference_stats = self.collect_solution_values()

        # Restore the saved model state.
        # iterate_index=0 -> iterate_index=0
        # iterate_index=0 -> time_step_index=0
        self._set_model_state(
            current_solution,
            current_solution,
            dt=saved_dt,
            hc_parameter=saved_hc_parameter,
        )

        pressure_diff_norm, pressure_diff_max, pressure_diff_min = _difference_stats(
            solution_stats.pressure, reference_stats.pressure
        )
        saturation_diff_norm, saturation_diff_max, saturation_diff_min = (
            _difference_stats(solution_stats.saturation, reference_stats.saturation)
        )
        total_flux_diff_norm, total_flux_diff_max, total_flux_diff_min = (
            _difference_stats(solution_stats.total_flux, reference_stats.total_flux)
        )
        wetting_flux_diff_norm, wetting_flux_diff_max, wetting_flux_diff_min = (
            _difference_stats(solution_stats.wetting_flux, reference_stats.wetting_flux)
        )

        absolute_stats = ComparisonStats(
            pressure_diff_norm=pressure_diff_norm,
            pressure_diff_max=pressure_diff_max,
            pressure_diff_min=pressure_diff_min,
            saturation_diff_norm=saturation_diff_norm,
            saturation_diff_max=saturation_diff_max,
            saturation_diff_min=saturation_diff_min,
            total_flux_diff_norm=total_flux_diff_norm,
            total_flux_diff_max=total_flux_diff_max,
            total_flux_diff_min=total_flux_diff_min,
            wetting_flux_diff_norm=wetting_flux_diff_norm,
            wetting_flux_diff_max=wetting_flux_diff_max,
            wetting_flux_diff_min=wetting_flux_diff_min,
            flow_residual_norm=np.linalg.norm(solution_stats.flow_residual).item(),
            transport_residual_norm=np.linalg.norm(
                solution_stats.transport_residual
            ).item(),
        )

        pressure_norm = np.linalg.norm(reference_stats.pressure).item()
        saturation_norm = np.linalg.norm(reference_stats.saturation).item()
        total_flux_norm = np.linalg.norm(reference_stats.total_flux).item()
        wetting_flux_norm = np.linalg.norm(reference_stats.wetting_flux).item()

        relative_stats = ComparisonStats(
            pressure_diff_norm=pressure_diff_norm / pressure_norm,
            pressure_diff_max=pressure_diff_max / pressure_norm,
            pressure_diff_min=pressure_diff_min / pressure_norm,
            saturation_diff_norm=saturation_diff_norm / saturation_norm,
            saturation_diff_max=saturation_diff_max / saturation_norm,
            saturation_diff_min=saturation_diff_min / saturation_norm,
            total_flux_diff_norm=total_flux_diff_norm / total_flux_norm,
            total_flux_diff_max=total_flux_diff_max / total_flux_norm,
            total_flux_diff_min=total_flux_diff_min / total_flux_norm,
            wetting_flux_diff_norm=wetting_flux_diff_norm / wetting_flux_norm,
            wetting_flux_diff_max=wetting_flux_diff_max / wetting_flux_norm,
            wetting_flux_diff_min=wetting_flux_diff_min / wetting_flux_norm,
            flow_residual_norm=np.linalg.norm(solution_stats.flow_residual).item()
            / np.linalg.norm(reference_stats.flow_residual).item(),
            transport_residual_norm=np.linalg.norm(
                solution_stats.transport_residual
            ).item()
            / np.linalg.norm(reference_stats.transport_residual).item(),
        )

        return absolute_stats, relative_stats

    def _set_model_state(
        self,
        current_solution: np.ndarray,
        previous_solution: np.ndarray,
        dt: float,
        hc_parameter: float = 0.0,
    ) -> None:
        """Set to a given model state and rediscretize all equations.

        Parameters:
            current_solution: Saturation and pressure values (in that order) for the
                current approximation.
            previous_solution: Saturation and pressure values (in that order)
                for the previous time step.
            dt: Time step size.
            hc_parameter: Homotopy continuation parameter. Defaults to 0.0, i.e., the
            target problem.

        """
        self.equation_system.set_variable_values(
            values=current_solution,
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )
        self.equation_system.set_variable_values(
            previous_solution,
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            time_step_index=0,
            additive=False,
        )

        self.time_manager.dt = dt
        self.ad_time_step.set_value(dt)

        if self.uses_hc:
            # nonlinear_solver_statistics will be an instance of HCSolverStatistics and
            # have the attributes hc_lambda_fl and hc_lambda_ad. Ignore mypy.
            self.nonlinear_solver_statistics.hc_lambda_fl = hc_parameter  # type: ignore
            self.nonlinear_solver_statistics.hc_lambda_ad.set_value(  # type: ignore
                hc_parameter
            )

        self.eval_secondary_variables()  # type: ignore[attr-defined]
        self.set_discretization_parameters()  # type: ignore[attr-defined]
        self.rediscretize()  # type: ignore[attr-defined]

    def collect_solution_values(self) -> SolutionVals:
        g: pp.Grid = self.g
        es: pp.EquationSystem = self.equation_system

        primary_variables = es.get_variable_values(
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )
        saturation, pressure = (
            primary_variables[: g.num_cells],
            primary_variables[g.num_cells :],
        )

        # The values of the fluxes and equations are always np.ndarrays for the
        # TwoPhaseFlow model. Cast them to np.ndarray to satisfy mypy.
        total_flux = cast(np.ndarray, self.total_flux(g).value(es))
        wetting_flux = cast(np.ndarray, self.wetting_flux(g).value(es))

        flow_residual = cast(
            np.ndarray, self.equation_system.equations[self.flow_equation].value(es)
        )
        transport_residual = cast(
            np.ndarray,
            self.equation_system.equations[self.transport_equation].value(es),
        )

        return SolutionVals(
            pressure=pressure,
            saturation=saturation,
            total_flux=total_flux,
            wetting_flux=wetting_flux,
            flow_residual=flow_residual,
            transport_residual=transport_residual,
        )


def save_comparison_stats(stats: ComparisonStats, filename: pathlib.Path) -> None:
    with filename.open("w") as f:
        json.dump(asdict(stats), f)
