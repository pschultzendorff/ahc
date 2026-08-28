import json
import pathlib
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import porepy as pp
from matplotlib.pylab import cast

from ahc.models.protocol import TPFProtocol


# No attributes are added dynamically -> use slots=True for memory efficiency.
@dataclass(slots=True)
class SolutionVals:
    pressure: np.ndarray
    saturation: np.ndarray
    total_flux: np.ndarray
    wetting_flux: np.ndarray
    flow_residual: np.ndarray
    transport_residual: np.ndarray


# No attributes are added dynamically -> use slots=True for memory efficiency.
@dataclass(slots=True)
class ComparisonStats:
    pressure_diff_norm: float
    pressure_diff_max: float
    pressure_diff_min: float

    saturation_diff_norm: float
    saturation_diff_max: float
    saturation_diff_min: float

    total_flux_diff_norm: float
    total_flux_diff_max: float
    total_flux_diff_min: float

    wetting_flux_diff_norm: float
    wetting_flux_diff_max: float
    wetting_flux_diff_min: float

    flow_residual_norm: float
    transport_residual_norm: float


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

    def save_solution(self) -> None:
        """Save the current variable values as numpy arrays."""
        solution = self.equation_system.get_variable_values(
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )
        np.save(self.params["folder_name"] / "solution.npy", solution)

    def compare_with_reference(self, reference_solution: np.ndarray) -> ComparisonStats:
        """Compare current approximation with a reference solution and return statistics
        about the differences.

        Parameters:
            reference_solution: Reference saturation and pressure values (in that order)
                for the same problem.

        Returns:
            _description_

        """
        current_solution = self.equation_system.get_variable_values(
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )
        # Evaluate current and reference solution statistics
        solution_stats = self.collect_solution_values()
        self.equation_system.set_variable_values(
            values=reference_solution,
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )
        reference_stats = self.collect_solution_values()
        # Restore the current solution values.
        self.equation_system.set_variable_values(
            values=current_solution,
            variables=[self.primary_saturation_var, self.primary_pressure_var],
            iterate_index=0,
        )

        # Calculate differences between current and reference statistics.

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

        return ComparisonStats(
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
