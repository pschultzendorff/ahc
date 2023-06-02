import numpy as np
import porepy as pp
import pytest
from typing import Any
from tpf_lab.applications.convergence_analysis import ConvergenceAnalysisExtended
from tpf_lab.models.two_phase_flow import TwoPhaseFlowSetup


@pytest.fixture
def analysis() -> ConvergenceAnalysisExtended:
    return ConvergenceAnalysisExtended(
        TwoPhaseFlowSetup,
        {
            "folder_name": "folder_name",
            "file_name": "file_name",
            "meshing_arguments": {"cell_size": 0.1},
            "time_manager": pp.TimeManager(
                schedule=[0, 1], dt_init=0.1, constant_dt=True
            ),
        },
    )


@pytest.fixture
def list_of_results_single_level() -> list[dict[str, Any]]:
    return [
        {
            "results": [
                {"iteration_counter": 10, "time": 0.0},
                {"iteration_counter": 15, "time": 0.5},
                {"iteration_counter": 1, "time": 1.0, "l2_error": 5.9},
            ],
            "cell_diameter": 0.1,
            "dt": 0.01,
        }
    ]


@pytest.fixture
def list_of_results_multiple_levels() -> list[dict[str, Any]]:
    return [
        {
            "results": [
                {"iteration_counter": 10, "time": 0.0},
                {"iteration_counter": 15, "time": 0.5},
                {"iteration_counter": 1, "time": 1.0, "l2_error": 5.9},
            ],
            "cell_diameter": 0.1,
            "dt": 0.01,
        },
        {
            "results": [
                {"iteration_counter": 5, "time": 0.0},
                {"iteration_counter": 10, "time": 0.5},
                {"iteration_counter": 3, "time": 1.0, "l2_error": 3.1},
            ],
            "cell_diameter": 0.05,
            "dt": 0.005,
        },
    ]


def test_average_number_of_iterations_single_level(
    analysis: ConvergenceAnalysisExtended,
    list_of_results_single_level: list[dict[str, Any]],
) -> None:
    """Test ``average_number_of_iterations`` on a single refinement level."""
    (
        avg_iters,
        spatial_refinement,
        temporal_refinement,
    ) = analysis.average_number_of_iterations(list_of_results_single_level)
    assert np.array_equal(avg_iters, np.array([26 / 3]))
    assert np.array_equal(spatial_refinement, np.array([0.1]))
    assert np.array_equal(temporal_refinement, np.array([0.01]))


def test_average_number_of_iterations_multiple_levels(
    analysis: ConvergenceAnalysisExtended,
    list_of_results_multiple_levels: list[dict[str, Any]],
) -> None:
    """Test ``average_number_of_iterations`` on multiple refinement levels."""
    (
        avg_iters,
        spatial_refinement,
        temporal_refinement,
    ) = analysis.average_number_of_iterations(list_of_results_multiple_levels)
    assert np.array_equal(avg_iters, np.array([26 / 3, 6.0]))
    assert np.array_equal(spatial_refinement, np.array([0.1, 0.05]))
    assert np.array_equal(temporal_refinement, np.array([0.01, 0.005]))


def test_average_number_of_iterations_empty_list(
    analysis: ConvergenceAnalysisExtended,
) -> None:
    """Test ``average_number_of_iterations`` with an empty level list."""
    list_of_results: list[dict[str, Any]] = []
    (
        avg_iters,
        spatial_refinement,
        temporal_refinement,
    ) = analysis.average_number_of_iterations(list_of_results)
    assert np.array_equal(avg_iters, np.array([]))
    assert np.array_equal(spatial_refinement, np.array([]))
    assert np.array_equal(temporal_refinement, np.array([]))


def test_average_number_of_iterations_empty_results(
    analysis: ConvergenceAnalysisExtended,
) -> None:
    """Test ``average_number_of_iterations`` with empty results inside the level
    list."""
    list_of_results = [{"results": [], "cell_diameter": 0.1, "dt": 0.01}]
    (
        avg_iters,
        spatial_refinement,
        temporal_refinement,
    ) = analysis.average_number_of_iterations(list_of_results)
    # Expected output
    assert np.array_equal(avg_iters, np.array([0]))
    assert np.array_equal(spatial_refinement, np.array([0.2]))
    assert np.array_equal(temporal_refinement, np.array([0.02]))


def test_final_l2_error_single_level(
    analysis: ConvergenceAnalysisExtended,
    list_of_results_single_level: list[dict[str, Any]],
) -> None:
    """Test ``final_l2_error`` on a single refinement level."""
    (
        l2_errors,
        spatial_refinement,
        temporal_refinement,
    ) = analysis.final_l2_error(list_of_results_single_level)
    assert np.array_equal(l2_errors, np.array([5.9]))
    assert np.array_equal(spatial_refinement, np.array([0.1]))
    assert np.array_equal(temporal_refinement, np.array([0.01]))


def test_final_l2_error_multiple_levels(
    analysis: ConvergenceAnalysisExtended,
    list_of_results_multiple_levels: list[dict[str, Any]],
) -> None:
    """Test ``final_l2_error`` on multiple refinement levels."""
    (
        l2_errors,
        spatial_refinement,
        temporal_refinement,
    ) = analysis.final_l2_error(list_of_results_multiple_levels)
    assert np.array_equal(l2_errors, np.array([5.9, 3.1]))
    assert np.array_equal(spatial_refinement, np.array([0.1, 0.05]))
    assert np.array_equal(temporal_refinement, np.array([0.01, 0.005]))
