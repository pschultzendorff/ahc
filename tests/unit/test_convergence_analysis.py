import pytest
from tpf_lab.applications.convergence_analysis import ConvergenceAnalysisExtended
import numpy as np


@pytest.fixture
def analysis() -> ConvergenceAnalysisExtended:
    return ConvergenceAnalysisExtended


def test_average_number_of_iterations_single_level() -> None:
    list_of_results = [
        {
            "results": [
                {"iteration_counter": 10},
                {"iteration_counter": 15},
                {"iteration_counter": 1},
            ],
            "cell_diameter": 0.1,
            "dt": 0.01,
        }
    ]

    avg_iters, spatial_refinement, temporal_refinement = average_number_of_iterations(
        list_of_results
    )
    assert np.array_equal(avg_iters, np.array([26 / 3]))
    assert np.array_equal(spatial_refinement, np.array([0.1]))
    assert np.array_equal(temporal_refinement, np.array([0.01]))


def test_average_number_of_iterations_multiple_levels() -> None:
    list_of_results = [
        {
            "results": [
                {"iteration_counter": 10},
                {"iteration_counter": 15},
                {"iteration_counter": 1},
            ],
            "cell_diameter": 0.1,
            "dt": 0.01,
        },
        {
            "results": [
                {"iteration_counter": 5},
                {"iteration_counter": 10},
                {"iteration_counter": 3},
            ],
            "cell_diameter": 0.05,
            "dt": 0.005,
        },
    ]

    avg_iters, spatial_refinement, temporal_refinement = average_number_of_iterations(
        list_of_results
    )
    assert np.array_equal(avg_iters, np.array([26 / 3, 6.0]))
    assert np.array_equal(spatial_refinement, np.array([0.1, 0.05]))
    assert np.array_equal(temporal_refinement, np.array([0.01, 0.005]))


def test_average_number_of_iterations_empty_list() -> None:
    list_of_results = []
    avg_iters, spatial_refinement, temporal_refinement = average_number_of_iterations(
        list_of_results
    )
    assert np.array_equal(avg_iters, np.array([]))
    assert np.array_equal(spatial_refinement, np.array([]))
    assert np.array_equal(temporal_refinement, np.array([]))


def test_average_number_of_iterations_empty_results() -> None:
    list_of_results = [{"results": [], "cell_diameter": 0.1, "dt": 0.01}]
    avg_iters, spatial_refinement, temporal_refinement = average_number_of_iterations(
        list_of_results
    )
    # Expected output
    assert np.array_equal(avg_iters, np.array([0]))
    assert np.array_equal(spatial_refinement, np.array([0.2]))
    assert np.array_equal(temporal_refinement, np.array([0.02]))
