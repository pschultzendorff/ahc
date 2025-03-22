from typing import Any

import numpy as np
import porepy as pp
import pytest


def test_adaptive_lambda_(
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
