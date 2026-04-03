from typing import Literal

import porepy as pp
import pytest
from ahc.derived_models.utils import position_cell_id


# FIXME Fix all tests in this file.
@pytest.fixture
def cartesian_grid_2d():
    g2d = pp.CartGrid([3, 3])
    g2d.compute_geometry()
    return g2d


@pytest.fixture
def cartesian_grid_3d():
    g3d = pp.CartGrid([3, 3, 3])
    g3d.compute_geometry()
    return g3d


@pytest.mark.parametrize(
    "grid, x, y, expected_corners",
    [
        ("cartesian_grid_2d", [0, 2, 6, 8]),
        ("cartesian_grid_3d", [0, 2, 6, 8, 18, 20, 24, 26]),
    ],
)
def test_cell_id_position(
    grid: Literal["cartesian_grid_2d"] | Literal["cartesian_grid_3d"],
    expected_corners: list[int],
):
    grid = None
    assert set(position_cell_id(grid)) == set(expected_corners)


@pytest.mark.parametrize(
    "grid, expected_center",
    [
        ("cartesian_grid_2d", [0, 2, 6, 8]),
        ("cartesian_grid_3d", [0, 2, 6, 8, 18, 20, 24, 26]),
    ],
)
def test_center_cell_ids(
    grid: Literal["cartesian_grid_2d"] | Literal["cartesian_grid_3d"],
    expected_center: list[int],
):
    assert EquationsSPE10.center_cell_id(grid) == expected_center
