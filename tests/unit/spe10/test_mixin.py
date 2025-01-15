import porepy as pp
import pytest
from tpf.spe10.model import EquationsSPE10


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
    "grid, spe10_instance, expected_corners",
    [
        (pytest.lazy_fixture("cartesian_grid_2d"), [0, 2, 6, 8]),
        (pytest.lazy_fixture("cartesian_grid_3d"), [0, 2, 6, 8, 18, 20, 24, 26]),
    ],
)
def test_corner_cell_ids(grid, expected_corners):
    assert set(EquationsSPE10.corner_cell_ids(grid)) == set(expected_corners)


@pytest.mark.parametrize(
    "grid, spe10_instance, expected_center",
    [
        (pytest.lazy_fixture("cartesian_grid_2d"), [0, 2, 6, 8]),
        (pytest.lazy_fixture("cartesian_grid_3d"), [0, 2, 6, 8, 18, 20, 24, 26]),
    ],
)
def test_center_cell_ids(grid, expected_center):
    assert EquationsSPE10.center_cell_id(grid) == expected_center


if __name__ == "__main__":
    pytest.main([__file__])
