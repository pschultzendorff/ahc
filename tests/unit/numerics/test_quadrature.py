import itertools
import math
import os

import numpy as np
import porepy as pp
import pytest
from ahc.numerics.quadrature import (
    GaussLegendreQuadrature1D,
    GaussLegendreQuadrature2D,
    GaussLegendreQuadrature3D,
    TrapezoidRule,
    TriangleQuadrature,
    get_quadpy_elements,
)

# TODO test_transform does not have to run for every degree. test_reference_points and
# test_weights does not have to run for every domain, grid size, and degree.


@pytest.fixture(scope="session")
def test_run_tracker() -> dict:
    """Tracks whether a test was already run in a session or not."""
    return {}


@pytest.fixture(scope="class")
def domain(request: pytest.FixtureRequest) -> np.ndarray:
    """Create a line/square/cube domain of given size and dimension.

    Parameters:
        request: _description_

    Returns:
        _description_

    """
    size, dim = request.param
    # The PorePy gridding functions we use in ``pp_grid`` require only the upper bound
    # of the unit square/cube. However, the 1D grid requires both, so we pass both.
    return np.repeat(np.array([0, size])[None, ...], dim, axis=0)


@pytest.fixture(scope="class")
def grid_params(request) -> tuple[int, str]:
    """Shared params for :meth:`pp_grid` and :meth:`np_grid`."""
    return request.param


@pytest.fixture(scope="class")
def pp_grid(domain: np.ndarray, grid_params: tuple[int, str]) -> pp.Grid | None:
    """Create grid with given number of elements on ``domain``.

    Parameters:
        domain: _description_
        grid_params: _description_

    Returns:
        grid: A PorePy grid.

    """
    dim: int = domain.shape[0]
    num_elements, grid_type = grid_params
    if dim == 1:
        # Handled by ``np_grid``.
        return None
    elif dim >= 2:
        if grid_type == "cartesian":
            grid: pp.Grid = pp.CartGrid(np.array([num_elements] * dim), domain[..., 1])
        elif grid_type == "simplex":
            if dim == 2:
                grid = pp.StructuredTriangleGrid(
                    np.array([num_elements] * dim), domain[..., 1]
                )
            elif dim == 3:
                grid = pp.StructuredTetrahedralGrid(
                    np.array([num_elements] * dim), domain[..., 1]
                )
    return grid


@pytest.fixture(scope="class")
def np_grid(
    domain: np.ndarray, pp_grid: pp.Grid, grid_params: tuple[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Create given number of elements on ``domain``. Return elements in ``quadpy``
    format and cell volumes as ``np.ndarray``.

    Parameters:
        domain: _description_
        grid_params: _description_

    Returns:
        elements: Elements in ``quadpy`` format.
        volumes: Cell volumes.

    """
    dim: int = domain.shape[0]
    num_elements, grid_type = grid_params
    if dim == 1:
        points: np.ndarray = np.linspace(domain[0, 0], domain[0, 1], num_elements + 1)
        elements: np.ndarray = np.stack((points[:-1], points[1:]), 0)[..., None]
        volumes: np.ndarray = elements[1, ...] - elements[0, ...]
    elif dim >= 2:
        # For 2D and 3D grids, we use Porepy's grid generation. This makes both grid
        # generation and computation of the cell volumes easy.

        # Extract grid elements in quadpy shape and volumes calculated by PorePy.
        # NOTE ``get_quadpy_elements`` is not implemented yet for 3D grids. The
        # corresponding tests are skipped.
        elements = get_quadpy_elements(pp_grid, grid_type=grid_type)
        pp_grid.compute_geometry()
        volumes = pp_grid.cell_volumes
    return elements, volumes.flatten()


def sin_x(points: np.ndarray) -> np.ndarray:
    r"""Compute :math:`f(x) = \sin (x)`.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    return np.sin(points)


@pytest.fixture
def sin_x_exact(domain: np.ndarray) -> float:
    r"""Exact integral of :math:`f(x) = \sin (x)` on an interval.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    return -1 * math.cos(domain[0, -1]) - (-1) * math.cos(domain[0, 0])


def x_times_y(points: np.ndarray) -> np.ndarray:
    """Compute :math:`f(x,y) = xy`.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    assert points.shape[2] == 2, "Dimension of ``points`` must be two-dimensional."
    return points[..., 0] * points[..., 1]


@pytest.fixture
def x_times_y_exact(domain: np.ndarray) -> float:
    """Exact integral of :math:`f(x,y) = xy` on a rectangular domain.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    return (
        (domain[0, 1] ** 2 - domain[0, 0] ** 2)
        * (domain[1, 1] ** 2 - domain[1, 0] ** 2)
        / 4
    )


@pytest.fixture
# Ignore mypy error while function is not fixed.
def x_times_y_exact_triangle(domain: np.ndarray) -> float:  # type: ignore
    """Exact integral of :math:`f(x,y) = xy` on a triangular domain.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    pass
    # TODO Calculate this again.
    # return (
    #     (domain[0, 1] ** 2 - domain[0, 0] ** 2)
    #     * (domain[1, 1] ** 2 - domain[1, 0] ** 2)
    #     / 4
    # )


@pytest.fixture
def x_times_y_times_z(points: np.ndarray) -> np.ndarray:
    """Compute :math:`f(x,y,z) = xyz`.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    assert points.shape[2] == 3, "Dimension of ``points`` must be three-dimensional."
    return points[..., 0] * points[..., 1] * points[..., 2]


@pytest.fixture
def x_times_y_times_z_exact(domain: np.ndarray) -> float:
    """Exact integral of :math:`f(x,y,z) = xyz` on a cubilateral domain.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    return (
        (domain[0, 1] ** 2 - domain[0, 0] ** 2)
        * (domain[1, 1] ** 2 - domain[1, 0] ** 2)
        * (domain[2, 1] ** 2 - domain[2, 0] ** 2)
        / 8
    )


@pytest.mark.parametrize(
    "domain", [(1, 1), (5, 1), (18.9, 1), (51.234, 1)], indirect=True
)
# Pass an empty string for grid type, as this does not matter on a 1D domain.
@pytest.mark.parametrize(
    "grid_params",
    list(itertools.product([30, 100, 500, 800], [""])),
    indirect=True,
)
class TestTrapezoidRule:
    @pytest.fixture(scope="class")
    def quad(self, np_grid: tuple[np.ndarray, np.ndarray]) -> TrapezoidRule:
        quad_instance = TrapezoidRule()
        # Precalulcate integration points and volumes to save some computation time when
        # reused for different tests.
        elements, _ = np_grid
        quad_instance.points = quad_instance.transform(elements)
        quad_instance.volumes = quad_instance.calc_volumes(elements)
        return quad_instance

    def test_integrate(
        self,
        quad: TrapezoidRule,
        sin_x_exact: float,
        domain: np.ndarray,
        np_grid: tuple[np.ndarray, np.ndarray],
    ) -> None:
        if (domain[0, 1] == 51.234 and np_grid[0].shape[1] in [30, 100]) or (
            domain[0, 1] == 5 and np_grid[0].shape[1] == 30
        ):
            pytest.skip("Too few grid points to approximate the integral.")

        integral = quad.integrate(sin_x, recalc_points=False, recalc_volumes=False)
        assert pytest.approx(integral.sum(), rel=1e-3, abs=1e-3) == sin_x_exact

    def test_transform(
        self,
        quad: TrapezoidRule,
        np_grid: tuple[np.ndarray, np.ndarray],
    ) -> None:
        elements, _ = np_grid
        integration_points: np.ndarray = quad.transform(elements)
        assert pytest.approx(integration_points) == elements

    def test_volume(
        self, quad: TrapezoidRule, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        _, volumes = np_grid
        assert pytest.approx(quad.volumes) == volumes

    def test_reference_points(
        self,
        quad: TrapezoidRule,
        test_run_tracker: dict,
    ) -> None:
        # Run the test only for one parametrization.
        test_name: str = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]  # type: ignore
        if test_run_tracker.get(test_name, False):
            pytest.skip(f"{test_name} ran already with a different parametrization.")
        test_run_tracker[test_name] = True

        assert pytest.approx(quad.reference_points) == np.array([[0], [1]])

    def test_weights(
        self,
        quad: TrapezoidRule,
        test_run_tracker: dict,
    ) -> None:
        # Run the test only for one parametrization.
        test_name: str = os.environ.get("PYTEST_CURRENT_TEST").split(" ")[0]  # type: ignore
        if test_run_tracker.get(test_name, False):
            pytest.skip(f"{test_name} ran already with a different parametrization.")
        test_run_tracker[test_name] = True

        assert pytest.approx(quad.weights) == np.array([0.5, 0.5])


@pytest.mark.parametrize("degree", [1, 2, 3, 4], scope="class")
@pytest.mark.parametrize(
    "domain", [(1, 2), (5, 2), (18.9, 2), (51.234, 2)], indirect=True
)
@pytest.mark.parametrize(
    "grid_params",
    list(itertools.product([30, 75, 140], ["simplex"])),
    indirect=True,
    scope="class",
)
class TestTriangleQuadrature:
    @pytest.fixture(scope="class")
    def quad(
        self, degree: int, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> TriangleQuadrature:
        quad_instance = TriangleQuadrature(degree=degree)
        # Precalulcate integration points and volumes to save some computation time when
        # reused for different tests.
        elements, _ = np_grid
        quad_instance.points = quad_instance.transform(elements)
        quad_instance.volumes = quad_instance.calc_volumes(elements)
        return quad_instance

    def test_integrate(
        self,
        quad: TriangleQuadrature,
        degree: int,
        x_times_y_exact: float,
    ) -> None:
        integral = quad.integrate(x_times_y, recalc_points=False, recalc_volumes=False)
        if degree in [1, 2]:
            # The tolerances have to be quite high for degree 1 and 2.
            assert pytest.approx(integral.sum(), rel=1e-1, abs=1e-1) == x_times_y_exact
        else:
            assert pytest.approx(integral.sum()) == x_times_y_exact

    # TODO Implement tests for different reference elements.
    @pytest.mark.skip("Test not implemented.")
    def test_transform(
        self, quad: TriangleQuadrature, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pass

    # TODO Implement tests for different reference elements.
    @pytest.mark.skip("Test not implemented.")
    def test_get_affine_coefficients(
        self,
        quad: TriangleQuadrature,
        expected: np.ndarray,
    ) -> None:
        pass

    def test_volume(
        self, quad: TriangleQuadrature, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        _, volumes = np_grid
        assert pytest.approx(quad.volumes) == volumes

    def test_reference_points(
        self,
        quad: TriangleQuadrature,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        # Run the test only for degree 1.
        if degree != 1:
            pytest.skip("Reference points are tested only for degree 1.")
        assert pytest.approx(quad.reference_points) == np.array([[1 / 3, 1 / 3]])

    def test_weights(
        self,
        quad: TriangleQuadrature,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        # Run the test only for degree 1.
        if degree != 1:
            pytest.skip("Weights are tested only for degree 1.")
        assert pytest.approx(quad.weights) == np.array([1 / 2])


@pytest.mark.parametrize("degree", [2, 3, 4, 5], scope="class")
@pytest.mark.parametrize(
    "domain", [(1, 1), (5, 1), (18.9, 1), (51.234, 1)], indirect=True
)
@pytest.mark.parametrize(
    "grid_params",
    list(itertools.product([30, 100, 500, 800], ["cartesian"])),
    indirect=True,
)
class TestGaussLegendreQuadrature1D:
    @pytest.fixture(scope="class")
    def quad(
        self, degree: int, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> GaussLegendreQuadrature1D:
        quad_instance = GaussLegendreQuadrature1D(degree=degree)
        # Precalulcate integration points and volumes to save some computation time when
        # reused for different tests.
        elements, _ = np_grid
        quad_instance.points = quad_instance.transform(elements)
        quad_instance.volumes = quad_instance.calc_volumes(elements)
        return quad_instance

    def test_integrate(
        self,
        quad: GaussLegendreQuadrature1D,
        sin_x_exact: float,
        np_grid: tuple[np.ndarray, np.ndarray],
        domain: np.ndarray,
        degree: int,
    ) -> None:
        # Skip cases with too few grid points to approximate the integral.
        if (
            (domain[0, 1] == 51.234 and np_grid[0].shape[1] == 30 and degree in [2, 3])
            or (domain[0, 1] == 18.9 and np_grid[0].shape[1] == 30 and degree == 2)
            or (domain[0, 1] == 51.234 and np_grid[0].shape[1] == 100 and degree == 2)
        ):
            pytest.skip("Too few grid points to approximate the integral.")
        integral = quad.integrate(sin_x, recalc_points=False, recalc_volumes=False)
        assert pytest.approx(integral.sum()) == sin_x_exact

    # TODO Implement tests for different reference elements.
    @pytest.mark.skip("Test not implemented.")
    def test_transform(
        self, quad: GaussLegendreQuadrature1D, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pass

    def test_volume(
        self, quad: GaussLegendreQuadrature1D, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        _, volumes = np_grid
        assert pytest.approx(quad.volumes) == volumes

    def test_reference_points(
        self,
        quad: GaussLegendreQuadrature1D,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        if degree == 2:
            assert pytest.approx(quad.reference_points) == np.array(
                [[-0.5773502691896257], [0.5773502691896257]]
            )
        elif degree == 3:
            assert pytest.approx(quad.reference_points) == np.array(
                [[-0.7745966692414834], [0], [0.7745966692414834]]
            )
        elif degree == 4:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.8611363115940526],
                    [-0.3399810435848563],
                    [0.3399810435848563],
                    [0.8611363115940526],
                ]
            )
        elif degree == 5:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.9061798459386640],
                    [-0.5384693101056831],
                    [0],
                    [0.5384693101056831],
                    [0.9061798459386640],
                ]
            )

    def test_weights(
        self,
        quad: GaussLegendreQuadrature1D,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        if degree == 2:
            assert pytest.approx(quad.weights) == np.array([1, 1])
        elif degree == 3:
            assert pytest.approx(quad.weights) == np.array(
                [0.5555555555555556, 0.8888888888888888, 0.5555555555555556]
            )
        elif degree == 4:
            assert pytest.approx(quad.weights) == np.array(
                [
                    0.3478548451374538,
                    0.6521451548625461,
                    0.6521451548625461,
                    0.3478548451374538,
                ]
            )
        elif degree == 5:
            assert pytest.approx(quad.weights) == np.array(
                [
                    0.2369268850561891,
                    0.4786286704993665,
                    0.5688888888888889,
                    0.4786286704993665,
                    0.2369268850561891,
                ]
            )


@pytest.mark.parametrize("degree", [2, 3, 4, 5], scope="class")
@pytest.mark.parametrize(
    "domain", [(1, 2), (5, 2), (18.9, 2), (51.234, 2)], indirect=True
)
@pytest.mark.parametrize(
    "grid_params",
    list(itertools.product([30, 75, 140], ["cartesian"])),
    indirect=True,
)
class TestGaussLegendreQuadrature2D:
    @pytest.fixture(scope="class")
    def quad(
        self, degree: int, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> GaussLegendreQuadrature2D:
        quad_instance = GaussLegendreQuadrature2D(degree=degree)
        # Precalulcate integration points and volumes to save some computation time when
        # reused for different tests.
        elements, _ = np_grid
        quad_instance.points = quad_instance.transform(elements)
        quad_instance.volumes = quad_instance.calc_volumes(elements)
        return quad_instance

    def test_integrate(
        self,
        quad: GaussLegendreQuadrature2D,
        x_times_y_exact: float,
    ) -> None:
        integral = quad.integrate(x_times_y, recalc_points=False, recalc_volumes=False)
        assert pytest.approx(integral.sum()) == x_times_y_exact

    # TODO Implement tests for different reference elements.
    @pytest.mark.skip("Test not implemented.")
    def test_transform(
        self, quad: GaussLegendreQuadrature2D, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pass

    def test_volume(
        self, quad: GaussLegendreQuadrature2D, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        _, volumes = np_grid
        assert pytest.approx(quad.volumes) == volumes

    def test_reference_points(
        self,
        quad: GaussLegendreQuadrature2D,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        if degree == 2:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.5773502691896257, -0.5773502691896257],
                    [0.5773502691896257, -0.5773502691896257],
                    [-0.5773502691896257, 0.5773502691896257],
                    [0.5773502691896257, 0.5773502691896257],
                ]
            )
        elif degree == 3:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.7745966692414834, -0.7745966692414834],
                    [0, -0.7745966692414834],
                    [0.7745966692414834, -0.7745966692414834],
                    [-0.7745966692414834, 0],
                    [0, 0],
                    [0.7745966692414834, 0],
                    [-0.7745966692414834, 0.7745966692414834],
                    [0, 0.7745966692414834],
                    [0.7745966692414834, 0.7745966692414834],
                ]
            )
        elif degree == 4:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.8611363115940526, -0.8611363115940526],
                    [-0.3399810435848563, -0.8611363115940526],
                    [0.3399810435848563, -0.8611363115940526],
                    [0.8611363115940526, -0.8611363115940526],
                    [-0.8611363115940526, -0.3399810435848563],
                    [-0.3399810435848563, -0.3399810435848563],
                    [0.3399810435848563, -0.3399810435848563],
                    [0.8611363115940526, -0.3399810435848563],
                    [-0.8611363115940526, 0.3399810435848563],
                    [-0.3399810435848563, 0.3399810435848563],
                    [0.3399810435848563, 0.3399810435848563],
                    [0.8611363115940526, 0.3399810435848563],
                    [-0.8611363115940526, 0.8611363115940526],
                    [-0.3399810435848563, 0.8611363115940526],
                    [0.3399810435848563, 0.8611363115940526],
                    [0.8611363115940526, 0.8611363115940526],
                ]
            )
        elif degree == 5:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.9061798459386640, -0.9061798459386640],
                    [-0.5384693101056831, -0.9061798459386640],
                    [0, -0.9061798459386640],
                    [0.5384693101056831, -0.9061798459386640],
                    [0.9061798459386640, -0.9061798459386640],
                    [-0.9061798459386640, -0.5384693101056831],
                    [-0.5384693101056831, -0.5384693101056831],
                    [0, -0.5384693101056831],
                    [0.5384693101056831, -0.5384693101056831],
                    [0.9061798459386640, -0.5384693101056831],
                    [-0.9061798459386640, 0],
                    [-0.5384693101056831, 0],
                    [0, 0],
                    [0.5384693101056831, 0],
                    [0.9061798459386640, 0],
                    [-0.9061798459386640, 0.5384693101056831],
                    [-0.5384693101056831, 0.5384693101056831],
                    [0, 0.5384693101056831],
                    [0.5384693101056831, 0.5384693101056831],
                    [0.9061798459386640, 0.5384693101056831],
                    [-0.9061798459386640, 0.9061798459386640],
                    [-0.5384693101056831, 0.9061798459386640],
                    [0, 0.9061798459386640],
                    [0.5384693101056831, 0.9061798459386640],
                    [0.9061798459386640, 0.9061798459386640],
                ]
            )

    def test_weights(
        self,
        quad: GaussLegendreQuadrature2D,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        if degree == 2:
            assert pytest.approx(quad.weights) == np.array([1, 1, 1, 1])
        elif degree == 3:
            assert pytest.approx(quad.weights) == np.array(
                [
                    0.308641975308642,
                    0.493827160493827,
                    0.308641975308642,
                    0.493827160493827,
                    0.790123456790123,
                    0.493827160493827,
                    0.308641975308642,
                    0.493827160493827,
                    0.308641975308642,
                ]
            )
        elif degree == 4:
            assert pytest.approx(quad.weights) == np.array(
                [
                    0.121002993285602,
                    0.226851851851852,
                    0.226851851851852,
                    0.121002993285602,
                    0.226851851851852,
                    0.425293303010694,
                    0.425293303010694,
                    0.226851851851852,
                    0.226851851851852,
                    0.425293303010694,
                    0.425293303010694,
                    0.226851851851852,
                    0.121002993285602,
                    0.226851851851852,
                    0.226851851851852,
                    0.121002993285602,
                ]
            )
        elif degree == 5:
            assert pytest.approx(quad.weights) == np.array(
                [
                    0.062176616655347,
                    0.134785072387521,
                    0.180381692347541,
                    0.134785072387521,
                    0.062176616655347,
                    0.134785072387521,
                    0.292042683679683,
                    0.390184423106851,
                    0.292042683679683,
                    0.134785072387521,
                    0.180381692347541,
                    0.390184423106851,
                    0.520408163265306,
                    0.390184423106851,
                    0.180381692347541,
                    0.134785072387521,
                    0.292042683679683,
                    0.390184423106851,
                    0.292042683679683,
                    0.134785072387521,
                    0.062176616655347,
                    0.134785072387521,
                    0.180381692347541,
                    0.134785072387521,
                    0.062176616655347,
                ]
            )


@pytest.mark.parametrize("degree", [2, 3], scope="class")
@pytest.mark.parametrize(
    "domain", [(1, 3), (5, 3), (18.9, 3), (51.234, 3)], indirect=True
)
@pytest.mark.parametrize(
    "grid_params", list(itertools.product([30, 75], ["cartesian"])), indirect=True
)
class TestGaussLegendreQuadrature3D:
    @pytest.fixture(scope="class")
    def quad(
        self, degree: int, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> GaussLegendreQuadrature3D:
        quad_instance = GaussLegendreQuadrature3D(degree=degree)
        # Precalulcate integration points and volumes to save some computation time when
        # reused for different tests.
        elements, _ = np_grid
        quad_instance.points = quad_instance.transform(elements)
        quad_instance.volumes = quad_instance.calc_volumes(elements)
        return quad_instance

    @pytest.mark.skip("``get_quadpy_elements`` not implemented for 3D grids.")
    def test_integrate(
        self,
        quad: GaussLegendreQuadrature3D,
        x_times_y_times_z_exact: float,
        np_grid: tuple[np.ndarray, np.ndarray],
    ) -> None:
        integral = quad.integrate(
            x_times_y_times_z, recalc_points=False, recalc_volumes=False
        )
        assert pytest.approx(integral.sum()) == x_times_y_times_z_exact

    # TODO Implement tests for different reference elements.
    @pytest.mark.skip("``get_quadpy_elements`` not implemented for 3D grids.")
    def test_transform(
        self, quad: GaussLegendreQuadrature3D, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        pass

    @pytest.mark.skip("``get_quadpy_elements`` not implemented for 3D grids.")
    def test_volume(
        self, quad: GaussLegendreQuadrature3D, np_grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        _, volumes = np_grid
        assert pytest.approx(quad.volumes) == volumes

    @pytest.mark.skip("``get_quadpy_elements`` not implemented for 3D grids.")
    def test_reference_points(
        self,
        quad: GaussLegendreQuadrature3D,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        if degree == 2:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.5773502691896257, -0.5773502691896257, -0.5773502691896257],
                    [0.5773502691896257, -0.5773502691896257, -0.5773502691896257],
                    [-0.5773502691896257, 0.5773502691896257, -0.5773502691896257],
                    [0.5773502691896257, 0.5773502691896257, -0.5773502691896257],
                    [-0.5773502691896257, -0.5773502691896257, 0.5773502691896257],
                    [0.5773502691896257, -0.5773502691896257, 0.5773502691896257],
                    [-0.5773502691896257, 0.5773502691896257, 0.5773502691896257],
                    [0.5773502691896257, 0.5773502691896257, 0.5773502691896257],
                ]
            )
        elif degree == 3:
            assert pytest.approx(quad.reference_points) == np.array(
                [
                    [-0.7745966692414834, -0.7745966692414834, -0.7745966692414834],
                    [0, -0.7745966692414834, -0.7745966692414834],
                    [0.7745966692414834, -0.7745966692414834, -0.7745966692414834],
                    [-0.7745966692414834, 0, -0.7745966692414834],
                    [0, 0, -0.7745966692414834],
                    [0.7745966692414834, 0, -0.7745966692414834],
                    [-0.7745966692414834, 0.7745966692414834, -0.7745966692414834],
                    [0, 0.7745966692414834, -0.7745966692414834],
                    [0.7745966692414834, 0.7745966692414834, -0.7745966692414834],
                    [-0.7745966692414834, -0.7745966692414834, 0],
                    [0, -0.7745966692414834, 0],
                    [0.7745966692414834, -0.7745966692414834, 0],
                    [-0.7745966692414834, 0, 0],
                    [0, 0, 0],
                    [0.7745966692414834, 0, 0],
                    [-0.7745966692414834, 0.7745966692414834, 0],
                    [0, 0.7745966692414834, 0],
                    [0.7745966692414834, 0.7745966692414834, 0],
                    [-0.7745966692414834, -0.7745966692414834, 0.7745966692414834],
                    [0, -0.7745966692414834, 0.7745966692414834],
                    [0.7745966692414834, -0.7745966692414834, 0.7745966692414834],
                    [-0.7745966692414834, 0, 0.7745966692414834],
                    [0, 0, 0.7745966692414834],
                    [0.7745966692414834, 0, 0.7745966692414834],
                    [-0.7745966692414834, 0.7745966692414834, 0.7745966692414834],
                    [0, 0.7745966692414834, 0.7745966692414834],
                    [0.7745966692414834, 0.7745966692414834, 0.7745966692414834],
                ]
            )

    @pytest.mark.skip("``get_quadpy_elements`` not implemented for 3D grids.")
    def test_weights(
        self,
        quad: GaussLegendreQuadrature3D,
        degree: int,
    ) -> None:
        # TODO Add functionality to run this only for one domain and grid.
        if degree == 2:
            assert pytest.approx(quad.weights) == np.array([1] * 8)
        elif degree == 3:
            assert pytest.approx(quad.weights) == np.array(
                [
                    0.171467764060357,
                    0.274348422496571,
                    0.171467764060357,
                    0.274348422496571,
                    0.438957475994513,
                    0.274348422496571,
                    0.171467764060357,
                    0.274348422496571,
                    0.171467764060357,
                    0.274348422496571,
                    0.438957475994513,
                    0.274348422496571,
                    0.438957475994513,
                    0.702331961591221,
                    0.438957475994513,
                    0.274348422496571,
                    0.438957475994513,
                    0.702331961591221,
                    0.438957475994513,
                    0.274348422496571,
                    0.171467764060357,
                    0.274348422496571,
                    0.171467764060357,
                    0.274348422496571,
                    0.438957475994513,
                    0.274348422496571,
                    0.171467764060357,
                ]
            )


@pytest.mark.parametrize(
    "domain", [(1, 2), (5, 2), (18.9, 2), (51.234, 2)], indirect=True
)
@pytest.mark.parametrize(
    "grid_params",
    list(itertools.product([30, 75, 140], ["simplex", "cartesian"])),
    indirect=True,
    scope="class",
)
def test_get_quadpy_elements(pp_grid: pp.Grid) -> None:
    """Test that quadpy elements have the right shape."""
    grid_type = "cartesian" if "cart" in pp_grid.name.lower() else "simplex"
    quadpy_elements = get_quadpy_elements(pp_grid, grid_type)
    num_nodes_per_cell = 4 if grid_type == "cartesian" else 3
    assert quadpy_elements.shape == (num_nodes_per_cell, pp_grid.num_cells, pp_grid.dim)
