import math

import numpy as np
import porepy as pp
import pytest
from src.quadrature import (
    GaussLegendreQuadrature1D,
    GaussLegendreQuadrature2D,
    GaussLegendreQuadrature3D,
    TrapezoidRule,
    TriangleQuadrature,
)
from src.utils import get_quadpy_elements

# TODO: test_transform does not have to run for every degree. test_points and
# test_weights does not have to run for every domain, grid size, and degree.


@pytest.fixture
def domain(request) -> np.ndarray:
    """Create a line/square/cube domain of given size and dimension.

    Parameters:
        request: _description_

    Returns:
        _description_

    """
    size, dim = request.param
    return np.repeat(np.array([0, size])[None, ...], dim, 0)


@pytest.fixture
def grid(domain: np.ndarray, request) -> tuple[np.ndarray, np.ndarray]:  # type:ignore
    """Create given number of elements on ``domain``. Return elements in ``quadpy``
    format and cell volumes as ``np.ndarray``.

    Parameters:
        domain: _description_
        request: _description_

    Returns:
        elements (np.ndarray): Elements in ``quadpy`` format.
        volumes (np.ndarray): Cell volumes.

    """
    dim: int = domain.shape[0]
    num_elements: int = request.param
    if dim == 1:
        points: np.ndarray = np.linspace(domain[0, 0], domain[0, 1], num_elements + 1)
        elements: np.ndarray = np.stack((points[:-1], points[1:]), 0)[..., None]
        volumes: np.ndarray = elements[1, ...] - elements[0, ...]
    elif dim >= 2:
        grid: pp.Grid = pp.CartGrid(  # type: ignore
            np.array([num_elements] * 2), domain[..., 1]
        )
        elements: np.ndarray = get_quadpy_elements(grid)
        grid.compute_geometry()
        volumes: np.ndarray = grid.cell_volumes
    return elements, volumes


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
    assert points.shape[0] == 2, "Dimension of ``points`` must be two-dimensional."
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
def x_times_y_exact_triangle(domain: np.ndarray) -> float:
    """Exact integral of :math:`f(x,y) = xy` on a triangular domain.

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
def x_times_y_times_z(points: np.ndarray) -> np.ndarray:
    """Compute :math:`f(x,y,z) = xyz`.

    Parameters:
        points: _description_

    Returns:
        _description_

    """
    assert points.shape[0] == 3, "Dimension of ``points`` must be three-dimensional."
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
@pytest.mark.parametrize("grid", [30, 100, 500, 800], indirect=True)
class TestTrapezoidRule:

    @pytest.fixture(scope="class")
    def quad(self) -> TrapezoidRule:
        return TrapezoidRule()

    @pytest.mark.skipif(
        (
            pytest.param("domain[0,1]") == 51.234
            and pytest.param("grid[0].shape[1]") in [30, 100]
        )
        or (
            pytest.param("domain[0, 1]") == 5 and pytest.param("grid[0].shape[1]") == 30
        ),
        reason="Too little grid points to approximate the integral.",
    )
    def test_integrate(
        self,
        quad: TrapezoidRule,
        sin_x_exact: float,
        grid: tuple[np.ndarray, np.ndarray],
        domain: np.ndarray,
    ) -> None:
        elements, _ = grid
        integral = quad.integrate(sin_x, elements)
        assert pytest.approx(integral.sum(), rel=1e-3, abs=1e-3) == sin_x_exact

    def test_transform(
        self, quad: TrapezoidRule, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, _ = grid
        integration_points: np.ndarray = quad.transform(elements, quad.points)
        assert pytest.approx(integration_points) == elements

    def test_volume(
        self, quad: TrapezoidRule, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, volumes = grid
        integration_volumes: np.ndarray = quad.volume_factor(elements)
        assert pytest.approx(integration_volumes) == volumes

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 1) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_points(
        self, quad: TrapezoidRule, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        assert pytest.approx(quad.points) == np.array([0, 1])

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 1) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_weights(
        self, quad: TrapezoidRule, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        assert pytest.approx(quad.weights) == np.array([0.5, 0.5])


@pytest.mark.parametrize(
    "domain", [(1, 2), (5, 2), (18.9, 2), (51.234, 2)], indirect=True
)
@pytest.mark.parametrize("grid", [30, 100, 500, 800], indirect=True)
class TestTriangleQuadrature:

    @pytest.fixture(scope="class")
    def quad(self) -> TriangleQuadrature:
        return TriangleQuadrature()

    def test_integrate(
        self,
        quad: TriangleQuadrature,
        x_times_y_exact: float,
        grid: tuple[np.ndarray, np.ndarray],
    ) -> None:
        elements, _ = grid
        integral = quad.integrate(x_times_y, elements)
        assert pytest.approx(integral.sum()) == x_times_y_exact

    def test_transform(
        self, quad: TriangleQuadrature, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, _ = grid
        integration_points: np.ndarray = quad.transform(quad.points, elements)
        assert pytest.approx(integration_points) == elements

    @pytest.mark.parametrize("expected", [])
    def test_get_affine_coefficients(
        self,
        quad: TriangleQuadrature,
        grid: tuple[np.ndarray, np.ndarray],
        expected: np.ndarray,
    ) -> None:
        elements, _ = grid
        # TODO: Implement tests for different reference elements.
        quad.calc_affine_coefficients(elements)
        assert pytest.approx(quad._affine_coefficients) == expected

    def test_volume(
        self, quad: TriangleQuadrature, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, volumes = grid
        integration_volumes: np.ndarray = quad.volume_factor(elements)
        assert pytest.approx(integration_volumes) == volumes

    def test_points(
        self, quad: TriangleQuadrature, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        assert pytest.approx(quad.points) == np.array([0, 1])

    def test_weights(
        self, quad: TriangleQuadrature, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        assert pytest.approx(quad.weights) == np.array([0.5, 0.5])


@pytest.mark.parametrize("degree", [2, 3, 4, 5], scope="class")
@pytest.mark.parametrize(
    "domain", [(1, 1), (5, 1), (18.9, 1), (51.234, 1)], indirect=True
)
@pytest.mark.parametrize("grid", [30, 100, 500, 800], indirect=True)
class TestGaussLegendreQuadrature1D:

    @pytest.fixture(scope="class")
    def quad(self, degree: int) -> GaussLegendreQuadrature1D:
        return GaussLegendreQuadrature1D(degree=degree)

    def test_integrate(
        self,
        quad: GaussLegendreQuadrature1D,
        sin_x_exact: float,
        grid: tuple[np.ndarray, np.ndarray],
        domain: np.ndarray,
        degree: int,
    ) -> None:
        # Skip cases with too little grid points to approximate the integral.
        if (
            (domain[0, 1] == 51.234 and grid[0].shape[1] == 30 and degree in [2, 3])
            or (domain[0, 1] == 18.9 and grid[0].shape[1] == 30 and degree == 2)
            or (domain[0, 1] == 51.234 and grid[0].shape[1] == 100 and degree == 2)
        ):
            pytest.skip()
        elements, _ = grid
        integral = quad.integrate(sin_x, elements)
        assert pytest.approx(integral.sum()) == sin_x_exact

    def test_transform(
        self, quad: GaussLegendreQuadrature1D, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, _ = grid
        integration_points: np.ndarray = quad.transform(elements, quad.points)
        # TODO: Implement tests for different reference elements.

    def test_volume(
        self, quad: GaussLegendreQuadrature1D, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, volumes = grid
        integration_volumes: np.ndarray = quad.volume_factor(elements)
        assert pytest.approx(integration_volumes) == volumes / 2

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 1) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_points(
        self,
        quad: GaussLegendreQuadrature1D,
        grid: tuple[np.ndarray, np.ndarray],
        degree: int,
    ) -> None:
        if degree == 2:
            assert pytest.approx(quad.points) == np.array(
                [-0.5773502691896257, 0.5773502691896257]
            )
        elif degree == 3:
            assert pytest.approx(quad.points) == np.array(
                [-0.7745966692414834, 0, 0.7745966692414834]
            )
        elif degree == 4:
            assert pytest.approx(quad.points) == np.array(
                [
                    -0.8611363115940526,
                    -0.3399810435848563,
                    0.3399810435848563,
                    0.8611363115940526,
                ]
            )
        elif degree == 5:
            assert pytest.approx(quad.points) == np.array(
                [
                    -0.9061798459386640,
                    -0.5384693101056831,
                    0,
                    0.5384693101056831,
                    0.9061798459386640,
                ]
            )

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 1) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_weights(
        self,
        quad: GaussLegendreQuadrature1D,
        grid: tuple[np.ndarray, np.ndarray],
        degree: int,
    ) -> None:
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
@pytest.mark.parametrize("grid", [30, 100, 500, 800], indirect=True)
class TestGaussLegendreQuadrature2D:

    @pytest.fixture(scope="class")
    def quad(self, degree: int) -> GaussLegendreQuadrature2D:
        return GaussLegendreQuadrature2D(degree=degree)

    def test_integrate(
        self,
        quad: GaussLegendreQuadrature2D,
        x_times_y_exact: float,
        grid: tuple[np.ndarray, np.ndarray],
    ) -> None:
        elements, _ = grid
        integral = quad.integrate(x_times_y, elements)
        assert pytest.approx(integral.sum()) == x_times_y_exact

    def test_transform(
        self, quad: GaussLegendreQuadrature2D, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, _ = grid
        integration_points: np.ndarray = quad.transform(elements, quad.points)
        # TODO: Implement tests for different reference elements.

    def test_volume(
        self, quad: GaussLegendreQuadrature2D, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, volumes = grid
        integration_volumes: np.ndarray = quad.volume_factor(elements)
        assert pytest.approx(integration_volumes) == volumes / 4

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 2) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_points(
        self,
        quad: GaussLegendreQuadrature2D,
        grid: tuple[np.ndarray, np.ndarray],
        degree: int,
    ) -> None:
        if degree == 2:
            assert pytest.approx(quad.points) == np.array(
                [
                    [-0.5773502691896257, -0.5773502691896257],
                    [0.5773502691896257, -0.5773502691896257],
                    [-0.5773502691896257, 0.5773502691896257],
                    [0.5773502691896257, 0.5773502691896257],
                ]
            )
        elif degree == 3:
            assert pytest.approx(quad.points) == np.array(
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
            assert pytest.approx(quad.points) == np.array(
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
            assert pytest.approx(quad.points) == np.array(
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

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 2) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_weights(
        self,
        quad: GaussLegendreQuadrature2D,
        grid: tuple[np.ndarray, np.ndarray],
        degree: int,
    ) -> None:
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


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize(
    "domain", [(1, 3), (5, 3), (18.9, 3), (51.234, 3)], indirect=True
)
@pytest.mark.parametrize("grid", [30, 100, 500, 800], indirect=True)
class TestGaussLegendreQuadrature3D:

    @pytest.fixture(scope="class")
    def quad(self, degree: int) -> GaussLegendreQuadrature3D:
        return GaussLegendreQuadrature3D(degree=degree)

    def test_integrate(
        self,
        quad: GaussLegendreQuadrature3D,
        x_times_y_times_z_exact: float,
        grid: tuple[np.ndarray, np.ndarray],
    ) -> None:
        elements, _ = grid
        integral = quad.integrate(x_times_y_times_z, elements)
        assert pytest.approx(integral.sum()) == x_times_y_times_z_exact

    def test_transform(
        self, quad: GaussLegendreQuadrature3D, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, _ = grid
        integration_points: np.ndarray = quad.transform(elements, quad.points)
        # TODO: Implement tests for different reference elements.

    def test_volume(
        self, quad: GaussLegendreQuadrature3D, grid: tuple[np.ndarray, np.ndarray]
    ) -> None:
        elements, volumes = grid
        integration_volumes: np.ndarray = quad.volume_factor(elements)
        assert pytest.approx(integration_volumes) == volumes / 8

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 3) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_points(
        self,
        quad: GaussLegendreQuadrature3D,
        grid: tuple[np.ndarray, np.ndarray],
        degree: int,
    ) -> None:
        if degree == 2:
            assert pytest.approx(quad.points) == np.array(
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
            assert pytest.approx(quad.points) == np.array(
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

    @pytest.mark.skipif(
        pytest.param("domain") != (1, 3) or pytest.param("grid") != 30,
        reason="Run only once for specific domain and grid",
    )
    def test_weights(
        self,
        quad: GaussLegendreQuadrature3D,
        grid: tuple[np.ndarray, np.ndarray],
        degree: int,
    ) -> None:
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
