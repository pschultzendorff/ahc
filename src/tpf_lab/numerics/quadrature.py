import abc
import logging
import math
from typing import Callable, Optional

import numpy as np
import scipy as sp

logger = logging.getLogger("__name__")


class Integral:

    def __init__(self, values: np.ndarray) -> None:
        self.shape: tuple = values.shape
        self._elementwise: np.ndarray = values

    @property
    def elementwise(self) -> np.ndarray:
        return self._elementwise

    @elementwise.setter
    def elementwise(self, values: np.ndarray):
        if values.shape == self.shape:
            self._elementwise = values
        else:
            raise ValueError(
                f"values needs to have shape {self.shape}, not {values.shape}"
            )

    def sum(self) -> float:
        return np.sum(self._elementwise)


class BaseScheme(abc.ABC):
    r"""_summary_

    All integration schemes work by Gaussian integration, i.e.,
    1. Define integration points :math:`X` and weights :math:`W` on a reference element
        :math:`K_{ref}` of length/area/volume :math:`|K_{ref}|`.
    2. Transform integration points from the reference element to the goal element
       :math:`K` via :math:`\varphi(X)`.
    3. Evaluate the function to be integrated at all points is evaluated at all points
       :math:`f(\varphi(X))`.
    4. Form the dot products of the values and weights :math:`f(\varphi(X)) \cdot W`.

    Now we only need to adjust for length/area/volume of :math:`K`. By the substitution
    formula for multivariate integration, it holds

    ..math::
        \int_{K} f(\mathbf{v})\, d\mathbf{v} =
        \int_{\varphi(K_{ref})} f(\mathbf{v})\, d\mathbf{v} =
        \int_{K_{ref})} f(\varphi(\mathbf{u})) \left|\det(D\varphi)(\mathbf{u})\right|
        \,d\mathbf{u}.

    The reference domain has length/area/volume :math:`|K_{ref}|` and the transformation
    :math:`\varphi` is nonsingular and affine. Thus, :math:`\det(D\varphi)` is constant
    and we observe by evaluation of :math:`f(\mathbf{v}) = 1` that
    :math:`\left|\det(D\varphi)(\mathbf{u})\right|` is exactly the length/area/volume of
    the domain of interest.

    Thus the quadrature scheme reads

    ..math::
        \int_{K} f(\mathbf{v})\, d\mathbf{v} \approx \frac{|K|}{|K_{ref}|} f(\varphi(X))
        \cdot W.


    Parameters:
        abc: _description_

    Returns:
        _description_

    """

    _weights: np.ndarray
    """``shape=(num_points,)``"""
    _points: np.ndarray
    """``shape=(num_points, dim)``"""

    def __init__(self, degree: int = 2) -> None:
        self.degree: int = degree
        self.dim: int
        pass

    @abc.abstractmethod
    def transform(self, elements: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        """

        Note: We assume ``elements`` to have shape ``(num_vertices, num_elements, dim)``
            and the return array to have shape ``(num_ref_points, num_elements, dim)``.

        """
        pass

    @abc.abstractmethod
    def volume_factor(self, elements: np.ndarray) -> np.ndarray:
        pass

    @property
    def points(self) -> np.ndarray:
        return self._points

    @points.setter
    def points(self, points: np.ndarray) -> None:
        self._points_setter(points)

    @abc.abstractmethod
    def _points_setter(self, points: np.ndarray) -> None:
        pass

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self._weights_setter(weights)

    @abc.abstractmethod
    def _weights_setter(self, weights: np.ndarray) -> None:
        pass

    def integrate(
        self, func: Callable[[np.ndarray], np.ndarray], elements: np.ndarray
    ) -> Integral:
        """Evalue the integral of ``func`` on all ``elements``.

        Note: We assume ``elements`` to have shape ``(num_vertices, num_elements, dim)``
            and the array returned by ``self.transform`` to have shape
            ``(num_ref_points, num_elements, dim)``.

        Parameters:
            func: Function to integrate.
            elements ``shape=(num_vertices, num_elements, dim)``:

        Returns:
            _description_

        """
        # TODO: Add vectorization with np.vectorize or numba or whatever makes this
        # fast.
        # Transform from reference domain.
        x: np.ndarray = self.transform(elements, self.points)
        volumes: np.ndarray = self.volume_factor(elements)

        # Evaluate function and integral.
        y: np.ndarray = func(x)

        # Calculate product over `num_vertices` for `y`; `self.weights` is
        # one-dimensional anyways.
        values: np.ndarray = volumes * np.tensordot(self.weights, y, axes=([0], [0]))
        return Integral(values)


class TrapezoidRule(BaseScheme):
    """Trapezoid rule for 1D-intervals."""

    def __init__(self, degree: int = 2) -> None:
        super().__init__(degree)
        self._points_setter()
        self._weights_setter()

    def transform(
        self, elements: np.ndarray, ref_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the endpoints of the goal elements."""
        return elements

    def volume_factor(self, elements: np.ndarray) -> np.ndarray:
        """Return lengths of the goal elements."""
        return elements[1, ...] - elements[0, ...]

    def _points_setter(self) -> None:
        # TODO: Can the signature of the abstract base method just be overriden?
        """Endpoints of the reference interval :math:`[0,1]`."""
        self._points = np.array([0, 1])

    def _weights_setter(self) -> None:
        # TODO: Can the signature of the abstract base method just be overriden?
        self._weights = np.array([0.5, 0.5])


class SimpsonsRule(BaseScheme):
    ...
    # TODO


class GaussLegendreQuadrature1D(BaseScheme):
    """Gauss-Legendre quadrature rule for 1D-intervals.

    The reference domain is :math:`[-1,1]`. See
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature.

    """

    def __init__(self, degree: int = 2) -> None:
        super().__init__(degree)
        self._points_setter()
        self._weights_setter()

    def transform(self, elements: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        """

        ..math::

        """
        self.sanity_check(elements)
        return (elements[1, ...] - elements[0, ...]) / 2 * ref_points.reshape(
            -1, 1, 1
        ) + (elements[1, ...] + elements[0, ...]) / 2

    def volume_factor(self, elements: np.ndarray) -> np.ndarray:
        """Return lengths of the goal elements divided by length of ref element."""
        self.sanity_check(elements)
        return (elements[1, ...] - elements[0, ...]) / 2

    def sanity_check(self, elements: np.ndarray) -> None:
        """Assert that all elements satisfy the following assumptions:
        - ``elements`` has shape ``(2, num_elements, 1)``
        - The vertices of each element are in the order left, right.

        """
        assert elements.shape == (2, elements.shape[1], 1)
        assert np.all(
            elements[0, :, 0] < elements[1, :, 0]
        ), "Element vertices ordered wrongly."

    def _points_setter(self) -> None:
        """Integration points on the reference interval :math:`[-1,1]`."""
        self._points = np.polynomial.legendre.leggauss(self.degree)[0]

    def _weights_setter(self) -> None:
        r"""Integration weights on the reference interval :math:`[-1,1]`."""
        self._weights = np.polynomial.legendre.leggauss(self.degree)[1]


class GaussLegendreQuadrature2D(BaseScheme):
    r"""Gauss-Legendre quadrature rule for 2D-rectangles.

    The reference domain is :math:`[-1,1] \times [-1,1]`. See
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature.

    """

    def __init__(self, degree: int = 2) -> None:
        super().__init__(degree)
        self._points_setter()
        self._weights_setter()

    def transform(self, elements: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        """
        Note:We assume ``elements`` to have shape ``(4, num_elements, 2)`` and the
            vertices of each element to be in the order top-left, top-right, bottom-left,
            bottom-right.

        """
        self.sanity_check(elements)
        x1: np.ndarray = elements[0, :, 0]
        x2: np.ndarray = elements[1, :, 0]
        y1: np.ndarray = elements[0, :, 1]
        y2: np.ndarray = elements[2, :, 1]
        return np.array(
            [
                [
                    x * (x2 - x1) * 0.5 + (x1 + x2) * 0.5,
                    y * (y2 - y1) * 0.5 + (y1 + y2) * 0.5,
                ]
                for x, y in ref_points
            ]
        )

    def volume_factor(self, elements: np.ndarray) -> np.ndarray:
        """

        Note:We assume ``elements`` to have shape ``(4, num_elements, 2)`` and the
            vertices of each element to be in the order top-left, top-right, bottom-left,
            bottom-right.

        """
        self.sanity_check(elements)
        return (
            (elements[1, :, 0] - elements[0, :, 0])
            * (elements[3, :, 1] - elements[2, :, 1])
            / 4
        )

    def sanity_check(self, elements: np.ndarray) -> None:
        """Assert that all elements satisfy the following assumptions:
        - ``elements`` has shape ``(4, num_elements, 2)``
        - Each element aligns with the coordinate axes
        - The vertices of each element are in the order top-left, top-right,
            bottom-left, bottom-right.

        """
        assert elements.shape == (4, elements.shape[1], 2)
        assert np.all(
            elements[0, :, 0] == elements[2, :, 0]
        ), "Left side (x-axis) vertices of an element have differing x-coordinates."
        assert np.all(
            elements[1, :, 0] == elements[3, :, 0]
        ), "Right side (x-axis) vertices of an element have differing x-coordinates."
        assert np.all(
            elements[0, :, 1] == elements[1, :, 1]
        ), "Top side (y-axis) vertices of an element have differing y-coordinates."
        assert np.all(
            elements[2, :, 1] == elements[3, :, 1]
        ), "Bottom side (y-axis) vertices of an element have differing y-coordinates."
        assert np.all(
            elements[0, :, 0] < elements[1, :, 0]
        ), "Element vertices ordered wrongly."
        assert np.all(
            elements[0, :, 1] < elements[2, :, 1]
        ), "Element vertices ordered wrongly."

    def _points_setter(self) -> None:
        points_1d, _ = np.polynomial.legendre.leggauss(self.degree)
        xv, yv = np.meshgrid(points_1d, points_1d)
        self._points = np.stack((xv, yv), axis=2).reshape(self.degree**2, 2)

    def _weights_setter(self) -> None:
        _, weights_1d = np.polynomial.legendre.leggauss(self.degree)
        self._weights = np.outer(weights_1d, weights_1d).flatten()


class GaussLegendreQuadrature3D(BaseScheme):
    r"""Gauss-Legendre quadrature rule for 3D-rectangular cuboids.

    The reference domain is :math:`[-1,1] \times [-1,1] \times [-1,1]`. See
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature.

    """

    def __init__(self, degree: int = 2) -> None:
        super().__init__(degree)
        self._points_setter()
        self._weights_setter()

    def transform(self, elements: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
        """
        Note: We assume ``elements`` to have shape ``(4, num_elements, 2)`` and the
            vertices of each element to be in the order top-left, top-right,
            bottom-left, bottom-right.

        """
        self.sanity_check(elements)
        x1: np.ndarray = elements[0, :, 0]
        x2: np.ndarray = elements[1, :, 0]
        y1: np.ndarray = elements[0, :, 1]
        y2: np.ndarray = elements[2, :, 1]
        z1: np.ndarray = elements[1, :, 2]
        z2: np.ndarray = elements[4, :, 2]
        return np.array(
            [
                [
                    x * (x2 - x1) * 0.5 + (x1 + x2) * 0.5,
                    y * (y2 - y1) * 0.5 + (y1 + y2) * 0.5,
                    z * (z2 - z1) * 0.5 + (z1 + z2) * 0.5,
                ]
                for x, y, z in ref_points
            ]
        )

    def volume_factor(self, elements: np.ndarray) -> np.ndarray:
        """

        Note: We assume ``elements`` to have shape ``(4, num_elements, 2)`` and the
            vertices of each element to be in the order top-left, top-right,
            bottom-left, bottom-right.

        """
        self.sanity_check(elements)
        return (
            (elements[1, :, 0] - elements[0, :, 0])
            * (elements[3, :, 1] - elements[2, :, 1])
            * (elements[4, :, 2] - elements[0, :, 2])
            / 8
        )

    def sanity_check(self, elements: np.ndarray) -> None:
        """Assert that all elements satisfy the following assumptions:
        - ``elements`` has shape ``(8, num_elements, 3)``
        - Each element aligns with the coordinate axes
        - The vertices of each element are in the order top-left-front, top-right-front,
            bottom-left-front, bottom-right-front, top-left-back, top-right-back,
            bottom-left-back, bottom-right-back

        """
        assert elements.shape == [8, elements.shape[1], 3]
        assert np.all(
            elements[0, :, 0]
            == elements[2, :, 0]
            == elements[4, :, 0]
            == elements[6, :, 0]
        ), "Left side (x-axis) vertices of an element have differing x-coordinates."
        assert np.all(
            elements[1, :, 0]
            == elements[3, :, 0]
            == elements[5, :, 0]
            == elements[7, :, 0]
        ), "Right side (x-axis) vertices of an element have differing x-coordinates."
        assert np.all(
            elements[0, :, 1]
            == elements[1, :, 1]
            == elements[4, :, 1]
            == elements[5, :, 1]
        ), "Top side (y-axis) vertices of an element have differing y-coordinates."
        assert np.all(
            elements[2, :, 1]
            == elements[3, :, 1]
            == elements[6, :, 1]
            == elements[7, :, 1]
        ), "Bottom side (y-axis) vertices of an element have differing y-coordinates."
        assert np.all(
            elements[0, :, 2]
            == elements[1, :, 2]
            == elements[2, :, 2]
            == elements[3, :, 2]
        ), "Front side (z-axis) vertices of an element have differing z-coordinates."
        assert np.all(
            elements[4, :, 2]
            == elements[5, :, 2]
            == elements[6, :, 2]
            == elements[7, :, 2]
        ), "Back side (z-axis) vertices of an element have differing z-coordinates."
        assert np.all(
            elements[0, :, 0] < elements[1, :, 0]
        ), "Element vertices ordered wrongly."
        assert np.all(
            elements[0, :, 1] < elements[2, :, 1]
        ), "Element vertices ordered wrongly."
        assert np.all(
            elements[0, :, 2] < elements[5, :, 2]
        ), "Element vertices ordered wrongly."

    def _points_setter(self) -> None:
        points_1d, _ = np.polynomial.legendre.leggauss(self.degree)
        xv, yv, zv = np.meshgrid(points_1d, points_1d, points_1d)
        self._points = np.stack((xv, yv, zv), axis=2).reshape(self.degree**3, 3)

    def _weights_setter(self) -> None:
        _, weights_1d = np.polynomial.legendre.leggauss(self.degree)
        self._weights = np.outer(weights_1d, weights_1d, weights_1d).flatten()


class GaussJacobiQuadrature(BaseScheme):
    ...
    # TODO


class TriangleQuadrature(BaseScheme):
    """... rule for triangles."""

    _affine_coefficients: np.ndarray
    r"""``shape=(num_elements, 6)``. Coefficients for the affine transformation from the
    reference element to each goal element.

    For each goal element, the coefficients are stored in the order
    :math:`(\varphi_{1,1}, \varphi_{1,2}, \varphi_{1,3}, \varphi_{2,1}, \varphi_{2,2},
    \varphi_{2,3})`.

    """
    _A: np.ndarray
    r"""``shape=(num_elements, dim=2, 2)``. Matrix for the affine transformation from the
    reference element to each goal element.

    ..math::
        A = 
        \begin{pmatrix}
            \varphi_{1,1} & \varphi_{1,2} \\
            \varphi_{2,1} & \varphi_{2,2}
        \end{pmatrix}

    """
    _b: np.ndarray
    r"""``shape=(num_elements, dim=2, 1)``. Vector for the affine transformation from the
    reference element to each goal element.

    ..math::
        b = 
        \begin{pmatrix}
            \varphi_{1,3} \\
            \varphi_{2,3}
        \end{pmatrix}.

    """
    _vertices: np.ndarray
    """``shape=(3, 2)``. Vertices of the reference element."""

    def __init__(self, degree: int = 2) -> None:
        super().__init__(degree)
        self._vertices = np.array([[0, 0], [0, math.sqrt(2)], [math.sqrt(2), 0]])
        self._points_setter()
        self._weights_setter()

    def calc_affine_coefficients(
        self, elements: np.ndarray, ref_element: Optional[np.ndarray] = None
    ) -> None:
        r"""Compute coefficients for affine transformation from reference element to
        goal elements.

        Solve equation systems to find coordinates of the affine transformation for each
        goal element. We have
        ..math::
            \varphi(x, y) =
            \begin{pmatrix}
                \varphi_{1,1} & \varphi_{1,2} \\
                \varphi_{2,1} & \varphi_{2,2}
            \end{pmatrix}
            (x, y)^T +
            \begin{pmatrix}
                \varphi_{1,3} \\
                \varphi_{2,3}
            \end{pmatrix}.

        To find :math:`\varphi_{i,j},` we solve
        ..math::
            \begin{pmatrix}
                x_1 & y_1 & 1 \\
                x_2 & y_2 & 1 \\
                x_3 & y_3 & 1
            \end{pmatrix}
            (\varphi_{1,1}, \varphi_{1,2}, \varphi_{1,3})^T =
            (\hat{x}_1, \hat{x}_2, \hat{x}_3)^T

        and
        ..math::
            \begin{pmatrix}
                x_1 & y_1 & 1 \\
                x_2 & y_2 & 1 \\
                x_3 & y_3 & 1
            \end{pmatrix}
            (\varphi_{2,1}, \varphi_{2,2}, \varphi_{2,3})^T =
            (\hat{y}_1, \hat{y}_2, \hat{y}_3)^T,

        where :math:`(\hat{x}_i, \hat{y}_i)` are the vertices of the goal element.

        Parameters:
            elements ``shape=(num_vertices=3,  num_elements, dim=2)``:
                Vertices of goal elements.
            ref_element ``shape=(num_vertices=3, dim=2)``:
                Vertices of reference element.

        """
        if ref_element is None:
            ref_element = self._vertices
        # Compute coefficients of affine transformation to new x coordinate.
        b_x: list[np.ndarray] = list(elements[..., 0].reshape(-1, 3))
        # Matrix is the same for both the transformation to the new x and y coordinate.
        A: list[np.ndarray] = [
            np.tile(
                np.stack(
                    [
                        ref_element[:, 0],
                        ref_element[:, 1],
                        np.ones(
                            3,
                        ),
                    ],
                    axis=-1,
                ),
                (1, 2),
            )
        ] * len(b_x)
        affine_coefficients_x: np.ndarray = np.array(np.linalg.solve(A, b_x))

        # Compute coefficients of affine transformation to new y coordinate.
        b_y: list[np.ndarray] = list(elements[..., 1].reshape(-1, 3))
        affine_coefficients_y: np.ndarray = np.array(np.linalg.solve(A, b_y))

        self._affine_coefficients = np.concatenate(
            [affine_coefficients_x, affine_coefficients_y], axis=-1
        )
        # Set matrix and vector for affine transformation.
        self._A = np.stack(
            [
                self._affine_coefficients[..., [0, 1]],
                self._affine_coefficients[..., [3, 4]],
            ],
            axis=-2,
        )
        self._b = np.expand_dims(self._affine_coefficients[..., [2, 5]], axis=-1)

    def transform(
        self, elements: np.ndarray, ref_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return integration points on the goal elements.

        Parameters:
            elements ``shape=(num_vertices=3,  num_elements, dim=2)``:
                Vertices of goal elements.
            ref_points ``shape=(num_ref_points, dim=2)``:
                Integration points on reference element.

        Returns:
            transformed_points ``shape=(num_elements, num_ref_points, 2)

        """
        if ref_points is None:
            ref_points = self._points
        if not self._affine_coefficients:
            self.calc_affine_coefficients(elements)
        transformed_points: np.ndarray = (
            np.matmul(self._A, np.expand_dims(ref_points, axis=-1)).squeeze() + self._b
        )
        return transformed_points

    def volume_factor(self, elements: np.ndarray) -> np.ndarray:
        r"""Return area of the goal elements.

        The area of a triangle with vertices :math:`(x_1, y_1), (x_2, y_2), (x_3, y_3)`
        is given by

        ..math::
            \frac{1}{2} \left[x_1(y_2 - y_3), + x_2(y_3 - y_1) + x_3(y_1 - y_2)\right]

        Parameters:
            elements ``shape=(num_vertices=3,  num_elements, dim=2)``:

        Returns:
            volumes ``shape(num_elements,)``:

        """
        return (
            elements[0, :, 0] * (elements[1, :, 1] - elements[2, :, 1])
            + elements[1, :, 0] * (elements[2, :, 1] - elements[0, :, 1])
            + elements[2, :, 0] * (elements[0, :, 1] - elements[1, :, 1])
        ) / 2

    def _points_setter(self) -> None:
        r"""Integration points on the reference triangle :math:`\{(x,y) | 0 \leq x,
        x + y \leq \sqrt{2}\}` of area 1."""
        self._points = np.array([0, 1])

    def _weights_setter(self) -> None:
        self._weights = np.array([0.5, 0.5])


class MonteCarloQuadrature(BaseScheme):
    ...
    # TODO
