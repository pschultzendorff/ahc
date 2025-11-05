import abc
import logging
from typing import Callable, Literal

import numpy as np
import porepy as pp
import scipy.sparse as sps
from numba import njit

logger = logging.getLogger("__name__")


class Integral:
    def __init__(self, values: np.ndarray) -> None:
        self.shape: tuple = values.shape
        """Shape of the integral values. Must be ``(num_elements, dim_func)``, where
        ``dim_func`` is the dimension of the function values that were integrated."""
        if len(self.shape) != 2:
            raise ValueError("Values must have shape (num_elements, dim_func).")
        self._elementwise: np.ndarray = values
        """``shape=(num_elements, dim_func)``. Values of the integral on each
        element."""

    def __add__(self, other: "Integral") -> "Integral":
        if self.shape == other.shape:
            return Integral(self.elementwise + other.elementwise)
        else:
            raise ValueError(f"Shapes {self.shape} and {other.shape} not aligned.")

    def __neg__(self, other: "Integral") -> "Integral":
        if self.shape == other.shape:
            return Integral(self.elementwise - other.elementwise)
        else:
            raise ValueError(f"Shapes {self.shape} and {other.shape} not aligned.")

    def __mul__(self, other: "Integral | float") -> "Integral":
        if isinstance(other, float):
            return Integral(other * self.elementwise)
        else:
            raise TypeError(
                f"__mul__ not implemented for types {type(self)}"
                + f" and {type(other)}."
            )

    def __rmul__(self, other: "Integral | float") -> "Integral":
        if isinstance(other, float):
            return Integral(other * self.elementwise)
        else:
            raise TypeError(
                f"__rmul__ not implemented for types {type(self)}"
                + f" and {type(other)}."
            )

    def __pow__(self, other: float) -> "Integral":
        if isinstance(other, float) or isinstance(other, int):
            return Integral(self.elementwise**other)
        else:
            raise TypeError(
                f"__pow__ not implemented for types {type(self)}"
                + f" and {type(other)}."
            )

    @property
    def elementwise(self) -> np.ndarray:
        return self._elementwise

    @elementwise.setter
    def elementwise(self, values: np.ndarray):
        if values.shape == self.shape:
            self._elementwise = values
        else:
            raise ValueError(
                f"Values needs to have shape {self.shape}, not {values.shape}."
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

    .. math::
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

    .. math::
        \int_{K} f(\mathbf{v})\, d\mathbf{v} \approx \frac{|K|}{|K_{ref}|} f(\varphi(X))
        \cdot W.

    """

    _points: np.ndarray
    """``shape=(num_ref_points, num_elements, dim)``"""
    _volumes: np.ndarray
    """``shape=(num_elements,)``"""

    _reference_volume: float
    _reference_points: np.ndarray
    """``shape=(num_ref_points, dim)``"""
    _weights: np.ndarray
    """``shape=(num_ref_points,)``"""

    def __init__(self, degree: int = 2) -> None:
        self.degree: int = degree
        self.dim: int
        pass

    @abc.abstractmethod
    def transform(self, elements: np.ndarray) -> np.ndarray:
        """

        Note: We assume ``elements`` to have shape ``(num_vertices, num_elements, dim)``
            and the return array to have shape ``(num_ref_points, num_elements, dim)``.

        """
        pass

    @abc.abstractmethod
    def calc_volumes(self, elements: np.ndarray) -> np.ndarray:
        pass

    @property
    def points(self) -> np.ndarray:
        """``shape=(num_ref_points, num_elements, dim)``"""
        return self._points

    @points.setter
    def points(self, points: np.ndarray) -> None:
        self._points = points

    @property
    def volumes(self) -> np.ndarray:
        """``shape=(num_elements,)``"""
        return self._volumes

    @volumes.setter
    def volumes(self, volumes: np.ndarray) -> None:
        self._volumes = volumes

    @property
    def reference_volume(self) -> float:
        return self._reference_volume

    @reference_volume.setter
    def reference_volume(self, reference_volume: float) -> None:
        self._reference_volume_setter(reference_volume)

    @abc.abstractmethod
    def _reference_volume_setter(self, reference_volume: float | None = None) -> None:
        pass

    @property
    def reference_points(self) -> np.ndarray:
        return self._reference_points

    @reference_points.setter
    def reference_points(self, reference_points: np.ndarray) -> None:
        self._reference_points_setter(reference_points)

    @abc.abstractmethod
    def _reference_points_setter(
        self, reference_points: np.ndarray | None = None
    ) -> None:
        pass

    @property
    def weights(self) -> np.ndarray:
        """``shape=(num_ref_points,)``"""
        return self._weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self._weights_setter(weights)

    @abc.abstractmethod
    def _weights_setter(self, weights: np.ndarray | None = None) -> None:
        pass

    def integrate(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        elements: np.ndarray | None = None,
        recalc_points: bool = True,
        recalc_volumes: bool = True,
        njit: bool = False,
    ) -> Integral:
        """Evaluate the integral of a function on all elements.

        Parameters:
            func: Function to integrate. Must take an array of shape
                ``(num_ref_points, num_elements, dim)`` as input and return an array of
                shape ``(num_ref_points, num_elements, dim_func)``. Here, ``dim_func``
                is the dimension of the function values. If ``dim_func`` is 1, the
                output may have shape ``(num_ref_points, num_elements)`` instead.
            elements: ``shape=(num_vertices, num_elements, dim)``
                Elements to integrate over.
            recalc_points: Whether to recalculate the integration points.
            recalc_volumes: Whether to recalculate the volumes.
            njit: Whether to use numba for the integration. WARNING: Ideally, ``func``
                should be decorated with ``@njit`` as well. However, if ``func`` changes
                between calls, this gets super slow.

        Returns:
            Integral: Integral object containing the integrated values.

        """
        if recalc_points or not hasattr(self, "_points"):
            if elements is None:
                raise ValueError(
                    "Elements must be provided for points to be recalculated."
                )
            self.points = self.transform(elements)

        if recalc_volumes or not hasattr(self, "_volumes"):
            if elements is None:
                raise ValueError(
                    "Elements must be provided for volumes to be recalculated."
                )
            self.volumes = self.calc_volumes(elements)

        y: np.ndarray = func(self.points)

        # If y has shape (num_ref_points, num_elements), add a dimension.
        if y.ndim == 2:
            y = y[..., None]
        if njit:
            values: np.ndarray = self._integrate_njit(
                y, self.weights, self.volumes, self.reference_volume
            )
        else:
            values = (self.volumes / self.reference_volume)[..., None] * np.tensordot(
                self.weights, y, axes=([0], [0])
            )
        return Integral(values)

    # When recalc_points and recalc_volumes are set to False, only this method should
    # make use of numba as the other methods are only called once. When they are
    # recalculated each time, ``transform`` and ``calc_volumes`` should be decorated
    # with ``@njit`` as well.
    # TODO How to implement conditional numba decoration?
    # TODO Must ``func`` be decorated with ``@njit`` for this to be efficient?
    # TODO Make this efficient!
    @staticmethod
    @njit
    def _integrate_njit(
        y: np.ndarray,
        weights: np.ndarray,
        volumes: np.ndarray,
        reference_volume: float,
    ) -> np.ndarray:
        """Evaluate the integral of a function on all elements.

        Parameters:
            y: ``shape=(num_ref_points, num_elements, dim_func)``
                Function values at integration points for all elements.
            weights: ``shape=(num_ref_points,)``
                Integration weights for all elements.
            volumes: ``shape=(num_elements,)``
                Volumes of the elements.
            reference_volume: Volume of the reference element.

        Returns:
            values: ``shape=(num_elements, dim_func)``
                Array containing the integrated values.

        """
        # Broadcast all arrays to 3 dimensions s.t. they can be multiplied elementwise.
        # If this is not done, the automatic broadcasting might do weird things.
        values: np.ndarray = (
            volumes[..., None]
            / reference_volume
            * np.sum(weights[..., None, None] * y, axis=0)
        )
        return values


class TrapezoidRule(BaseScheme):
    """
    Trapezoid rule for 1D-intervals.
    """

    def __init__(self, degree: int = 2) -> None:
        """
        Initialize the TrapezoidRule class.

        Parameters:
            degree: Degree of the integration scheme.
        """
        super().__init__(degree)
        self._reference_volume_setter()
        self._reference_points_setter()
        self._weights_setter()

    def transform(self, elements: np.ndarray) -> np.ndarray:
        """
        Return the endpoints of the goal elements.

        Parameters:
            elements: ```(num_vertices, num_elements, 1)``
                Elements to transform.

        Returns:
            np.ndarray: ``shape=(num_ref_points, num_elements, 1)``
                Transformed integration points.

        """
        return elements

    def calc_volumes(self, elements: np.ndarray) -> np.ndarray:
        """
        Return lengths of the goal elements.

        Parameters:
            elements: Elements to calculate volumes for.

        Returns:
            np.ndarray: Volumes of the elements.

        """
        volumes: np.ndarray = elements[1, ..., 0] - elements[0, ..., 0]
        assert volumes.shape == (elements.shape[1],), "Volumes have wrong shape."
        return volumes

    def _reference_volume_setter(self, reference_volumes: float | None = None) -> None:
        """
        Set the volume of the reference element.

        Parameters:
            reference_volumes: Volume of the reference element.

        """
        self._reference_volume = 1.0

    def _reference_points_setter(
        self, reference_points: np.ndarray | None = None
    ) -> None:
        """
        Set the points on the reference element.

        Parameters:
            reference_points: Points on the reference element.

        """
        self._reference_points = np.array([[0], [1]])

    def _weights_setter(self, weights: np.ndarray | None = None) -> None:
        """
        Set the weights for the integration points.

        Parameters:
            weights: Weights for the integration points.

        """
        self._weights = np.array([0.5, 0.5])


class SimpsonsRule(BaseScheme):
    """
    Simpson's rule for 1D-intervals.
    """

    # TODO: Implement SimpsonsRule


# TODO Unify GaussLegendreQuadrature1D, GaussLegendreQuadrature2D, and
# GaussLegendreQuadrature3D into one class with a dimension parameter.
class GaussLegendreQuadrature1D(BaseScheme):
    """
    Gauss-Legendre quadrature rule for 1D-intervals.

    The reference domain is :math:`[-1,1]`.

    """

    def __init__(self, degree: int = 2) -> None:
        """
        Initialize the GaussLegendreQuadrature1D class.

        Parameters:
            degree: Degree of the integration scheme.

        """
        super().__init__(degree)
        self._reference_volume_setter()
        self._reference_points_setter()
        self._weights_setter()

    def transform(self, elements: np.ndarray) -> np.ndarray:
        """
        Transform integration points from the reference element to the goal element.

        Parameters:
            elements: Elements to transform.

        Returns:
            np.ndarray: Transformed integration points.

        """
        self.sanity_check(elements)
        return (
            elements[1, ...] - elements[0, ...]
        ) / 2 * self.reference_points.reshape(-1, 1, 1) + (
            elements[1, ...] + elements[0, ...]
        ) / 2

    def calc_volumes(self, elements: np.ndarray) -> np.ndarray:
        """Return lengths of the goal elements."""
        self.sanity_check(elements)
        volumes: np.ndarray = elements[1, ..., 0] - elements[0, ..., 0]
        assert volumes.shape == (elements.shape[1],), "Volumes have wrong shape."
        return volumes

    def sanity_check(self, elements: np.ndarray) -> None:
        """Assert that all elements satisfy the following assumptions:
        - ``elements`` has shape ``(2, num_elements, 1)``
        - The vertices of each element are in the order left, right.

        Note that we allow elements to have volume 0.

        """
        assert elements.shape == (2, elements.shape[1], 1)
        assert np.all(elements[0, :, 0] <= elements[1, :, 0]), (
            "Element vertices ordered wrongly."
        )

    def _reference_volume_setter(self, reference_volumes: float | None = None) -> None:
        self._reference_volume = 2.0

    def _reference_points_setter(
        self, reference_points: np.ndarray | None = None
    ) -> None:
        """Integration points on the reference interval :math:`[-1,1]`."""
        self._reference_points = np.polynomial.legendre.leggauss(self.degree)[0][
            ..., None
        ]

    def _weights_setter(self, weights: np.ndarray | None = None) -> None:
        r"""Integration weights on the reference interval :math:`[-1,1]`."""
        self._weights = np.polynomial.legendre.leggauss(self.degree)[1]
        assert self._weights.shape == (self.degree,), "Weights have wrong shape."


class GaussLegendreQuadrature2D(BaseScheme):
    r"""Gauss-Legendre quadrature rule for 2D-rectangles.

    The reference domain is :math:`[-1,1] \times [-1,1]`. See
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature.

    To obtain correct results, ``elements`` passed to ``integrate`` must be in the
    correct format and shape. Check :meth:~`GaussLegendreQuadrature2D.transform` for
    details.

    """

    def __init__(self, degree: int = 2) -> None:
        super().__init__(degree)
        self._reference_volume_setter()
        self._reference_points_setter()
        self._weights_setter()

    def transform(self, elements: np.ndarray) -> np.ndarray:
        """

        Note: We assume ``elements`` to have shape ``(4, num_elements, 2)`` and the
            vertices of each element to be in the order bottom-left, bottom-right,
          top-left, top-right.

        """
        self.sanity_check(elements)
        x1: np.ndarray = elements[0, :, 0]
        x2: np.ndarray = elements[1, :, 0]
        y1: np.ndarray = elements[0, :, 1]
        y2: np.ndarray = elements[2, :, 1]
        return np.stack(
            [  # These have ``shape==(num_elements,2)``
                np.column_stack(
                    [
                        # These have ``shape==(num_elements,)``
                        x * (x2 - x1) * 0.5 + (x1 + x2) * 0.5,
                        y * (y2 - y1) * 0.5 + (y1 + y2) * 0.5,
                    ]
                )
                for x, y in self.reference_points
            ],
            axis=0,
        )

    def calc_volumes(self, elements: np.ndarray) -> np.ndarray:
        """

        Note: We assume ``elements`` to have shape ``(4, num_elements, 2)`` and the
            vertices of each element to be in the order bottom-left, bottom-right,
          top-left, top-right.

        """
        self.sanity_check(elements)
        volumes: np.ndarray = (
            (elements[0, :, 0] - elements[1, :, 0])
            * (elements[0, :, 1] - elements[2, :, 1])
        ).squeeze()
        assert volumes.shape == (elements.shape[1],), "Volumes have wrong shape."
        return volumes

    def sanity_check(self, elements: np.ndarray) -> None:
        """Assert that all elements satisfy the following assumptions:
        - ``elements`` has shape ``(4, num_elements, 2)``
        - Each element aligns with the coordinate axes
        - The vertices of each element are in the order bottom-left, bottom-right,
          top-left, top-right.
            .

        Note: We allow elements to have volume 0.

        """
        assert elements.shape == (4, elements.shape[1], 2)
        assert np.all(elements[0, :, 0] == elements[2, :, 0]), (
            "Left side (x-axis) vertices of an element have differing x-coordinates."
        )
        assert np.all(elements[1, :, 0] == elements[3, :, 0]), (
            "Right side (x-axis) vertices of an element have differing x-coordinates."
        )
        assert np.all(elements[0, :, 1] == elements[1, :, 1]), (
            "Top side (y-axis) vertices of an element have differing y-coordinates."
        )
        assert np.all(elements[2, :, 1] == elements[3, :, 1]), (
            "Bottom side (y-axis) vertices of an element have differing y-coordinates."
        )
        assert np.all(elements[0, :, 0] <= elements[1, :, 0]), (
            "Element vertices ordered wrongly."
        )
        assert np.all(elements[0, :, 1] <= elements[2, :, 1]), (
            "Element vertices ordered wrongly."
        )

    def _reference_volume_setter(self, reference_volumes: float | None = None) -> None:
        self._reference_volume = 4.0

    def _reference_points_setter(
        self, reference_points: np.ndarray | None = None
    ) -> None:
        points_1d = np.polynomial.legendre.leggauss(self.degree)[0]
        xv, yv = np.meshgrid(points_1d, points_1d)
        self._reference_points = np.stack((xv, yv), axis=2).reshape(self.degree**2, 2)

    def _weights_setter(self, weights: np.ndarray | None = None) -> None:
        weights_1d = np.polynomial.legendre.leggauss(self.degree)[1]
        self._weights = np.outer(weights_1d, weights_1d).flatten()
        assert self._weights.shape == (self.degree**2,), "Weights have wrong shape."


class GaussLegendreQuadrature3D(BaseScheme):
    r"""Gauss-Legendre quadrature rule for 3D-rectangular cuboids.

    The reference domain is :math:`[-1,1] \times [-1,1] \times [-1,1]`. See
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature.

    To obtain correct results, ``elements`` passed to ``integrate`` must be in the
    correct format and shape. Check :meth:~`GaussLegendreQuadrature3D.transform` for
    details.

    """

    def __init__(self, degree: int = 2) -> None:
        super().__init__(degree)
        self._reference_volume_setter()
        self._reference_points_setter()
        self._weights_setter()

    def transform(self, elements: np.ndarray) -> np.ndarray:
        """
        Note: We assume ``elements`` to have shape ``(4, num_elements, 3)`` and the
            vertices of each element to be in the order bottom-left-front,
            bottom--front, top-left-front, top-right-front, bottom-left-back,
            bottom--back, top-left-back, top-right-back.

        """
        self.sanity_check(elements)
        x1: np.ndarray = elements[0, :, 0]
        x2: np.ndarray = elements[1, :, 0]
        y1: np.ndarray = elements[0, :, 1]
        y2: np.ndarray = elements[2, :, 1]
        z1: np.ndarray = elements[1, :, 2]
        z2: np.ndarray = elements[4, :, 2]
        return np.stack(
            [
                # These have ``shape==(num_elements,3)``
                np.column_stack(
                    [
                        # These have ``shape==(num_elements,)``
                        x * (x2 - x1) * 0.5 + (x1 + x2) * 0.5,
                        y * (y2 - y1) * 0.5 + (y1 + y2) * 0.5,
                        z * (z2 - z1) * 0.5 + (z1 + z2) * 0.5,
                    ]
                )
                for x, y, z in self.reference_points
            ],
            axis=0,
        )

    def calc_volumes(self, elements: np.ndarray) -> np.ndarray:
        """

        Note: We assume ``elements`` to have shape ``(4, num_elements, 2)`` and the
            vertices of each element to be in the order bottom-left-front,
            bottom--front, top-left-front, top-right-front, bottom-left-back,
            bottom--back, top-left-back, top-right-back.

        """
        self.sanity_check(elements)
        volumes: np.ndarray = (
            (elements[1, :, 0] - elements[0, :, 0])
            * (elements[3, :, 1] - elements[2, :, 1])
            * (elements[4, :, 2] - elements[0, :, 2])
        ).squeeze()
        assert volumes.shape == elements.shape[1], "Volumes have wrong shape."
        return volumes

    def sanity_check(self, elements: np.ndarray) -> None:
        """Assert that all elements satisfy the following assumptions:
        - ``elements`` has shape ``(8, num_elements, 3)``
        - Each element aligns with the coordinate axes
        - The vertices of each element are in the order bottom-left-front,
            bottom--front, top-left-front, top-right-front, bottom-left-back,
            bottom--back, top-left-back, top-right-back.

        Note: We allow elements to have volume 0.

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
        assert np.all(elements[0, :, 0] <= elements[1, :, 0]), (
            "Element vertices ordered wrongly."
        )
        assert np.all(elements[0, :, 1] <= elements[2, :, 1]), (
            "Element vertices ordered wrongly."
        )
        assert np.all(elements[0, :, 2] <= elements[5, :, 2]), (
            "Element vertices ordered wrongly."
        )

    def _reference_volume_setter(self, reference_volumes: float | None = None) -> None:
        self._reference_volume = 8.0

    def _reference_points_setter(
        self, reference_points: np.ndarray | None = None
    ) -> None:
        points_1d = np.polynomial.legendre.leggauss(self.degree)[0]
        xv, yv, zv = np.meshgrid(points_1d, points_1d, points_1d)
        self._reference_points = np.stack((xv, yv, zv), axis=2).reshape(
            self.degree**3, 3
        )

    def _weights_setter(self, weights: np.ndarray | None = None) -> None:
        weights_1d = np.polynomial.legendre.leggauss(self.degree)[1]
        self._weights = np.outer(weights_1d, np.outer(weights_1d, weights_1d)).flatten()
        assert self._weights.shape == (self.degree**3,), "Weights have wrong shape."


class GaussJacobiQuadrature(BaseScheme):
    ...
    # TODO


class TriangleQuadrature(BaseScheme):
    r"""... rule for triangles.

    FIXME Should we even allow for different reference triangles? This would also
    require to change the points etc!

    https://mathsfromnothing.au/triangle-quadrature-rules/

    The reference triangle is :math:`\{(x,y) | 0 \leq x + y \leq 1\}` with area 1/2.

    """

    _affine_coefficients: np.ndarray
    r"""``shape=(num_elements, 6)``. Coefficients for the affine transformation from the
    reference element to each goal element.

    For each goal element, the coefficients are stored in the order
    :math:`(\varphi_{1,1}, \varphi_{1,2}, \varphi_{1,3}, \varphi_{2,1}, \varphi_{2,2},
    \varphi_{2,3})`.

    """
    _A: np.ndarray
    r"""``shape=(1, num_elements, dim=2, 2)``. Matrix for the affine transformation from
     the reference element to each goal element.

    .. math::
        A = 
        \begin{pmatrix}
            \varphi_{1,1} & \varphi_{1,2} \\
            \varphi_{2,1} & \varphi_{2,2}
        \end{pmatrix}

    """
    _b: np.ndarray
    r"""``shape=(1, num_elements, dim=2)``. Vector for the affine transformation from
     the reference element to each goal element.

    .. math::
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
        self._vertices = np.array([[0, 0], [0, 1], [1, 0]])
        self._reference_volume_setter()
        self._reference_points_setter()
        self._weights_setter()

    def calc_affine_coefficients(
        self, elements: np.ndarray, ref_element: np.ndarray | None = None
    ) -> None:
        r"""Compute coefficients for affine transformation from reference element to
        goal elements.

        Solve equation systems to find coordinates of the affine transformation for each
        goal element. We have
        .. math::
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
        .. math::
            \begin{pmatrix}
                x_1 & y_1 & 1 \\
                x_2 & y_2 & 1 \\
                x_3 & y_3 & 1
            \end{pmatrix}
            (\varphi_{1,1}, \varphi_{1,2}, \varphi_{1,3})^T =
            (\hat{x}_1, \hat{x}_2, \hat{x}_3)^T

        and
        .. math::
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
        b_x: list[np.ndarray] = list(elements[..., 0].transpose())
        # Matrix is the same for both the transformation to the new x and y coordinate.
        A: np.ndarray = np.stack(
            [
                ref_element[:, 0],
                ref_element[:, 1],
                np.ones(
                    3,
                ),
            ],
            axis=-1,
        )
        # NOTE This is inefficient. We would ideally want to use a parallelized version.
        affine_coefficients_x: np.ndarray = np.array(
            [np.linalg.solve(A, b) for b in b_x]
        )

        # Compute coefficients of affine transformation to new y coordinate.
        b_y: list[np.ndarray] = list(elements[..., 1].transpose())
        affine_coefficients_y: np.ndarray = np.array(
            [np.linalg.solve(A, b) for b in b_y]
        )

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
        )[None, ...]
        self._b = self._affine_coefficients[..., [2, 5]][None, ...]

    def transform(self, elements: np.ndarray) -> np.ndarray:
        """Return integration points on the goal elements.

        Parameters:
            elements ``shape=(num_vertices=3,  num_elements, dim=2)``:
                Vertices of goal elements.

        Returns:
            transformed_points ``shape=(num_elements, num_ref_points, 2)

        """
        self.calc_affine_coefficients(elements)
        transformed_points: np.ndarray = (
            np.matmul(self._A, self.reference_points[:, None, :, None]).squeeze()
            + self._b
        )
        return transformed_points

    def calc_volumes(self, elements: np.ndarray) -> np.ndarray:
        r"""Return area of the goal elements.

        The area of a triangle with vertices :math:`(x_1, y_1), (x_2, y_2), (x_3, y_3)`
        is given by

        .. math::
            \frac{1}{2} \left[x_1(y_2 - y_3), + x_2(y_3 - y_1) + x_3(y_1 - y_2)\right]

        Parameters:
            elements ``shape=(num_vertices=3,  num_elements, dim=2)``:

        Returns:
            volumes ``shape(num_elements,)``:

        """
        return (
            np.abs(
                elements[0, :, 0] * (elements[1, :, 1] - elements[2, :, 1])
                + elements[1, :, 0] * (elements[2, :, 1] - elements[0, :, 1])
                + elements[2, :, 0] * (elements[0, :, 1] - elements[1, :, 1])
            )
            / 2
        )

    def _reference_volume_setter(self, reference_volumes: float | None = None) -> None:
        self._reference_volume = 0.5

    def _reference_points_setter(
        self, reference_points: np.ndarray | None = None
    ) -> None:
        r"""Integration points on the reference triangle :math:`\{(x,y) | 0 \leq x + y
        \leq 1\}` of area 1/2."""
        if self.degree == 1:
            self._reference_points = np.array([[1 / 3, 1 / 3]])
        elif self.degree == 2:
            self._reference_points = np.array(
                [[1 / 6, 2 / 3], [1 / 6, 1 / 6], [1 / 6, 2 / 3]]
            )
        elif self.degree == 3:
            self._reference_points = np.array(
                [[1 / 3, 1 / 3], [0.2, 0.6], [0.2, 0.2], [0.6, 0.2]]
            )
        elif self.degree == 4:
            self._reference_points = np.array(
                [
                    [0.445948490915965, 0.108103018168070],
                    [0.445948490915965, 0.445948490915965],
                    [0.108103018168070, 0.445948490915965],
                    [0.091576213509771, 0.816847572980459],
                    [0.091576213509771, 0.091576213509771],
                    [0.816847572980459, 0.091576213509771],
                ]
            )
        else:
            raise ValueError(f"Degree {self.degree} not implemented.")

    def _weights_setter(self, weights: np.ndarray | None = None) -> None:
        if self.degree == 1:
            self._weights = np.array([1 / 2])
        elif self.degree == 2:
            self._weights = np.array([1 / 6, 1 / 6, 1 / 6])
        elif self.degree == 3:
            self._weights = np.array([-9 / 32, 25 / 96, 25 / 96, 25 / 96])
        elif self.degree == 4:
            self._weights = np.array(
                [
                    0.1116907948390055,
                    0.1116907948390055,
                    0.1116907948390055,
                    0.054975871827661,
                    0.054975871827661,
                    0.054975871827661,
                ]
            )
        else:
            raise ValueError(f"Degree {self.degree} not implemented.")


class MonteCarloQuadrature(BaseScheme):
    ...
    # TODO


def get_quadpy_elements(
    sd: pp.Grid, grid_type: Literal["simplex", "cartesian"] = "simplex"
) -> np.ndarray:
    """
    Assembles the elements of a given grid in quadpy format:
    https://pypi.org/project/quadpy/.

    Parameters:
        sd: PorePy grid.

    Returns:
        quadpy_elements: ``shape=(num_nodes, num_cells, sd.dim)``
            Elements in QuadPy format. For line segments, the shape is
            ``(2, num_cells)``.

    """
    nodes_per_cell: int = sd.dim + 1 if grid_type == "simplex" else sd.dim * 2
    # Renaming variables
    num_cells = sd.num_cells

    # Getting node coordinates for each cell
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(num_cells, nodes_per_cell)
    nodes_coor_cell = sd.nodes[:, nodes_of_cell]

    # Stacking node coordinates
    # cnc_stckd = np.empty([num_cells, nodes_per_cell * sd.dim])
    # col = 0
    # for vertex in range(nodes_per_cell):
    #     for dim in range(sd.dim):
    #         cnc_stckd[:, col] = nodes_coor_cell[dim][:, vertex]
    #         col += 1

    # # Reshaping to please quadpy format i.e., (corners, num_cells, dim)
    # elelemt_coords: np.ndarray = np.reshape(cnc_stckd, np.array([num_cells, nodes_per_cell, sd.dim]))
    # elements: np.ndarray = np.swapaxes(elelemt_coords, 0, 1)
    # Stacking node coordinates
    elements: np.ndarray = np.empty([nodes_per_cell, num_cells, sd.dim])
    for vertex in range(nodes_per_cell):
        for dim in range(sd.dim):
            elements[vertex, :, dim] = nodes_coor_cell[dim][:, vertex]

    # For some reason, quadpy needs a different formatting for line segments
    if sd.dim == 1:
        # TODO Swapaxes here?
        elements = elements.reshape(sd.dim + 1, num_cells)

    return elements
