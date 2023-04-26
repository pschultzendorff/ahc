import numpy as np
import pytest
import scipy.sparse as sps
from porepy.numerics.ad.forward_mode import AdArray

from src.tpflab.numerics.ad import functions as af


# Function: pow
def test_pow_scalar():
    a = AdArray(np.array([3]), np.array([[0]]))
    assert a.val == 3 and a.jac == 0

    # Positive exponent
    b = af.pow(a, 3)
    assert b.val == np.power(3, 3) and b.jac == 0

    # Zero exponent
    b = af.pow(a, 0)
    assert b.val == np.power(3, 0) and b.jac == 0

    # Negative exponent
    b = af.pow(a, -3)
    assert b.val == 1 / np.power(3, 3) and b.jac == 0


def test_pow_advar():
    a = AdArray(np.array([2]), np.array([[3]]))
    assert a.val == 2 and a.jac == 3

    # Positive exponent
    b = af.pow(a, 3)
    assert b.val == np.power(2, 3) and b.jac == 3 * 3 * np.power(2, 2)

    # Zero exponent
    b = af.pow(a, 0)
    assert b.val == np.power(2, 0) and b.jac == 0

    # Negative exponent
    b = af.pow(a, -3)
    assert b.val == 1 / np.power(2, 3)
    assert b.val == 1 / np.power(2, 3) and b.jac == -3 * 3 * (
        np.divide(1, np.power(2, 4, dtype=np.float64))
    )


def test_pow_vector():
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    assert np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))

    # Positive exponent
    b = af.pow(a, 3)
    jac = (3 * np.diag(np.power(val, 2))) @ J
    assert np.all(b.val == np.power(val, 3)) and np.all(b.jac == jac)

    # Zero exponent
    b = af.pow(a, 0)
    jac = 0 * np.diag(np.power(val, 2))
    assert np.all(b.val == np.power(val, 0)) and np.all(b.jac == jac)

    # Negative exponent
    b = af.pow(a, -3)
    # Convert to diag. matrix after division, to avoid ``inf`` values from division
    # by zero.
    jac = np.diag(-3 / np.power(val, 4)) @ J
    assert np.all(b.val == 1 / np.power(val, 3)) and np.all(b.jac == jac)


def test_pow_sparse_jac():
    val = np.array([1, 2, 3])
    J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
    a = AdArray(val, J)

    # Positive exponent
    b = af.pow(a, 3)
    jac = (3 * np.diag(np.power(val, 2))) @ J.A
    assert np.all(b.val == np.power(val, 3)) and np.all(b.jac == jac)

    # Zero exponent
    b = af.pow(a, 0)
    jac = 0 * np.diag(np.power(val, 2))
    assert np.all(b.val == np.power(val, 0)) and np.all(b.jac == jac)

    # Negative exponent
    b = af.pow(a, -3)
    jac = np.diag(-3 / np.power(val, 4)) @ J.A
    assert np.all(b.val == 1 / np.power(val, 3)) and np.all(b.jac == jac)


def test_pow_scalar_times_ad_var():
    # Why use ``jac.A`` and ``np.allclose`` here?
    val = np.array([1, 2, 3])
    J = sps.diags(np.array([1, 1, 1]))
    a = AdArray(val, J)
    c = 2
    assert np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A)

    # Positive exponent
    b = af.pow(c * a, 3)
    # ? Without the ``c`` in the next line the test fails, although I am not
    # completely sure why it is needed. Probably something with the ad system, i.e.
    # the ``c`` is a constant and we need to use the chain rule. Weirdly enough, it
    # isn't needed in the test with a negative exponent.
    jac = (3 * c * sps.diags(np.power(c * val, 2))) @ J
    assert np.allclose(b.val, np.power(c * val, 3)) and np.allclose(b.jac.A, jac.A)

    # Zero exponent
    b = af.pow(c * a, 0)
    jac = 0 * sps.diags(np.power(c * val, 2))
    assert np.all(b.val == np.power(c * val, 0)) and np.allclose(b.jac.A, jac.A)

    # Negative exponent
    b = af.pow(c * a, -3)
    jac = sps.diags((-3 * c) / np.power(c * val, 4)) @ J
    assert np.all(b.val == 1 / np.power(c * val, 3)) and np.allclose(b.jac.A, jac.A)


def test_minimum_AdArray_scalar():
    """Test ``minimum`` for inputs of type ``AdArray`` and ``int``/``float``."""
    a = AdArray(np.array([3]), sps.csr_matrix([[3]]))

    # Second scalar is minimum.
    # As ``np.ndarray([0])`` does not have a jacobian, it's jacobian is set to zero.
    # The jacobian of the minimum is the rowwise jacobian corresponding to the rows of
    # the inputs, depending on which is the pointwise minimum.
    b = af.minimum(a, 0)
    assert np.all(b.val == np.minimum(np.array([3]), np.array([0])))
    assert np.all(b.jac == np.array([0]))

    # First scalar is minimum.
    b = af.minimum(a, 4)
    assert np.all(b.val == np.minimum(np.array([3]), np.array([4])))
    assert np.all(b.jac == np.array([3]))

    # Both scalars are equal. ``jac`` is taken from the second input.
    b = af.minimum(a, 3)
    assert np.all(b.val == np.minimum(np.array([3]), np.array([3])))
    assert np.all(b.jac == np.array([0]))


def test_minimum_ndarray_ndarray():
    """Test ``minimum`` for inputs of type ``np.ndarray`` and ``np.ndarray``.

    Note, that no jacobian is returned in this case.
    """
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])

    c = af.minimum(a, b)
    assert np.all(c == np.minimum(a, b))


def test_minimum_AdArray_ndarray():
    """Test ``minimum`` for inputs of type ``AdArray`` and ``np.ndarray``."""
    val_a = np.array([1, 2, 3])
    J_a = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val_a, sps.csc_matrix(J_a))

    b = np.array([3, 2, 1])
    # As ``np.ndarray`` does not have a jacobian, it's jacobian is set to zero.
    # The jacobian of the minimum corresponds to the rows of the inputs, depending on
    # which is the pointwise minimum. If both values are equal at a point, the row of
    # the jacobian equals the row of the second input.
    c = af.minimum(a, b)
    jac = np.array([[3, 2, 1], [0, 0, 0], [0, 0, 0]])
    assert np.all(c.val == np.minimum(val_a, b))
    assert np.all(c.jac.todense() == jac)


def test_minimum_AdArray_AdArray():
    """Test ``minimum`` for inputs of type ``AdArray`` and ``AdArray``."""
    val_a = np.array([1, 2, 3])
    J_a = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val_a, sps.csc_matrix(J_a))

    val_b = np.array([3, 2, 1])
    J_b = np.array([[1, 0, -1], [7, 8, 9], [1, 2, 3]])
    b = AdArray(val_b, sps.csc_matrix(J_b))

    # The jacobian of the minimum corresponds to the rows of the inputs, depending on
    # which is the pointwise minimum. If both values are equal at a point, the row of
    # the jacobian equals the row of the second input.
    c = af.minimum(a, b)
    jac = np.array([[3, 2, 1], [7, 8, 9], [1, 2, 3]])
    assert np.all(c.val == np.minimum(val_a, val_b))
    assert np.all(c.jac.todense() == jac)
