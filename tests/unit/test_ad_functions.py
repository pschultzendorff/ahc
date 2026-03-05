import unittest

import numpy as np
import scipy.sparse as sps

from porepy_adaptions.numerics.ad import functions as af
from porepy.numerics.ad.forward_mode import Ad_array


class AdFunctionTest(unittest.TestCase):

    # Function: pow
    def test_pow_scalar(self):
        # Positive exponent
        a = Ad_array(3, 0)
        b = af.pow(a, 3)
        self.assertTrue(b.val == np.power(3, 3) and b.jac == 0)
        self.assertTrue(a.val == 3 and a.jac == 0)
        # Zero exponent
        b = af.pow(a, 0)
        self.assertTrue(b.val == np.power(3, 0) and b.jac == 0)
        # Negative exponent
        b = af.pow(a, -3)
        self.assertTrue(b.val == 1 / np.power(3, 3) and b.jac == 0)

    def test_pow_advar(self):
        # Positive exponent
        a = Ad_array(2, 3)
        b = af.pow(a, 3)
        self.assertTrue(b.val == np.power(2, 3) and b.jac == 3 * 3 * np.power(2, 2))
        self.assertTrue(a.val == 2 and a.jac == 3)
        # Zero exponent
        b = af.pow(a, 0)
        self.assertTrue(b.val == np.power(2, 0) and b.jac == 0)
        # Negative exponent
        b = af.pow(a, -3)
        self.assertTrue(
            b.val == 1 / np.power(2, 3) and b.jac == -3 * 3 * (1 / np.power(2, 4))
        )

    def test_pow_vector(self):
        # Positive exponent
        val = np.array([1, 2, 3])
        J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
        a = Ad_array(val, sps.csc_matrix(J))
        b = af.pow(a, 3)
        jac = (3 * np.diag(np.power(val, 2))) @ J

        self.assertTrue(np.all(b.val == np.power(val, 3)) and np.all(b.jac == jac))
        self.assertTrue(np.all(J == np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])))
        # Zero exponent
        b = af.pow(a, 0)
        jac = 0 * np.diag(np.power(val, 2))

        self.assertTrue(np.all(b.val == np.power(val, 0)) and np.all(b.jac == jac))
        # Negative exponent
        b = af.pow(a, -3)
        power = np.diag(np.power(val, 4))
        jac = (
            np.divide(
                -3, power, np.zeros_like(power), where=power != 0, casting="unsafe"
            )
            @ J
        )
        self.assertTrue(np.all(b.val == 1 / np.power(val, 3)) and np.all(b.jac == jac))

    def test_pow_sparse_jac(self):
        # Positive exponent
        val = np.array([1, 2, 3])
        J = sps.csc_matrix(np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]]))
        a = Ad_array(val, J)
        b = af.pow(a, 3)
        jac = (3 * np.diag(np.power(val, 2))) @ J.A
        self.assertTrue(np.all(b.val == np.power(val, 3)) and np.all(b.jac == jac))
        # Zero exponent
        b = af.pow(a, 0)
        jac = 0 * np.diag(np.power(val, 2))

        self.assertTrue(np.all(b.val == np.power(val, 0)) and np.all(b.jac == jac))
        # Negative exponent
        b = af.pow(a, -3)
        power = np.diag(np.power(val, 4))
        jac = (
            np.divide(
                -3, power, np.zeros_like(power), where=power != 0, casting="unsafe"
            )
            @ J.A
        )
        self.assertTrue(np.all(b.val == 1 / np.power(val, 3)) and np.all(b.jac == jac))

    def test_pow_scalar_times_ad_var(self):
        # Why use ``jac.A`` and ``np.allclose`` here?
        # Positive exponent
        val = np.array([1, 2, 3])
        J = sps.diags(np.array([1, 1, 1]))
        a = Ad_array(val, J)
        c = 2
        b = af.pow(c * a, 3)
        # ? Without the ``c`` in the next line the test fails, although I am not
        # completely sure why it is needed. Probably something with the ad system, i.e.
        # the ``c`` is a constant and we need to use the chain rule. Weirdly enough, it
        # isn't needed in the test with a negative exponent.
        jac = (3 * c * sps.diags(np.power(c * val, 2))) @ J
        self.assertTrue(
            np.allclose(b.val, np.power(c * val, 3)) and np.allclose(b.jac.A, jac.A)
        )
        self.assertTrue(np.all(a.val == [1, 2, 3]) and np.all(a.jac.A == J.A))
        # Zero exponent
        b = af.pow(c * a, 0)
        jac = 0 * sps.diags(np.power(c * val, 2))

        self.assertTrue(
            np.all(b.val == np.power(c * val, 0)) and np.allclose(b.jac.A, jac.A)
        )
        # Negative exponent
        b = af.pow(c * a, -3)
        power = np.power(c * val, 4)
        jac = sps.diags(
            np.divide(
                -3 * c, power, np.zeros_like(power), where=power != 0, casting="unsafe"
            )
            @ J.A
        )
        self.assertTrue(
            np.all(b.val == 1 / np.power(c * val, 3)) and np.allclose(b.jac.A, jac.A)
        )
