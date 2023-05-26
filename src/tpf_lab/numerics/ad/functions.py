"""
This module contains functions to be wrapped in a pp.ad.Function and used as part
of compound pp.ad.Operators, i.e. as (terms of) equations.

Some functions depend on non-ad objects. This requires that the function (f) be wrapped
in an ad Function using partial evaluation:

    from functools import partial
    AdFunction = pp.ad.Function(partial(f, other_parameter), "name")
    equation: pp.ad.Operator = AdFunction(var) - 2 * var

with var being some ad variable.

Note that while the argument to AdFunction is a pp.ad.Operator, the wrapping in
pp.ad.Function implies that upon parsing, the argument passed to f will be an AdArray.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import AdArray

__all__ = ["ad_pow", "minimum"]


def ad_pow(var, exponent: float):
    if isinstance(var, AdArray):
        if exponent >= 0:
            val = np.power(var.val, exponent)
        else:
            # Calculate the power expression explicitely with dtype=np.float64 to avoid
            # integer division.
            power = np.power(var.val, -exponent, dtype=np.float64)
            val = np.divide(
                1,
                power,
                out=np.zeros_like(power),
                where=power != 0,
                casting="unsafe",
            )
        if exponent - 1 >= 0:
            der = var._diagvec_mul_jac(exponent * np.power(var.val, exponent - 1))
        else:
            # Calculate the power expression explicitely with dtype=np.float64 to avoid
            # integer division (which can result in integer results).
            power = np.power(var.val, 1 - exponent, dtype=np.float64)
            der = var._diagvec_mul_jac(
                np.divide(
                    exponent,
                    power,
                    out=np.zeros_like(power),
                    where=power != 0,
                    casting="unsafe",
                )
            )
        return AdArray(val, der)
    else:
        if exponent >= 0:
            return np.power(var, exponent)
        else:
            power = np.power(var, -exponent, dtype=np.float64)
            return np.divide(
                1,
                power,
                out=np.zeros_like(power),
                where=power != 0,
                casting="unsafe",
            )


def minimum(var_0: AdArray, var_1: AdArray | np.ndarray) -> AdArray:
    """Ad minimum function represented as an ``AdArray``.

    The arguments can be either ``AdArrays`` or ``ndarrays``, this duality is needed to
    allow for parsing of operators that can be taken at the current iteration (in which
    case it will parse as an ``AdArray``) or at the previous iteration or time step (in
    which case it will parse as a ``ndarray``).


    Parameters:
        var_0: First argument to the minimum function.
        var_1: Second argument.

        If one of the input arguments is scalar, broadcasting will be used.


    Returns:
        The minimum of the two arguments, taken element-wise in the arrays. The return
        type is ``AdArray`` if at least one of the arguments is an ``AdArray``,
        otherwise it is an ``ndarray``. If an ``AdArray`` is returned, the Jacobian is
        computed according to the minimum values of the ``AdArrays`` (so if element
        ``i`` of the miminum is picked from ``var_0``, row ``i`` of the Jacobian is also
        picked from the Jacobian of ``var_0``). If ``var_0`` is an ``ndarray``, its
        Jacobian is set to zero.

    """
    # If neither var_0 or var_1 are ``AdArrays``, return the ``np.minimum`` function.
    if not isinstance(var_0, AdArray) and not isinstance(var_1, AdArray):
        # FIXME: According to the type hints, this should not be possible.
        return np.minimum(var_0, var_1)

    # Make a fall-back zero Jacobian for constant arguments.
    # EK: It is not clear if this is relevant, or if we filter out these cases with the
    # above parsing of ``np.ndarrays``. Keep it for now, but we should revisit once we
    # know clearer how the Ad-machinery should be used.
    zero_jac = 0
    if isinstance(var_0, AdArray):
        zero_jac = sps.csr_matrix(var_0.jac.shape)
    elif isinstance(var_1, AdArray):
        zero_jac = sps.csr_matrix(var_1.jac.shape)

    # Collect values and Jacobians.
    vals = []
    jacs = []
    for var in [var_0, var_1]:
        if isinstance(var, AdArray):
            v = var.val
            j = var.jac
        else:
            v = var
            j = zero_jac
        vals.append(v)
        jacs.append(j)

    # If both are scalar, return same. If one is scalar, broadcast explicitly
    if isinstance(vals[0], (float, int)):
        if isinstance(vals[1], (float, int)):
            # Both var_0 and var_1 are scalars. Treat vals as an ``ndarray`` to
            # return the minimum. The Jacobian of a scalar is 0.
            val = np.min(vals)
            return AdArray(val, 0)
        else:
            # ``var_0`` is a scalar, but ``var_1`` is not. Broadcast to shape of
            # ``var_1``.
            vals[0] = np.full_like(vals[1], vals[0])
    if isinstance(vals[1], (float, int)):
        # ``var_1`` is a scalar, but ``var_0`` is not (or else we would have hit the
        # return statement in the above double-if). Broadcast ``var_1`` to shape of
        # ``var_0``.
        vals[1] = np.full_like(vals[0], vals[1])

    # By now, we know that both vals are ``ndarrays``. Try to convince ``mypy`` that
    # this is the case.
    assert isinstance(vals[0], np.ndarray) and isinstance(vals[1], np.ndarray)
    # Minimum of the two arrays.
    # Note, that in the case of both values being equal, the jacobian from the second
    # input is taken.
    inds = (vals[1] <= vals[0]).nonzero()[0]

    min_val = vals[0].copy()
    min_val[inds] = vals[1][inds]
    # If both arrays are constant, a 0 matrix has been assigned to ``jacs``.
    # Return here to avoid calling copy on a number (immutable, no copy method) below.
    if isinstance(jacs[0], (float, int)):
        assert np.isclose(jacs[0], 0)
        assert np.isclose(jacs[1], 0)
        return AdArray(min_val, 0)

    # Start from ``var_0``, then change entries corresponding to inds.
    min_jac = jacs[0].copy()

    if isinstance(min_jac, sps.spmatrix):
        # We enforce matrix format ``csr`` s.t. ``merge_matrices`` gets passed matrices
        # of the same format.
        min_jac = min_jac.tocsr()
        lines = pp.matrix_operations.slice_mat(jacs[1].tocsr(), inds)
        pp.matrix_operations.merge_matrices(min_jac, lines, inds, min_jac.getformat())
    else:
        min_jac[inds] = jacs[1][inds]

    return AdArray(min_val, min_jac)


def bump_function(var, r_1: float, r_2: float):
    """Bump function needed, e.g., for partition of unity.

    The function equals :math:`y(x)=0` for :math:`x\leq r_1`,  :math:`y(x)=1` for
    :math:`x\geq r_2` and is smooth on the entire domain.

    .. warning:
        The function is unfinished at the moment.

    TODO: Finish the implementation.

    Parameters:
        var: _description_
        r_1: _description_
        r_2: _description_

    """
    if isinstance(var, AdArray):
        # Restrict the var to 1.
        # TODO restrict the var to 0 from below.
        # restrict_var = maximum(var, 1)
        # # add small epsilon to avoid divison by zero.
        # phi = exp(-1 / (1 + 1e-8 - pow(restrict_var, 2)))
        pass
    else:
        pass
