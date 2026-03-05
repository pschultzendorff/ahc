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
pp.ad.Function implies that upon parsing, the argument passed to f will be an Ad_array.
"""
from __future__ import annotations

from typing import Callable, Union

import numpy as np
import scipy.sparse as sps

import porepy_adaptions as pp
from porepy.numerics.ad.forward_mode import Ad_array

__all__ = ["pow"]


def pow(var, exponent: float):
    # TODO Write tests for this function.
    if isinstance(var, Ad_array):
        if exponent >= 0:
            val = np.power(var.val, exponent)
        else:
            val = 1 / np.power(var.val, -exponent)
        if exponent - 1 >= 0:
            der = var.diagvec_mul_jac(exponent * np.power(var.val, exponent - 1))
        else:
            power = np.power(var.val, 1 - exponent)
            der = var.diagvec_mul_jac(
                np.divide(
                    exponent,
                    power,
                    out=np.zeros_like(power),
                    where=power != 0,
                    casting="unsafe",
                )
            )
        return Ad_array(val, der)
    else:
        # TODO mypy gives an error, fix this!
        if exponent >= 0:
            return np.power(var, exponent)
        else:
            return np.power(var, -exponent)
