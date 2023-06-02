"""Homotopy continuations for the ``TwoPhaseFlow`` model."""

import logging
import time
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import porepy as pp
import torch

from tpf_lab.ml.nn import BaseNN
from tpf_lab.ml.nn_ad import nn_wrapper
from tpf_lab.models.two_phase_flow import (
    TwoPhaseFlowEquations,
    TwoPhaseFlowSolutionStrategy,
)
from tpf_lab.numerics.ad.functions import ad_pow
from tpf_lab.numerics.ad.functions import minimum as minimum_ad

logger = logging.getLogger(__name__)


class HomotopyContinuationRelPermEquations(TwoPhaseFlowEquations):
    _homotopy_continuation_param_ad: pp.ad.Scalar
    """Parameter for the homotopy continuation."""
    _rel_perm_w_init: Callable
    """Initial rel. perm. for the homotopy continuation. I.e., the rel. perm. equals
    this for :math:`k=1`.

    Provided by a mixin of type ``TwoPhaseFlowEquations``.

    """
    _rel_perm_w_goal: Callable
    """Goal rel. perm. for the homotopy continuation. I.e., the rel. perm. equals this
    for :math:`k=0`.

    Provided by a mixin of type ``TwoPhaseFlowEquations``.

    """

    _rel_perm_n_init: Callable
    """Initial rel. perm. for the homotopy continuation. I.e., the rel. perm. equals
    this for :math:`k=1`.

    Provided by a mixin of type ``TwoPhaseFlowEquations``.

    """
    _rel_perm_n_goal: Callable
    """Goal rel. perm. for the homotopy continuation. I.e., the rel. perm. equals this
    for :math:`k=0`.

    Provided by a mixin of type ``TwoPhaseFlowEquations``.

    """

    def _rel_perm_w(self) -> pp.ad.Operator:
        r"""Homotopy continuation wetting phase relative permeability."""
        return (
            self._homotopy_continuation_param_ad * self._rel_perm_w_init()
            + (pp.ad.Scalar(1) - self._homotopy_continuation_param_ad)
            * self._rel_perm_w_goal()
        )

    def _rel_perm_n(self) -> pp.ad.Operator:
        r"""Homotopy continuation nonwetting phase relative permeability."""
        return (
            self._homotopy_continuation_param_ad * self._rel_perm_n_init()
            + (pp.ad.Scalar(1) - self._homotopy_continuation_param_ad)
            * self._rel_perm_n_goal()
        )


class HomotopyContinuationRelPermSolutionStrategy(TwoPhaseFlowSolutionStrategy):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Parameters for the homotopy continuation.
        self._homotopy_continuation_param: float = 1
        self._homotopy_continuation_param_ad: pp.ad.Scalar = pp.ad.Scalar(
            self._homotopy_continuation_param
        )
        self._homotopy_continuation_param_min: float = params.get(
            "homotopy_continuation_param_min", 0.0
        )
        self._homotopy_continuation_decay: float = params.get(
            "homotopy_continuation_decay", 0.5
        )

    def before_nonlinear_loop(self) -> None:
        # Reset continuation parameter.
        self._homotopy_continuation_param = 1
        # Update ad homotopy continuation parameter.
        setattr(
            self._homotopy_continuation_param_ad,
            "_value",
            self._homotopy_continuation_param,
        )
        return super().before_nonlinear_loop()

    def before_nonlinear_iteration(self) -> None:
        return super().before_nonlinear_iteration()

    def after_nonlinear_iteration(self, solution: np.ndarray) -> None:
        # Decay continuation parameter.
        self._homotopy_continuation_param *= self._homotopy_continuation_decay
        if self._homotopy_continuation_param <= self._homotopy_continuation_param_min:
            self._homotopy_continuation_param = 0
        # Update ad homotopy continuation parameter.
        setattr(
            self._homotopy_continuation_param_ad,
            "_value",
            self._homotopy_continuation_param,
        )
        logger.info(
            f"Decayed homotopy_continuation_param to"
            + f" {self._homotopy_continuation_param:.2f}"
        )
        return super().after_nonlinear_iteration(solution)

    def check_convergence(
        self,
        solution: np.ndarray,
        prev_solution: np.ndarray,
        init_solution: np.ndarray,
        nl_params: dict[str, Any],
    ) -> tuple[float, bool, bool]:
        """Extend the convergence check of the super class s.t. it fails when only one
        nonlinear iteration has passed.

        This is to ensure, that the homotopy continuation problem gets solved instead of
        the problem at :math:`\lambda=1`, i.e. the initial problem of the homotopy
        continuation.

        Parameters:
            solution: Newly obtained solution vector prev_solution: Solution obtained in
            the previous non-linear iteration. init_solution: Solution obtained from the
            previous time-step. nl_params: Dictionary of parameters used for the
            convergence check.
                Which items are required will depend on the convergence test to be
                implemented.

        Returns:
            The method returns the following tuple:

            float:
                Error, computed to the norm in question.
            boolean:
                True if the solution is converged according to the test implemented by
                this method.
            boolean:
                True if the solution is diverged according to the test implemented by
                this method.

        """
        error, converged, diverged = super().check_convergence(
            solution, prev_solution, init_solution, nl_params
        )
        if self._nonlinear_iteration == 1:
            converged = False
        return error, converged, diverged


class HomotopyContinuationRelPermEquations_LineartoNN(
    HomotopyContinuationRelPermEquations
):
    _rel_perm_w_nn_function: Callable
    """Wetting rel. perm. function by a neural network.

    Provided by a mixin of type ``TwoPhaseFlowSolutionStrategy``."""

    def _rel_perm_w_init(self) -> pp.ad.Operator:
        r"""Linear wetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_w)
        rel_perm = s_normalized * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_w_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_w_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def _rel_perm_w_goal(self) -> pp.ad.Operator:
        r"""Machine learned wetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm = self._rel_perm_w_nn_function(s_normalized)
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_w_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_w_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm


class HomotopyContinuationRelPerm_LineartoNN_SolutionStrategy(
    HomotopyContinuationRelPermSolutionStrategy
):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # NN rel perm.
        relpermw_nn = BaseNN({"depth": 5, "hidden_size": 10, "final_act": "linear"})
        relpermw_nn.load_state_dict(torch.load("test.pt"))
        self._rel_perm_w_nn_function = pp.ad.Function(
            nn_wrapper(relpermw_nn), "rel perm w nn"
        )


class HomotopyContinuationRelPermEquations_LineartoPerturbatedCorey(
    HomotopyContinuationRelPermEquations
):
    """Wetting rel. perm. linear to wobbly. Nonwetting rel. perm linear to power."""

    _yscales: list[float]
    _xscales: list[float]
    _offsets: list[float]

    def _error_function_deriv(self) -> pp.ad.Operator:
        """Returns the derivative of the error function w.r.t. the saturation.

        This can be used to simulate perturbations in the cap. pressure and rel. perm.
        models.

        Returns:
            Derivative of the error function in terms of :math:`S_w`.
        """
        s = self.equation_system.md_variable(self.saturation_var)
        xscales = [pp.ad.Scalar(xscale) for xscale in self._xscales]
        yscales = [pp.ad.Scalar(yscale) for yscale in self._yscales]
        offsets = [pp.ad.Scalar(offset) for offset in self._offsets]
        exp_func = pp.ad.Function(pp.ad.functions.exp, "exp")
        square_func = pp.ad.Function(partial(ad_pow, exponent=2), "square")
        error = pp.ad.Scalar(0) * s
        for xscale, yscale, offset in zip(xscales, yscales, offsets):
            error = error + yscale * exp_func(
                pp.ad.Scalar(-1) * xscale * square_func(s - offset)
            )
        return error

    def _rel_perm_w_init(self) -> pp.ad.Operator:
        r"""Linear wetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_w)
        rel_perm = s_normalized * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_w_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_w_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def _rel_perm_w_goal(self) -> pp.ad.Operator:
        r"""Wobbly wetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_w)
        cube_func = pp.ad.Function(partial(ad_pow, exponent=3), "cube")
        rel_perm = cube_func(s_normalized) * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_w_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_w_max), "min"
            )
            rel_perm = minimum_func(maximum_func(rel_perm))
        return rel_perm + self._error_function_deriv()

    def _rel_perm_n_init(self) -> pp.ad.Operator:
        r"""Linear wonwetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_n)
        rel_perm = (pp.ad.Scalar(1) - s_normalized) * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_n_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_n_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def _rel_perm_n_goal(self) -> pp.ad.Operator:
        r"""Nonwetting phase relative permeability. Power model"""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_n)
        cube_func = pp.ad.Function(partial(ad_pow, exponent=3), "cube")
        rel_perm = cube_func(pp.ad.Scalar(1) - s_normalized) * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_n_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_n_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm


class HomotopyContinuationRelPermEquations_LineartoPower(
    HomotopyContinuationRelPermEquations
):
    def _rel_perm_w_init(self) -> pp.ad.Operator:
        r"""Linear wetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_w)
        rel_perm = s_normalized * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_w_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_w_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def _rel_perm_w_goal(self) -> pp.ad.Operator:
        r"""Wetting phase relative permeability. Power model"""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_w)
        cube_func = pp.ad.Function(partial(ad_pow, exponent=3), "cube")
        rel_perm = cube_func(s_normalized) * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_w_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_w_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def _rel_perm_n_init(self) -> pp.ad.Operator:
        r"""Linear wonwetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_n)
        rel_perm = (pp.ad.Scalar(1) - s_normalized) * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_n_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_n_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def _rel_perm_n_goal(self) -> pp.ad.Operator:
        r"""Nonwetting phase relative permeability. Power model"""
        s_normalized = self._s_normalized()
        rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_n)
        cube_func = pp.ad.Function(partial(ad_pow, exponent=3), "cube")
        rel_perm = cube_func(pp.ad.Scalar(1) - s_normalized) * rel_perm_linear_param
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_n_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum_ad, var_1=self._rel_perm_n_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm
