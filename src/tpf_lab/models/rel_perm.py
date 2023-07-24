"""Implementation of the Buckley-Leverett model in the fractional flow formulation."""

import logging
import math
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import porepy as pp
import scipy.sparse as sps
import sympy
import torch
from buckley_leverett import functions
from buckley_leverett import analytical_solution
from porepy.utils.examples_utils import VerificationUtils

from tpf_lab.ml.nn import BaseNN
from tpf_lab.ml.nn_ad import nn_wrapper
from tpf_lab.models.buckley_leverett import (
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettDefaultGeometry,
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettSolutionStrategy,
)
from tpf_lab.models.two_phase_flow import (
    TwoPhaseFlowEquations,
    TwoPhaseFlowSolutionStrategy,
    TwoPhaseFlowVariables,
)
from tpf_lab.numerics.ad.functions import ad_pow
from tpf_lab.numerics.ad.functions import minimum as minimum_ad
from tpf_lab.visualization.diagnostics import (
    BuckleyLeverettDataSaving,
    DiagnosticsMixinExtended,
)

# Setup logging.
logger = logging.getLogger(__name__)


class PerturbedRelPermEquations(TwoPhaseFlowEquations):
    _yscales: list[float]
    """_summary_
    
    
    
    Returns:
        _description_
    
    """
    _xscales: list[float]
    _offsets: list[float]

    def _rel_perm_w(self) -> pp.ad.Operator:
        """Add a perturbation to the wetting phase rel. perm."""
        rel_perm_w = super()._rel_perm_w()
        return rel_perm_w + self._error_function_deriv()

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


class PerturbedRelPermSolutionStrategy(TwoPhaseFlowSolutionStrategy):
    """Fetch the necessary parameters for the perturbation equations."""

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Parameters for the error function derivative:
        self._yscales: list[float] = params.get("yscales", [1.0])
        self._xscales: list[float] = params.get("xscales", [200])
        self._offsets: list[float] = params.get("offsets", [0.5])
        # Change flow function for the analytical solution.


class BuckleyLeverettPerturbedRelPermSolutionStrategy(
    BuckleyLeverettSolutionStrategy, PerturbedRelPermSolutionStrategy
):
    """Combines the BuckleyLeverettSolutionStrategy and
    PerturbedRelPermSolutionStrategy.

    Since the perturbations to rel. perm. change the analytical solution, the
    ``fractionalflow`` function of the analytical solver needs to be updated after
    initialization.

    """

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Reinitialize the fractional flow function of the analytical solver.
        self.analytical.fractionalflow = PerturbedRelPermFractionalFlowSympy(params)
        self.analytical.lambdify()


class BuckleyLeverettPerturbedRelPermSetup(  # type: ignore
    PerturbedRelPermEquations,
    TwoPhaseFlowVariables,
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettPerturbedRelPermSolutionStrategy,
    #
    BuckleyLeverettDefaultGeometry,
    #
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettDataSaving,
    VerificationUtils,
    DiagnosticsMixinExtended,
):
    ...


class PerturbedRelPermFractionalFlowSympy(functions.FractionalFlowSymPy):
    """Add the perturbed rel. perms. to the fractional flow function of the analytical
    solver."""

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        # Parameters for the error function derivative:
        self.yscales: list[float] = params.get("yscales", [1.0])
        self.xscales: list[float] = params.get("xscales", [200])
        self.offsets: list[float] = params.get("offsets", [0.5])

    def lambda_w(self):
        r"""Wetting phase mobility.

        Power model
        .. math::
            k_{r,w}(S_w)=S_w^3 + \epsilon(S_w)

        """
        return self.S_normalized() ** 3 + self.error_function_deriv()

    def error_function_deriv(self):
        return sympy.Add(
            *[
                yscale * sympy.exp(-xscale * (self.S_w - offset) ** 2)
                for xscale, yscale, offset in zip(
                    self.xscales, self.yscales, self.offsets
                )
            ]
        )


class RelPermNNEquations(TwoPhaseFlowEquations):
    """Mixin that provides relative permeabilities given by neural networks."""

    _rel_perm_w_nn_function: Callable
    """Wetting rel. perm. function by a neural network.

    Provided by a mixin of type ``TwoPhaseFlowSolutionStrategy``.

    """
    _rel_perm_n_nn_function: Callable
    """Nonwetting rel. perm. function by a neural network.

    Provided by a mixin of type ``TwoPhaseFlowSolutionStrategy``.

    """

    def _rel_perm_w(self) -> pp.ad.Operator:
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

    def _rel_perm_n(self) -> pp.ad.Operator:
        r"""Machine learned wetting phase relative permeability."""
        s_normalized = self._s_normalized()
        rel_perm = self._rel_perm_n_nn_function(s_normalized)
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


class RelPermNNSolutionStrategy(pp.SolutionStrategy):
    """Load architecture and parameters for neural networks that provide relative
    permeabilities."""

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # NN rel perm.
        rel_perm_w_nn = BaseNN(params.get("rel_perm_w_nn_params", {}))
        rel_perm_w_nn.load_state_dict(
            torch.load(params.get("rel_perm_w_nn_path", "test.pt"))
        )
        self._rel_perm_w_nn_function = pp.ad.Function(
            nn_wrapper(rel_perm_w_nn), "rel perm w nn"
        )

        rel_perm_n_nn = BaseNN(params.get("rel_perm_n_nn_params", {}))
        rel_perm_n_nn.load_state_dict(
            torch.load(params.get("rel_perm_n_nn_path", "test.pt"))
        )
        self._rel_perm_n_nn_function = pp.ad.Function(
            nn_wrapper(rel_perm_n_nn), "rel perm n nn"
        )


class RelPermNNFractionalFlowNumpy(functions.FractionalFlowNumPy):
    """Add the neural network rel. perms. to the fractional flow function of the
    analytical solver."""

    def __init__(self, params: dict[str, Any]) -> None:
        super().__init__(params)
        # NN rel perm.
        self.rel_perm_w_nn = BaseNN(params.get("rel_perm_w_nn_params", {}))
        self.rel_perm_w_nn.load_state_dict(
            torch.load(params.get("rel_perm_w_nn_path", "test.pt"))
        )

        self.rel_perm_n_nn = BaseNN(params.get("rel_perm_n_nn_params", {}))
        self.rel_perm_n_nn.load_state_dict(
            torch.load(params.get("rel_perm_n_nn_path", "test.pt"))
        )

    def lambda_w(self, S_w: np.ndarray) -> np.ndarray:
        r"""Wetting phase mobility provided by a neural network.

        In practice, the network provides the rel. perm. and phase viscosity is assumed
        to be 1.

        """
        return self.rel_perm_w_nn(torch.from_numpy(S_w)).numpy()

    def lambda_n(self, S_w: np.ndarray) -> np.ndarray:
        r"""Nonwetting phase mobility provided by a neural network.

        In practice, the network provides the rel. perm. and phase viscosity is assumed
        to be 1.

        """
        return self.rel_perm_n_nn(torch.from_numpy(S_w)).numpy()


class RelPermNNBuckleyLeverettAnalyticalSolution(analytical_solution.BuckleyLeverett):
    """Replace the ``sympy`` fractional flow function by a ``numpy`` fractional flow
    function to allow the implementation of ``torch.nn.Module`` as functions.

    This allows for computation of the analytical solution of a Buckley-Leverett
    problem that has neural networks in its fractional flow form.

    TODO: This is not a simple task, as the first and second deriatives of the
    ``torch.nn.Module`` needs to be constructed and translated to functions that handle
    ``numpy.ndarray``. Not implemented yet.

    """

    ...

    # def __init__(self, params: Optional[dict[str, Any]]) -> None:
    #     super.__init__(params)
    #     if params is None:
    #         params = {}
    #     self.fractionalflow: functions.FractionalFlowNumPy = (
    #         functions.FractionalFlowNumPy(params)
    #     )

    # def lambdify(self) -> None:
    #     """Calculate the derivative of the fractional flow function.""""
    #     self.total_flow = sp.lambdify(
    #         self.fractionalflow.S_w,
    #         self.fractionalflow.total_flow(),
    #         "numpy",
    #     )
    #     self.total_flow_prime = sp.lambdify(
    #         self.fractionalflow.S_w,
    #         self.fractionalflow.total_flow_prime(),
    #         "numpy",
    #     )

    #     self.total_flow_prime_prime = sp.lambdify(
    #         self.fractionalflow.S_w,
    #         self.fractionalflow.total_flow_prime_prime(),
    #         "numpy",
    #     )
