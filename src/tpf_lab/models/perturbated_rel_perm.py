"""Implementation of the Buckley-Leverett model in the fractional flow formulation."""

import logging
import math
from functools import partial
from typing import Any

import numpy as np
import porepy as pp
import sympy
import scipy.sparse as sps
from buckley_leverett import functions
from porepy.utils.examples_utils import VerificationUtils

from tpf_lab.models.two_phase_flow import (
    TwoPhaseFlowEquations,
    TwoPhaseFlowSolutionStrategy,
    TwoPhaseFlowVariables,
)
from tpf_lab.models.buckley_leverett import (
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettSolutionStrategy,
    BuckleyLeverettDefaultGeometry,
    BuckleyLeverettSemiAnalyticalSolution,
)
from tpf_lab.numerics.ad.functions import ad_pow
from tpf_lab.visualization.diagnostics import (
    BuckleyLeverettDataSaving,
    DiagnosticsMixinExtended,
)

# Setup logging.
logger = logging.getLogger(__name__)


class PerturbatedRelPermEquations(TwoPhaseFlowEquations):
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


class PerturbatedRelPermSolutionStrategy(TwoPhaseFlowSolutionStrategy):
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


class BuckleyLeverettPerturbatedRelPermSolutionStrategy(
    BuckleyLeverettSolutionStrategy, PerturbatedRelPermSolutionStrategy
):
    """Combines the BuckleyLeverettSolutionStrategy and
    PerturbatedRelPermSolutionStrategy.

    Since the perturbations to rel. perm. change the analytical solution, the
    ``fractionalflow`` function of the analytical solver needs to be updated after
    initialization.

    """

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Reinitialize the fractional flow function of the analytical solver.
        self.analytical.fractionalflow = PerturbatedRelPermFractionalFlowSympy(params)
        self.analytical.lambdify()


class BuckleyLeverettPerturbatedRelPermSetup(  # type: ignore
    PerturbatedRelPermEquations,
    TwoPhaseFlowVariables,
    BuckleyLeverettBoundaryConditions,
    PerturbatedRelPermSolutionStrategy,
    #
    BuckleyLeverettDefaultGeometry,
    #
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettDataSaving,
    VerificationUtils,
    DiagnosticsMixinExtended,
):
    ...


class PerturbatedRelPermFractionalFlowSympy(functions.FractionalFlowSymPy):
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
