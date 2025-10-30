"""Mixin to approximate tangent and curvature of a homotopy curve.


D. A. Brown and D. W. Zingg, “Efficient numerical differentiation of implicitly-defined
      curves for sparse systems,” Journal of Computational and Applied Mathematics, vol.
      304, pp. 138–159, Oct. 2016, doi: 10.1016/j.cam.2016.03.002.

"""

import numpy as np
import porepy as pp
import scipy.sparse as sp

from tpf.models.protocol import (
    HCProtocol,
    TPFProtocol,
)


class CurvatureMixin(HCProtocol, TPFProtocol):
    def __init__(self):
        self._tangent = None
        self._curvature = None

    def tangent(self) -> np.ndarray:
        """Exactly .

        Returns
        -------
        np.ndarray
            The curvature at each point on the homotopy curve.
        """
        # Evaluate the residual with :math:`\lambda = 1` and :math:`\lambda = 0`.
        self.nonlinear_solver_statistics.hc_lambda_ad.set_value(1.0)
        residual_g: np.ndarray = self.equation_system.assemble(evaluate_jacobian=False)
        self.nonlinear_solver_statistics.hc_lambda_ad.set_value(0.0)
        residual_f: np.ndarray = self.equation_system.assemble(evaluate_jacobian=False)
        # Restore lambda to the current value.
        self.nonlinear_solver_statistics.hc_lambda_ad.set_value(
            self.nonlinear_solver_statistics.hc_lambda_fl
        )

        rhs: np.ndarray = residual_g - residual_f
        jacobian: sp.spmatrix = self.equation_system.assemble(evaluate_jacobian=True)[0]

        tangent: np.ndarray = sp.linalg.spsolve(jacobian, rhs)
        assert isinstance(tangent, np.ndarray)
        return tangent

    def curvature(self) -> np.ndarray:
        """Approximate curvature with finite differences"""
        pass

    def after_nonlinear_convergence(self) -> None:
        """Compute tangent and curvature after nonlinear convergence."""
        super().after_nonlinear_convergence()
        self.nonlinear_solver_statistics.log_error(
            tangent=self.tangent(), curvature=self.curvature()
        )
