
import logging
from typing import Any, Callable, Literal, Optional

import numpy as np
import porepy as pp
import sympy as sym
from src.utils import get_quadpy_elements, interpolate_p1, interpolate_p2, poly2col
from src.quadrature import TriangleQuadrature
import quadpy

logger = logging.getLogger(__name__)


class HCPermeability:
    """

    TODO: Have two different fluxes and add them (weighted) together instead of a
    permeability that changes all the time. -> The system does not need to be assembled
    at each continuation iteration.

    """
    _hc_param_ad: pp.ad.Scalar
    """Homotopy continuation parameter."""
    perm_init: Callable[[], ]
    """Initial permeability for the homotopy continuation.

    Provided by a mixin of type ``TwoPhaseFlowEquations``.

    """
    solid: pp.SolidConstants
    """Solid constant object that takes care of scaling of solid-related quantities. The
    goal permeability of the homotopy continuation is defined in `solid.permeability()`.

    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    # TODO: Difference between :class:`~...` and :class:`...`? 
    """
    isotropic_second_order_tensor: Callable[
        [list[pp.Grid], pp.ad.Operator], pp.ad.Operator
    ]
    """Basis for the local coordinate system. Normally set by a mixin instance of
    :class:`porepy.models.constitutive_laws.SecondOrderTensorUtils`.

    """

    _final_perm: np.ndarray 
    """The goal permeability tensor of the homotopy continuation.

    Normally, this is set by a mixin of instance
    :class:`~porepy.models.solution_strategy.SolutionStrategy`.

    # TODO: Difference between :class:`~...` and :class:`...`? 
    """


    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2]. HC weighted sum of isotropic unit permeability and
        (anisotropic) permeability of the solid.

        The permeability is quantity which enters the discretized equations in a form
        that cannot be differentiated by Ad (this is at least true for a subset of the
        relevant discretizations). For this reason, the permeability is not returned as
        an Ad operator, but as a numpy array, to be wrapped as a SecondOrderTensor and
        passed as a discretization parameter.

        Parameters:
            subdomains: Subdomains where the permeability is defined.

        Returns:
            Cell-wise permeability tensor. Dependent on HC parameter and solid
                constants.

        """
        size = sum(sd.num_cells for sd in subdomains)
        perm_init = pp.wrap_as_dense_ad_array(
            pp.ad.Scalar(1), size, name="permeability init"
        )
        perm_final = pp.wrap_as_dense_ad_array(
            self.solid.permeability(), size, name="permeability final"
        )
        permeability = self._hc_param_ad * perm_init + (pp.ad.Scalar(1) - self._hc_param_ad) * perm_final
        # TODO: This will not work with anisotropic permeability. Use a
        # SecondOrderTensor instead and transform into an ad object.
        # pp.SecondOrderTensor(*permeability.flatten())
        return self.isotropic_second_order_tensor(subdomains, permeability)



class HCPermeabilitySolutionStrategy(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):
    
        transfer_solution: Callable
        """Provided by a mixin of type :class:`Postprocessing`."""
        extend_normal_fluxes: Callable
        """Provided by a mixin of type :class:`Postprocessing`."""
        reconstruct_potentials: Callable
        """Provided by a mixin of type :class:`Postprocessing`."""
        solid: pp.SolidConstants
        permeability: Callable

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        # Parameters for the homotopy continuation.
        self._hc_param: float = 1.0
        self._hc_param_ad: pp.ad.Scalar = pp.ad.Scalar(self._hc_param)
        self._hc_param_min: float = params.get("hc_param_min", 0.0)
        self._hc_decay: float = params.get("hc_decay", 0.5)
        self._hc_error_ratio: float = params.get("hc_error_ratio", 0.05)
        """Defines the ratio between the homotopy continuation error and the
        discretization errror at which the Newton homotopy continuation is stopped."""

        self.residuals_wrt_homotopy: list[float] = []
        """Store the residuals of the equation w.r.t. the current homotopy iteration."""
        self.residuals_wrt_goal_function: list[float] = []
        """Store the residuals of the equation w.r.t. the goal system, i.e., w.r.t.
        :math:`\lambda = 0`."""

        # Permeability.
        dim: int = self.params.get("dim", 2)
        perm: np.ndarray = self.params.get("permeability", np.eye(dim))
        assert perm.ndim == 2 and perm.shape[0] == perm.shape[1], f"Permeability matrix not square."
        assert perm.shape[0] == dim, f"Permeability matrix has the wrong {perm.shape[0]} dimension."
        assert perm == perm.T, "Permeability matrix not symmetric."
        self._final_perm: np.ndarray = perm
        # TODO: Why does mypy complain?

    def before_nonlinear_loop(self) -> None:
        """Reset HC parameter and residuals."""
        # Reset continuation parameter.
        self._hc_param = 1.0

        # Update ad homotopy continuation parameter.
        setattr(
            self._hc_param_ad,
            "_value",
            self._hc_param,
        )
        # Reset residuals arrays.
        self.residuals_wrt_homotopy = []
        self.residuals_wrt_goal_function = []
        return super().before_nonlinear_loop()

    def before_nonlinear_iteration(self) -> None:
        return super().before_nonlinear_iteration()

    def after_nonlinear_iteration(self, solution: np.ndarray) -> None:
        """Decay HC parameter and compute residuals."""
        # Compute residual (PorePys residual is actually the norm of the solution).
        b = self.linear_system[1]
        self.residuals_wrt_homotopy.append(float(np.linalg.norm(b)) / np.sqrt(b.size))
    
        # Set homotopy continuation param to zero, compute the residual and reset.
        setattr(
            self._hc_param_ad,
            "_value",
            0,
        )
        self.assemble_linear_system()
        b = self.linear_system[1]
        self.residuals_wrt_goal_function.append(
            float(np.linalg.norm(b)) / np.sqrt(b.size)
        )

        # Decay continuation parameter.
        self._hc_param *= self._hc_decay
        if self._hc_param <= self._hc_param_min:
            self._hc_param = 0
        # Update ad homotopy continuation parameter.
        setattr(
            self._hc_param_ad,
            "_value",
            self._hc_param,
        )
        logger.info(
            f"Decayed hc_param to"
            + f" {self._hc_param:.2f}"
        )
        # TODO: Include convergence criteria for NHC -> Is this already included in
        # super?
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

    def stopping_criteria(self) -> None:
        """Calculate error estimators and stop if NHC error is small enough."""
        self.transfer_solution()
        self.extend_normal_fluxes()
        self.reconstruct_potentials()
        hc_error = self.HC_error_est()
        discr_error = self.discr_error_est()
        if hc_error <= self._hc_error_ratio * discr_error:
            logger.info(f"HC error {hc_error} smaller than {self._hc_error_ratio} * "
                        f"discr_error {discr_error}. Stopping homotopy continuation.")
            self.is_converged = True
        else:
            self.is_converged = False

    def HC_error_est(self) -> None:
        """Calculate homotopy continuation error estimator.

        Note: For now, we assume homogeneous Dirichlet and Neuman bc.

        """
        # TODO: Write this function.
        pass

    def discr_error_est(self) -> None:
        """Calculate discretization error estimator.

        Note: For now, we assume homogeneous Dirichlet and Neuman bc. The error is then
        bounded by the sum of the residual estimator, the flux estimator, and the
        nonconformity estimator. As we use cell-centered finite volumes, the flux
        estimator is zero.

        """
        # TODO: Write this function.
        pass

    def residual_est(self) -> None:
        """
        TODO: Is this always zero for CCFV as suggested by chapter 8.4.2 of M. Vohralík,
        “A posteriori error estimates for efficiency and error control in numerical
        simulations”?

        """
        # TODO: Write this function.
        pass

    def nonconformity_est(self, norm: Literal["iteration", "final"]) -> None:
        # Retrieve subdomain and data.
        sd = self.mdg.subdomains()[0]
        d = self.mdg.subdomain_data(sd)

        # Retrieve cell-centered solution.
        pcc = d["estimates"]["fv_sd_pressure"]
        pcc = np.reshape(pcc, (sd.num_cells, 1))
        pccr = poly2col(pcc)

        # Retrieve reconstructed pressures.
        recon_p = d["estimates"]["p_recon_rt0_p1"]
        pr = poly2col(recon_p)

        # Cell centered and reconstructed pressure gradients.

        def integrand_l2_error(x, norm: Literal["iteration", "final"]):
            # Cell centered and reconstructed pressure gradient in x and y.
            # TODO: Extend to 3D.
            ones = np.ones_like(x[0])
            gradp_cc_x = pccr[0] * ones
            gradp_cc_y = pccr[1] * ones
            gradp_recon_x = pr[0] * ones
            gradp_recon_y = pr[1] * ones

            # Integral in x and y
            perm = self.permeability() if norm == "iteration" else self.solid.permeability()
            # TODO: This does not work correctly, because perm is a tensor but
            # gradp_cc_x is a scalar. -> 
            term_1_x = perm * (gradp_cc_x - gradp_recon_x)
            term_1_y = perm * (gradp_cc_y - gradp_recon_y)
            int_x = term_1_x * (gradp_cc_x - gradp_recon_x)
            int_y = term_1_y * (gradp_cc_y - gradp_recon_y)
            
            return int_x + int_y

        # Obtain elements and declare integration method
        method = TriangleQuadrature()
        elements = get_quadpy_elements(sd)

        integral = method.integrate(integrand_l2_error, elements)
        d["estimates"][f"local_nc_est_{norm}"] = integral
        d["estimates"][f"nc_est_{norm}"] = integral.sum() ** 0.5

    def total_error(self) -> None:
        """Calculate total error by comparing with the exact solution."""
        # TODO: Write this function.
        pass


