r"""This module contains an implementation of a base model for two-phase flow problems.

Both the nonwetting pressure-wetting saturation formulation
    .. math::
        -\nabla\cdot\left(\lambda_t\nabla p_n - \lambda_n\nabla p_c
        - \lambda_w\nabla\rho_w\bm{g} - \lambda_n\nabla\rho_n\bm{g}\right) = \bm{q}_t,\\
        \phi\frac{\partial S_w}{\partial t} + \nabla\cdot\left(f_w\bm{u}
        + f_w\lambda_n\nabla(p_c + \Delta\rho\bm{g})\right) = \bm{q}_w,

as well as the wetting pressure-wetting saturation formulation

    .. math::
        -\nabla\cdot\left(\lambda_t\nabla p_w + \lambda_n\nabla p_c
        - \lambda_w\nabla\rho_w\bm{g} - \lambda_n\nabla\rho_n\bm{g}\right) = \bm{q}_t,\\
        \phi\frac{\partial S_w}{\partial t}
        - \nabla\cdot\left(\lambda_w\nabla p_w - \lambda_w\nabla\rho_w\bm{g}\right)
        = \bm{q}_w,

are implemented.

Furthermore, multiple different models for both the capillary pressure, as well as the
relative permeability are implemented.

TODO:
    - Change bc_values to `ad.BoundaryCondition`
    - Make the model suitable for multi-grid domains.
    
Units:
    Collection of the SI units for all parameters.
    NOTE: It was not thoroughly checked, whether all units are correct.
    saturation: dimensionless
    pressure: pascal=kg/(m*s^2)S
    density: kg/m^3
    viscosity: kg/(m*s)
    permeability: m^2
    volumetric source terms: m^3/s
    mass flux: kg/s -> Not needed at the moment
"""

from __future__ import annotations

import logging
import time
from functools import partial
from typing import Callable, Literal, Optional

import numpy as np
import porepy as pp

from tpf_lab.numerics.ad.functions import ad_pow as ad_pow
from tpf_lab.numerics.ad.functions import minimum

logger = logging.getLogger(__name__)


class TwoPhaseFlowEquations:
    """This is a model class for two-phase flow problems.

    This class is intended to provide a standardized setup, with all discretizations
    in place and reasonable parameter and boundary values. The intended use is to
    inherit from this class, and do the necessary modifications and specifications
    for the problem to be fully defined. The minimal adjustment needed is to
    specify the method create_grid(). The class also serves as parent for other
    model classes (CompressibleFlow).

    Public attributes:
        primary_pressure_var: Name assigned to the pressure variable in the
            highest-dimensional subdomain. Will be used throughout the simulations,
            including in ParaView export. The default variable name is "wetting
            pressure" or "nonwetting pressure", depending on the chosen two-phase flow
            formulation.
        parameter_key: Keyword used to define parameters and discretizations.
        params (dict): Dictionary of parameters used to control the solution procedure.
            Some frequently used entries are file and folder names for export, mesh
            sizes...
        mdg (pp.MixedDimensionalGrid): Mixed-dimensional grid. Should be set by a method
            create_grid, which should be provided by the user.
        convergence_status (bool): Whether the non-linear iteration has converged.
        linear_solver (str): Specification of linear solver. Only known permissible
            value is 'direct'
        exporter (pp.Exporter): Used for writing files for visualization.

    All attributes are given natural values at initialization of the class.

    The implementation assumes use of AD.

    """

    # Managers and exporter:
    equation_system: pp.ad.EquationSystem
    time_manager: pp.TimeManager

    # Parameter keys
    params_key: str
    """Provided by ``TwoPhaseFlowSolutionStrategy``"""
    w_flux_key: str
    """Provided by ``TwoPhaseFlowSolutionStrategy``"""
    n_flux_key: str
    """Provided by ``TwoPhaseFlowSolutionStrategy``"""
    cap_flux_key: str
    """Provided by ``TwoPhaseFlowSolutionStrategy``"""
    mass_key: str
    """Provided by ``TwoPhaseFlowSolutionStrategy``"""

    _formulation: str
    # Variables:
    pressure_w_var: str
    pressure_n_var: str
    saturation_var: str
    primary_pressure_var: str
    secondary_pressure_var: str

    # Phase parameters:
    _viscosity_w: float
    _viscosity_n: float
    _density_w: float
    _density_n: float

    # Residual saturations:
    _residual_saturation_w: float
    _residual_saturation_n: float

    # Model selection and parameters for the capillary pressure function:
    _cap_pressure_model: Literal["Brooks-Corey", "van Genuchten", "linear", None]
    # van Genuchten model
    _n_g: float
    _m_g: float
    _beta_g: float
    # Brooks-Corey model
    _entry_pressure: float
    _n_b: int
    # linear model
    _cap_pressure_linear_param: float

    # Model selection and parameters for the relative permeability function:
    _rel_perm_model: Literal["Brooks-Corey", "Corey", "power", "linear"]
    # Brooks-Corey model
    _n1: int
    _n2: int
    _n3: int
    # Power law (Corey model)
    _rel_perm_power: int
    _rel_perm_linear_param_w: float
    _rel_perm_linear_param_n: float
    # Lower and upper limits for the rel.perm
    _limit_rel_perm: bool
    _rel_perm_w_max: float
    _rel_perm_w_min: float
    _rel_perm_n_max: float
    _rel_perm_n_min: float

    # Parameters for the error function derivative:
    _yscale: float
    _xscale: float
    _offset: float

    mdg: pp.MixedDimensionalGrid
    """Provided by a mixin of type ``ModelGeometry``."""

    # Boundary conditions:
    _bc_type_pressure_w: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_type_pressure_n: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_type_pressure_c: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_values_mobility_w: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_values_mobility_n: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _dirichlet_bc_values_pressure_w: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _dirichlet_bc_values_pressure_n: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_values_cap_pressure: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _neumann_bc_values_flux_w: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _neumann_bc_values_flux_n: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""

    def _source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting source.

        In the default model there is no source term.

        NOTE: This is the average value per grid cell, i.e., it gets scaled with the
        cell volume in the equation.

        SI Units: m^d/s

        """
        return np.zeros(g.num_cells)

    def _source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting source.

        SI Units: m^d/s

        NOTE: This is the average value per grid cell, i.e., it gets scaled with the
        cell volume in the equation.

        """
        return np.zeros(g.num_cells)

    def _source_t(self, g: pp.Grid) -> np.ndarray:
        """Volumetric total source; sum of the wetting and nonetting source.

        SI Units: m^d/s

        NOTE: This is the average value per grid cell, i.e., it gets scaled with the
        cell volume in the equation.

        """
        return self._source_w(g) + self._source_n(g)

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting vector source. Corresponds to the wetting buoyancy flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = pp.GRAVITY_ACCELERATION * self._w_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        # vals[-1] = pp.GRAVITY_ACCELERATION * self._density_w
        # For some reason this needs to be a flat array.
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric nonwetting vector source. Corresponds to the nonwetting buoyancy
        flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = pp.GRAVITY_ACCELERATION * self._n_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        # vals[-1] = pp.GRAVITY_ACCELERATION * self._density_n
        # For some reason this needs to be a flat array.
        return vals.ravel()

    # More matrix and phase parameters.
    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Solid permeability.

        SI Units: m^2
        """
        return np.full(g.num_cells, 1.0)

    def _porosity(self, g: pp.Grid) -> np.ndarray:
        return np.full(g.num_cells, 1.0)

    # Cap pressure and relative permeability functions.
    def _s_normalized(self) -> pp.ad.Operator:
        r"""Normalize the wetting saturation by the residual saturations.

        .. math::
            \hat{S}_w=\frac{S_w - S_w^{min}}{S_w^{max} - S_w^{min}},

        which is equal to

        .. math::
            \hat{S}_w=\frac{S_w - S_{w,res}}{1 - S_{n,res}  S_{w,res}}.


        Returns:
            s_normalized: Normalized wetting saturation.

        """
        s = self.equation_system.md_variable(self.saturation_var)
        residual_saturation_w = pp.ad.Scalar(self._residual_saturation_w)
        residual_saturation_n = pp.ad.Scalar(self._residual_saturation_n)
        s_normalized = (s - residual_saturation_w) / (
            pp.ad.Scalar(1) - residual_saturation_n - residual_saturation_w
        )
        return s_normalized

    def _cap_pressure(self) -> pp.ad.Operator:
        r"""Capillary pressure function.

        The following three models are implemented:

        Brooks-Corey model
        .. math::
            p_c(\hat{S}_w)=p_e\hat{S}_w^{n_b}

        Linear model
        .. math::
            p_c(\hat{S}_w)=c\hat{S}_w

        van Genuchten model
        .. math::
            p_c(\hat{S}_w)=\frac{(\hat{S}_w^{m_g}-1)^{-n_g}}{\beta_g}

        All three models are computed in terms of the normalized saturation
        .. math::
            \hat{S}_w=\frac{S_w - S_w^{min}}{S_w^{max} - S_w^{min}},

        If none of the models is chosen, the capillary pressure is set to 0.

        """
        s_normalized = self._s_normalized()
        if self._cap_pressure_model == "Brooks-Corey":
            entry_pressure = pp.ad.Scalar(self._entry_pressure)
            # Setup pp.ad.functions.pow
            pow_func = pp.ad.Function(partial(ad_pow, exponent=self._n_b), "pow")
            return entry_pressure * pow_func(s_normalized)
        elif self._cap_pressure_model == "linear":
            cap_pressure_linear_param = pp.ad.Scalar(self._cap_pressure_linear_param)
            return cap_pressure_linear_param * s_normalized
        elif self._cap_pressure_model == "van Genuchten":
            beta_g = pp.ad.Scalar(self._beta_g)
            # Setup pp.ad.functions.pow
            pow_func_1 = pp.ad.Function(partial(ad_pow, exponent=self._m_g), "pow")
            pow_func_2 = pp.ad.Function(partial(ad_pow, exponent=-self._n_g), "pow")
            return pow_func_2(pow_func_1(s_normalized) - pp.ad.Scalar(1)) / beta_g
        else:
            # Return cap. pressure 0.
            return s_normalized * pp.ad.Scalar(0)

    def _rel_perm_w(self) -> pp.ad.Operator:
        r"""Wetting phase relative permeability.

        The following two models are implemented:

        Brooks-Corey model
        .. math::
            k_{r,w}(\hat{S}_w)&=\hat{S}_w^{n_1+n_2\cdot n_3},\\
            \text{where}\\
            \hat{S}_w=\frac{S_w-S_{w,res}}{1-S_{w,res}-S_{n,res}}

        The default values are
        .. math::
            n_1=2,n_2=1+2/n_b,n_3=1.

        This is the Brooks–Corey–Burdine model.

        Corey model
        .. math::
            k_{r,w}(S_w)=\hat{S}_w^3

        To avoid ill-conditioned equation systems and crashing of the Newton solver at
        unphysical saturations (i.e., :math:`S_w\not\in[0,1]`), the nonwetting rel.
        perm. can be limited below and above.

        .. math::
            \hat{k}_{r,w}(S_w)=\min\{\max\{k_{r,w},k_{r,w}^{max}\},k_{r,w}^{min}},

        Returns:
            Wetting phase relative permeability.
        """
        s_normalized = self._s_normalized()
        if self._rel_perm_model in ["Corey", "power"]:
            rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_w)
            cube_func = pp.ad.Function(partial(ad_pow, exponent=3), "cube")
            rel_perm = cube_func(s_normalized) * rel_perm_linear_param
        elif self._rel_perm_model in ["linear"]:
            rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_w)
            rel_perm = s_normalized * rel_perm_linear_param
        elif self._rel_perm_model == "Brooks-Corey":
            power_func = pp.ad.Function(
                partial(ad_pow, exponent=self._n1 + self._n2 * self._n3), "power"
            )
            rel_perm = power_func(s_normalized)
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_w_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum, var_1=self._rel_perm_w_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def _rel_perm_n(self) -> pp.ad.Operator:
        r"""Nonwetting phase relative permeability.

        The following two models are implemented:

        Brooks-Corey model
        .. math::
            k_{r,w}(\hat{S}_w)&=(1-\hat{S}_w)^{n_1}(1-\hat{S}_w^{n_2})^{n_3},
            \text{where}\\
            \hat{S}_w=\frac{S_w-S_{w,res}}{1-S_{w,res}-S_{n,res}}

        The default values are (Brooks–Corey–Burdine model)
        .. math::
            n_1=2,n_2=1+2/n_b,n_3=1.

        Corey model
        .. math::
            k_{r,n}(S_w)=(1-\hat{S}_w)^3

        To avoid ill-conditioned equation systems and crashing of the Newton solver at
        unphysical saturations (i.e., :math:`S_w\not\in[0,1]`), the nonwetting rel.
        perm. can be limited below and above.

        .. math::
            \hat{k}_{r,n}(S_w)=\min\{\max\{k_{r,n},k_{r,n}^{max}\},k_{r,n}^{min}},

        Returns:
            Nonwetting phase relative permeability.
        """
        s_normalized = self._s_normalized()
        if self._rel_perm_model in ["Corey", "power"]:
            rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_n)
            cube_func = pp.ad.Function(
                partial(ad_pow, exponent=self._rel_perm_power), "cube"
            )
            rel_perm = cube_func(pp.ad.Scalar(1) - s_normalized) * rel_perm_linear_param
        elif self._rel_perm_model in ["linear"]:
            rel_perm_linear_param = pp.ad.Scalar(self._rel_perm_linear_param_n)
            rel_perm = (pp.ad.Scalar(1) - s_normalized) * rel_perm_linear_param
        elif self._rel_perm_model == "Brooks-Corey":
            power_func1 = pp.ad.Function(partial(ad_pow, exponent=self._n1), "power")
            power_func2 = pp.ad.Function(partial(ad_pow, exponent=self._n2), "power")
            power_func3 = pp.ad.Function(partial(ad_pow, exponent=self._n3), "power")
            rel_perm = power_func1(pp.ad.Scalar(1) - s_normalized) * power_func3(
                pp.ad.Scalar(1) - power_func2(s_normalized)
            )
        if self._limit_rel_perm:
            maximum_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var_1=self._rel_perm_n_min), "max"
            )
            minimum_func = pp.ad.Function(
                partial(minimum, var_1=self._rel_perm_n_max), "min"
            )
            return minimum_func(maximum_func(rel_perm))
        else:
            return rel_perm

    def set_equations(self) -> None:
        """Define equations."""
        try:
            self.equation_system.remove_equation("Flow equation")
            self.equation_system.remove_equation("Transport equation")
        except:
            Exception
        subdomains = self.mdg.subdomains()
        # self._ad.subdomains = subdomains
        if len(list(self.mdg.subdomains(dim=self.mdg.dim_max()))) != 1:
            raise NotImplementedError("This will require further work")

        # Neumann boundary conditions. Dirichlet bc are included in the flux
        p_w_neu_bc = pp.ad.DenseArray(self._neumann_bc_values_flux_w(subdomains[0]))

        # Spatial discretization operators.
        div = pp.ad.Divergence(subdomains)
        flux_mpfa_w = pp.ad.MpfaAd(self.w_flux_key, subdomains)
        flux_mpfa_n = pp.ad.MpfaAd(self.n_flux_key, subdomains)
        cap_flux_mpfa = pp.ad.TpfaAd(self.cap_flux_key, subdomains)
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, subdomains)
        upwind_n = pp.ad.UpwindAd(self.n_flux_key, subdomains)
        mass_matrix = pp.ad.MassMatrixAd(self.mass_key, subdomains)

        # Ad saturation variables, pressure is not needed as the flux is computed in a
        # seperate method (``_flux``).
        s = self.equation_system.md_variable(self.saturation_var)
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s = pp.ad.time_derivatives.dt(s, dt)

        # Ad source
        source_ad_w = pp.ad.DenseArray(self._source_w(subdomains[0]))
        source_ad_t = pp.ad.DenseArray(self._source_t(subdomains[0]))

        # Ad parameters
        porosity_ad = pp.ad.DenseArray(self._porosity(subdomains[0]))

        # Compute cap pressure and relative permeabilities.
        p_cap = self._cap_pressure()
        p_cap_bc = pp.ad.DenseArray(self._bc_values_cap_pressure(subdomains[0]))
        mobility_w = self._mobility_w(subdomains=subdomains)
        mobility_n = self._mobility_n(subdomains=subdomains)
        mobility_t = self._mobility_t(subdomains=subdomains)

        # Ad equations
        if self._formulation == "w_pressure_w_saturation":
            # TODO: This does not work correctly at the moment.
            # Wetting fluid flux.
            flux_w = self._flux_w(subdomains)
            flow_equation = (
                div
                @ (
                    (mobility_t @ flux_w + upwind_w.bound_transport_neu @ p_w_neu_bc)
                    + mobility_n
                    * (cap_flux_mpfa.flux @ p_cap + cap_flux_mpfa.bound_flux @ p_cap_bc)
                )
                - source_ad_t
            )
            # Add the flux at faces with Neumann bc. For PorePy reasons the bc need to
            # be negative.
            transport_equation = (
                porosity_ad * dt_s
                + (
                    div
                    @ (mobility_w * flux_w - upwind_w.bound_transport_neu @ p_w_neu_bc)
                )
                - source_ad_w
            )
        elif self._formulation == "n_pressure_w_saturation":
            # Note, that for ``flux_t``, the mobility is already included.
            flux_t = self._flux_t(subdomains)
            flow_equation = div @ flux_t - mass_matrix.mass @ source_ad_t
            fractional_flow_w = mobility_w / mobility_t

            vector_source_w = pp.ad.DenseArray(self._vector_source_w(subdomains[0]))
            vector_source_n = pp.ad.DenseArray(self._vector_source_n(subdomains[0]))
            transport_equation = (
                mass_matrix.mass @ (porosity_ad * dt_s)
                + div
                @ (
                    fractional_flow_w * flux_t
                    - fractional_flow_w
                    * mobility_n
                    * (
                        cap_flux_mpfa.flux @ p_cap
                        + flux_mpfa_w.vector_source @ vector_source_w
                        - flux_mpfa_n.vector_source @ vector_source_n
                    )
                )
                - mass_matrix.mass @ source_ad_w
            )
        flow_equation.set_name("Flow equation")
        transport_equation.set_name("Transport equation")
        # Update the equation list.
        self.equation_system.set_equation(flow_equation, subdomains, {"cells": 1})
        self.equation_system.set_equation(transport_equation, subdomains, {"cells": 1})

    def _mobility_w(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # TODO: Does it make more sense to get this from the dictionary? Would need to
        # change ``BuckleyLeverett`` then, as the mobilities are updated at iterations
        # via ``self._bc_values_mobility_w``.
        mobility_w_bc = pp.ad.DenseArray(self._bc_values_mobility_w(subdomains[0]))
        viscosity_w = pp.ad.Scalar(self._viscosity_w)
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, subdomains)
        # Neumann bc are included here. They don't influence total, wetting or
        # nonwetting flux at the Neumann boundaries, as those are fixed and independent
        # of the mobilities. However, they influence the fractional flow.
        mobility_w = (
            upwind_w.upwind @ (self._rel_perm_w() / viscosity_w)
            + upwind_w.bound_transport_dir @ mobility_w_bc
            + upwind_w.bound_transport_neu @ mobility_w_bc
        )
        return mobility_w

    def _mobility_n(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # TODO: Does it make more sense to get this from the dictionary? Would need to
        # change ``BuckleyLeverett`` then, as the mobilities are updated at iterations
        # via ``self._bc_values_mobility_n``.
        mobility_n_bc = self._bc_values_mobility_n(subdomains[0])
        viscosity_n = pp.ad.Scalar(self._viscosity_w)
        upwind_n = pp.ad.UpwindAd(self.n_flux_key, subdomains)
        # Neumann bc are included here. They don't influence total, wetting or
        # nonwetting flux at the Neumann boundaries, as those are fixed and independent
        # of the mobilities. However, they influence the fractional flow.
        mobility_n = (
            upwind_n.upwind @ (self._rel_perm_n() / viscosity_n)
            + upwind_n.bound_transport_dir @ mobility_n_bc
            + upwind_n.bound_transport_neu @ mobility_n_bc
        )
        return mobility_n

    def _mobility_t(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return (
            self._mobility_w(subdomains=subdomains)
            + self._mobility_n(subdomains=subdomains)
            # The small :math:`\epsilon` is not needed, if ``mobility_w`` and/or
            # ``mobility_n`` are not completely off.
            # + pp.ad.Scalar(1e-7)
        )

    def _flux_w(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Wetting phase volume flux.

        SI Units: kg/s
        """
        p_w = self.equation_system.md_variable(self.pressure_w_var)
        p_w_dir_bc = pp.ad.DenseArray(
            self._dirichlet_bc_values_pressure_w(subdomains[0])
        )
        vector_source_w = pp.ad.DenseArray(self._vector_source_w(subdomains[0]))
        flux_mpfa = pp.ad.MpfaAd(
            self.w_flux_key,
            subdomains,
        )
        flux: pp.ad.Operator = (
            flux_mpfa.flux @ p_w
            + flux_mpfa.bound_flux @ p_w_dir_bc
            - flux_mpfa.vector_source @ vector_source_w
        )
        flux.set_name("Wetting volume flux")
        return flux

    def _flux_n(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Nonwetting phase volume flux.

        SI Units: kg/s
        """
        p_n = self.equation_system.md_variable(self.pressure_n_var)
        p_n_dir_bc = pp.ad.DenseArray(
            self._dirichlet_bc_values_pressure_n(subdomains[0])
        )
        vector_source_n = pp.ad.DenseArray(self._vector_source_n(subdomains[0]))
        flux_mpfa = pp.ad.MpfaAd(
            self.n_flux_key,
            subdomains,
        )
        flux: pp.ad.Operator = (
            flux_mpfa.flux @ p_n
            + flux_mpfa.bound_flux @ p_n_dir_bc
            - flux_mpfa.vector_source @ vector_source_n
        )
        flux.set_name("Nonwetting volume flux")
        return flux

    def _flux_t(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Total volume flux.

        This is always calculated in terms of the nonwetting pressure and the capillary
        pressure (i.e. in terms of the wetting Saturation). Note that, unlike in the
        phase flux functions, the mobilities are already included in this formulation.

        SI Units: kg/s
        """
        # Variables, parameters and bc.
        p_n = self.equation_system.md_variable(self.pressure_n_var)
        flux_n_bc_neu = pp.ad.DenseArray(self._neumann_bc_values_flux_n(subdomains[0]))
        p_n_bc_dir = pp.ad.DenseArray(
            self._dirichlet_bc_values_pressure_n(subdomains[0])
        )
        p_cap_bc = pp.ad.DenseArray(self._bc_values_cap_pressure(subdomains[0]))
        vector_source_w = pp.ad.DenseArray(self._vector_source_w(subdomains[0]))
        vector_source_n = pp.ad.DenseArray(self._vector_source_n(subdomains[0]))
        # Spatial discretization operators.
        flux_mpfa = pp.ad.MpfaAd(self.n_flux_key, subdomains)
        upwind_n = pp.ad.UpwindAd(self.n_flux_key, subdomains)
        # NOTE: We use TPFA for discretization of the capillary flux.
        cap_flux_tpfa = pp.ad.TpfaAd(self.cap_flux_key, subdomains)
        # Cap pressure and relative permeabilities.
        p_cap = self._cap_pressure()
        mobility_w = self._mobility_w(subdomains)
        mobility_n = self._mobility_n(subdomains)
        mobility_t = self._mobility_t(subdomains)
        # Compute flux.
        flux_n: pp.ad.Operator = (
            flux_mpfa.flux @ p_n + flux_mpfa.bound_flux @ p_n_bc_dir
        )
        flux_p_cap: pp.ad.Operator = (
            cap_flux_tpfa.flux @ p_cap + cap_flux_tpfa.bound_flux @ p_cap_bc
        )
        flux_buoyancy_w: pp.ad.Operator = flux_mpfa.vector_source @ vector_source_w
        flux_buoyancy_n: pp.ad.Operator = flux_mpfa.vector_source @ vector_source_n
        total_flux = (
            mobility_t * flux_n
            # Add the flux at faces with Neumann bc. For PorePy reasons this needs to be
            # negative.
            - upwind_n.bound_transport_neu @ flux_n_bc_neu
            - mobility_w * flux_p_cap
            - mobility_w * flux_buoyancy_w
            - mobility_n * flux_buoyancy_n
        )
        total_flux.set_name("Total volume flux")
        return total_flux

    def _error_function_deriv(self) -> pp.ad.Operator:
        """Returns the derivative of the error function w.r.t. the saturation.

        This can be used to simulate perturbations in the cap. pressure and rel. perm.
        models.

        Returns:
            Derivative of the error function in terms of :math:`S_w`.
        """
        s = self.equation_system.md_variable(self.saturation_var)
        yscale = pp.ad.Scalar(self._yscale)
        xscale = pp.ad.Scalar(self._xscale)
        offset = pp.ad.Scalar(self._offset)
        exp_func = pp.ad.Function(pp.ad.functions.exp, "exp")
        square_func = pp.ad.Function(partial(ad_pow, exponent=2), "square")
        return yscale * exp_func(pp.ad.Scalar(-1) * xscale * square_func(s - offset))


class TwoPhaseFlowBoundaryConditions:
    domain_boundary_sides: Callable
    """Provided by a mixin of type ``ModelGeometry``."""

    def _bc_type_pressure_w(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Wetting pressure boundary conditions.

        Neumann conditions on three sides; Dirichlet on the south side to ensure
        existence of a unique solution.

        """
        # Ignore since MyPy complain that the named tuple has no ``south`` attribute.
        south = self.domain_boundary_sides(g).south  # type: ignore
        return pp.BoundaryCondition(g, south, "dir")

    def _bc_type_pressure_n(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Nonwetting pressure boundary conditions.

        Neumann conditions on three sides; Dirichlet on the south side to ensure
        existence of a unique solution.

        """
        # Ignore since MyPy complain that the named tuple has no ``south`` attribute.
        south = self.domain_boundary_sides(g).south  # type: ignore
        return pp.BoundaryCondition(g, south, "dir")

    def _bc_type_pressure_c(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Capillary pressure boundary conditions.

        Neumann conditions on all sides.

        """
        return pp.BoundaryCondition(g)

    def _dirichlet_bc_values_pressure_w(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet boundary values. Equals the initial state pressure."""
        array = np.zeros(g.num_faces)
        return array

    def _dirichlet_bc_values_pressure_n(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Dirichlet boundary values. Equals the initial state pressure."""
        array = np.zeros(g.num_faces)
        return array

    def _neumann_bc_values_flux_w(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Neumann boundary values.

        NOTE: In the wetting pressure-wetting saturation formulation, Neumann bc values
        correspond to ``_flux_w``, i.e. the mobility is already incorporated. In other
        formulations this is less clear, however the bc are not needed there. They are
        not needed for upwinding.

        NOTE: We set the values independently from the Dirichlet bc, because otherwise
        they would be counted twice on the RHS (once by ``mpfa.bound_flux`` and once by
        ``upwind.bound_transport_neu``).

        """
        array = np.zeros(g.num_faces)
        return array

    def _neumann_bc_values_flux_n(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous Neumann boundary values.

        NOTE: Neumann bc values correspond to ``_flux_t``, i.e. the mobility is already
        incorporated.

        NOTE: We set the values independently from the Dirichlet bc, because otherwise
        they would be counted twice on the RHS (once by ``mpfa.bound_flux`` and once by
        ``upwind.bound_transport_neu``).

        """
        array = np.zeros(g.num_faces)
        return array

    def _bc_values_cap_pressure(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values. Dirichlet pressure equals the initial state
        pressure"""
        array = np.zeros(g.num_faces)
        return array

    def _bc_values_mobility_w(self, g: pp.Grid) -> np.ndarray:
        """Wetting mobility at the boundaries.

        NOTE: These are both for Dirichlet bc and Neumann bc. The latter one is needed
        to compute the fractional flow at the boundaries. However, it does not affect
        the Neumann flux.

        NOTE: For some reason, we must choose the negative of the value to get a
        positive mobility at the boundary.

        """
        array = np.full(g.num_faces, -0.25)
        return array

    def _bc_values_mobility_n(self, g: pp.Grid) -> np.ndarray:
        """Nonwetting mobility at the boundaries.

        NOTE: These are both for Dirichlet bc and Neumann bc. The latter one is needed
        to compute the fractional flow at the boundaries.However, it does not affect
        the Neumann flux.

        NOTE: For some reason, we must choose the negative of the value to get a
        positive mobility at the boundary.

        """
        array = np.full(g.num_faces, -0.25)
        return array


class TwoPhaseFlowVariables:
    equation_system: pp.ad.EquationSystem
    pressure_w_var: str
    pressure_n_var: str
    saturation_var: str
    mdg: pp.MixedDimensionalGrid

    def create_variables(self) -> None:
        """Create primary variables (wetting pressure, nonwetting pressure,
        saturation)."""
        subdomains = self.mdg.subdomains()
        self.equation_system.create_variables(
            self.pressure_w_var, {"cells": 1}, subdomains
        )
        self.equation_system.create_variables(
            self.pressure_n_var, {"cells": 1}, subdomains
        )
        self.equation_system.create_variables(
            self.saturation_var, {"cells": 1}, subdomains
        )


class TwoPhaseFlowSolutionStrategy(pp.SolutionStrategy):
    # Ad discretization:
    _flux_w: Callable
    """Provided by ``TwoPhaseFlow``"""
    _flux_n: Callable
    """Provided by ``TwoPhaseFlow``"""
    _cap_pressure: Callable
    """Provided by ``TwoPhaseFlow``"""

    # Matrix properties:
    _permeability: Callable
    """Provided by ``TwoPhaseFlow``"""
    _porosity: Callable
    """Provided by ``TwoPhaseFlow``"""

    # Source:
    _source_w: Callable
    """Provided by ``TwoPhaseFlow``"""
    _source_n: Callable
    """Provided by ``TwoPhaseFlow``"""
    _source_t: Callable
    """Provided by ``TwoPhaseFlow``"""

    # Boundary conditions:
    _bc_type_pressure_w: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_type_pressure_n: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_type_pressure_c: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_values_mobility_w: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_values_mobility_n: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _dirichlet_bc_values_pressure_w: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _dirichlet_bc_values_pressure_n: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""
    _bc_values_cap_pressure: Callable
    """Provided by ``TwoPhaseFlowBoundaryConditions``"""

    def __init__(self, params: Optional[dict]) -> None:
        super().__init__(params)
        if params is None:
            params = {}
        self._formulation: str = params.get("formulation", "n_pressure_w_saturation")
        """Choose which formulation of two-phase flow shall be run. Note, that his has
        (!!!) to be passed as a parameter. Changing it after initialization may result
        in wrong results. "
        
        Valid values:
            'w_pressure_w_saturation':
            'n_pressure_w_saturation':

        """
        # Discretizations and parameter keywords:
        self.params_key: str = "params"
        """Keyword to define parameters that are not related to any phase (i.e.
        porosity, total source etc.)"""
        self.w_flux_key: str = "wetting flux"
        """Keyword to define parameters and discretizations for the wetting phase."""
        self.n_flux_key: str = "nonwetting flux"
        """Keyword to define parameters and discretizations for the nonwetting phase."""
        self.cap_flux_key: str = "capillary flux"
        """Keyword to define parameters and discretizations for the capillary flux."""
        self.mass_key: str = "mass"
        """Keyword to define parameters and discretizations for the mass matrix."""

        # Variables:
        self.pressure_w_var = "pressure_w"
        r"""Name for the wetting pressure variable :math:p_w . Depending on the chosen
        formulation, this is either a primary or a secondary variable."""
        self.pressure_n_var = "pressure_n"
        r"""Name for the nonwetting pressure variable :math:p_n . Depending on the
        chosen formulation, this is either a primary or a secondary variable."""
        self.saturation_var: str = "saturation_w"
        """Name for the wetting saturation variable :math:S_w ."""
        # Select primary pressure variable depending on the formulation.
        # NOTE: The division into primary/secondary variables is internal in this model
        # only and not connected to ``pp.PRIMARY_VARIABLES``.
        if self._formulation == "w_pressure_w_saturation":
            self.primary_pressure_var: str = self.pressure_w_var
            self.secondary_pressure_var: str = self.pressure_n_var
        elif self._formulation == "n_pressure_w_saturation":
            self.primary_pressure_var = self.pressure_n_var
            self.secondary_pressure_var = self.pressure_w_var

        # Phase parameters:
        self._viscosity_w: float = params.get("viscosity_w", 1.0)
        """Wetting fluid viscosity.

        SI Units: kg/(m*s)
        """
        self._viscosity_n: float = params.get("viscosity_n", 1.0)
        """Nonetting fluid viscosity.

        SI Units: kg/(m*s)
        """
        self._density_w: float = params.get("density_w", 0.0)
        """Wetting fluid density.

        SI Units: kg/m^3
        """
        self._density_n: float = params.get("density_n", 0.0)
        """Nonetting fluid density.

        SI Units: kg/m^3
        """

        # Residual saturations:
        self._residual_saturation_w: float = params.get("residual_saturation_w", 0.3)

        self._residual_saturation_n: float = params.get("residual_saturation_n", 0.3)

        # Model selection and parameters for the capillary pressure function:
        self._cap_pressure_model: Literal[
            "Brooks-Corey", "van Genuchten", "linear", None
        ] = "Brooks-Corey"
        # van Genuchten model
        self._n_g: float = 2.0
        self._m_g: float = 2 / 3
        self._beta_g: float = params.get("beta_g", 1.0)
        # Brooks-Corey model
        self._entry_pressure: float = 0.1
        self._n_b: int = 1
        # linear model
        self._cap_pressure_linear_param: float = params.get(
            "cap_pressure_linear_param", 0.1
        )

        # NOTE: Using the default values, the linear model and the Brooks-Corey model
        # are identical.

        # Model selection and parameters for the relative permeability function:
        # NOTE: "power" and "Corey" are the same model.
        self._rel_perm_model: Literal[
            "Brooks-Corey", "Corey", "power", "linear"
        ] = params.get("rel_perm_model", "Brooks-Corey")
        # Brooks-Corey model
        self._n1: int = 2
        self._n2: int = int(1 + 2 / self._n_b)
        self._n3: int = 1
        # Power law (Corey model)
        self._rel_perm_power: int = params.get("rel_perm_power", 3)
        self._rel_perm_linear_param_w: float = params.get(
            "rel_perm_linear_param_w", 1.0
        )
        self._rel_perm_linear_param_n: float = params.get(
            "rel_perm_linear_param_n", 1.0
        )

        # Lower and upper limits for the rel.perm
        # If the ``limit_rel_perm`` parameter is set to ``False``, these values are
        # ignored.
        self._limit_rel_perm: bool = params.get("limit_rel_perm", True)
        self._rel_perm_w_max: float = 0.99
        self._rel_perm_w_min: float = 0.01
        self._rel_perm_n_max: float = 0.99
        self._rel_perm_n_min: float = 0.01

        # Parameters for the error function derivative:
        self._yscale: float = params.get("yscale", 1.0)
        self._xscale: float = params.get("xscale", 200)
        self._offset: float = params.get("offset", 0.5)

        # Solvers:
        self._use_ad: bool = params.get("use_ad", True)
        self.linear_solver: Literal[
            "scipy_sparse", "pypardiso", "umfpack"
        ] = params.get("linear_solver", "scipy_sparse")

        # Option to limit the saturation change per timestep.
        self._limit_saturation_change: bool = False
        """If this is set to ``True``, the Newton method fails, if the final solution
        differs from the previous timestep by more than ``self._max_saturation_change``
        in any grid cell. The timestep is then shortened and recalculated.
        """
        self._max_saturation_change: float = 0.2

    def is_time_dependent(self) -> bool:
        return True

    def prepare_simulation(self) -> None:
        """This setups the model, s.t. a simulation can be run.

        The initial values are exported.
        """
        self.set_geometry()
        self.create_managers()
        #
        self.create_variables()
        self.initial_condition()
        self.set_discretization_parameters()
        self.set_equations()
        #
        self.discretize()
        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
        )
        self._export()

    def set_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameter fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).
        """
        super().set_discretization_parameters()
        for sd, data in self.mdg.subdomains(return_data=True):
            # Boundary conditions and parameters
            diffusivity = pp.SecondOrderTensor(self._permeability(sd))
            # all_bf, *_ = self._domain_boundary_sides(sd)
            # Parameters that are not used for discretization.
            pp.initialize_data(
                sd,
                data,
                self.params_key,
                {
                    "source_t": self._source_t(sd),
                    "porosity": self._porosity(sd),
                    "bc_values_mobility_w": self._bc_values_mobility_w(sd),
                    "bc_values_mobility_n": self._bc_values_mobility_n(sd),
                },
            )
            # Parameters for wetting phase.
            pp.initialize_data(
                sd,
                data,
                self.w_flux_key,
                {
                    "source_w": self._source_w(sd),
                    "bc": self._bc_type_pressure_w(sd),
                    "bc_values": self._dirichlet_bc_values_pressure_w(sd),
                    "darcy_flux": np.ones(sd.num_faces),
                    "second_order_tensor": diffusivity,
                },
            )
            # Parameters for nonwetting phase.
            pp.initialize_data(
                sd,
                data,
                self.n_flux_key,
                {
                    "source_n": self._source_n(sd),
                    "bc": self._bc_type_pressure_n(sd),
                    "bc_values": self._dirichlet_bc_values_pressure_n(sd),
                    "darcy_flux": np.ones(sd.num_faces),
                    "second_order_tensor": diffusivity,
                },
            )
            # Parameters for capillary flux.
            pp.initialize_data(
                sd,
                data,
                self.cap_flux_key,
                {
                    "bc": self._bc_type_pressure_c(sd),
                    "bc_values": self._bc_values_cap_pressure(sd),
                    "second_order_tensor": diffusivity,
                },
            )
            # Parameters for mass matrix.
            # NOTE: the porosity is not included here, but in the equation. -> The
            # resulting ``pp.ad.MassMatrix`` equals only the specific volume size.
            pp.initialize_data(
                sd,
                data,
                self.mass_key,
                {"mass_weight": np.ones(sd.num_cells)},
            )

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        # TODO: Update this for multiple subdomains.
        sd = self.mdg.subdomains()[0]
        self.equation_system.set_variable_values(
            np.full(sd.num_cells, 0.0),
            [self.pressure_w_var],
            time_step_index=self.time_manager.time_index,
        )
        self.equation_system.set_variable_values(
            np.full(sd.num_cells, 0.0),
            [self.pressure_n_var],
            time_step_index=self.time_manager.time_index,
        )
        self.equation_system.set_variable_values(
            np.full(sd.num_cells, 0.5),
            [self.saturation_var],
            time_step_index=self.time_manager.time_index,
        )

    def create_managers(self) -> None:
        """Create an ``EquationSystem`` and a ``TimeManager``."""
        self.equation_system = pp.ad.EquationSystem(self.mdg)

    def discretize(self) -> None:
        """Discretize all terms"""
        t_0 = time.time()
        self.equation_system.discretize()
        # Fluid flux induced by the secondary variable. This needs to be discretized
        # each nonlinear iteration s.t. the Darcy flux can be computed for the secondary
        # variable s.t. the upwind operator works correctly.
        if self._formulation == "w_pressure_w_saturation":
            flux_n = self._flux_n(self.mdg.subdomains())
            flux_n.discretize(self.mdg)
        elif self._formulation == "n_pressure_w_saturation":
            flux_w = self._flux_w(self.mdg.subdomains())
            flux_w.discretize(self.mdg)
        logger.debug(f"Discretized in {time.time() - t_0:.2e} seconds")

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system.

        The linear system is defined by the current state of the model.

        Attributes:
            linear_system is assigned.

        """
        t_0 = time.time()
        if self._use_ad:
            self.linear_system = self.equation_system.assemble_subsystem(
                variables=[self.primary_pressure_var, self.saturation_var]
            )
        logger.debug(f"Assembled linear system in {time.time() - t_0:.2e} seconds")

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu(
                [self.pressure_w_var, self.pressure_n_var, self.saturation_var],
                time_step=self.time_manager.time_index,
            )

    def _is_nonlinear_problem(self):
        return True

    # nonlinear loop.
    def before_nonlinear_loop(self) -> None:
        """Set the starting estimate to the solution from the previous timestep."""
        self.time_manager._recomp_sol = False
        self._nonlinear_iteration = 0
        assembled_variables = self.equation_system.get_variable_values(
            time_step_index=0
        )

        self.equation_system.set_variable_values(
            assembled_variables, iterate_index=0, additive=False
        )
        if self._limit_saturation_change:
            self._prev_saturation: np.ndarray = (
                self.equation_system.get_variable_values(
                    variables=[self.saturation_var], time_step_index=0
                )
            )

    def before_nonlinear_iteration(self) -> None:
        """Compute Darcy flux based on previous pressure solution to determine upstream
        direction.

        To evaluate the phase-mobilities separately, both the wetting, as well as the
        nonwetting flux need to be computed.

        """
        # Evaluate the pressure of the secondary pressure variable.
        secondary_pressure = (
            self.equation_system.md_variable(self.primary_pressure_var)
            - self._cap_pressure()
        )
        secondary_pressure_sol = secondary_pressure.evaluate(self.equation_system).val

        # Update the iterate of the secondary pressure variable. As the values were
        # computed with the additive value of the primary pressure and saturation
        # variable, we set ``additive=False``.
        self.equation_system.set_variable_values(
            secondary_pressure_sol,
            variables=[self.secondary_pressure_var],
            iterate_index=0,
            additive=False,
        )

        # Compute the Darcy flux for upwinding.
        # -> Needs to happen at each nonlinear iteration, because we are starting with a
        # bad guess (previous timestep) and improve towards the solution. We want to use
        # the better guess of the Darcy flux for discretization.

        # NOTE: This should only be done with care at the first iteration of the initial
        # time step, as the upwind direction will not align with the initital guess for
        # upwinding then.
        # -> Might be unwanted or wanted behavior.
        if self.time_manager.time_index >= 2 or self._nonlinear_iteration >= 1:
            logger.info(
                f"Recalculate Darcy flux for upwind discretization. Iteration {self._nonlinear_iteration}"
            )
            for sd, data in self.mdg.subdomains(return_data=True):
                # Update wetting flux.
                vals = self._flux_w([sd]).evaluate(self.equation_system).val
                data[pp.PARAMETERS][self.w_flux_key].update({"darcy_flux": vals})
                # Update nonwetting flux.
                vals = self._flux_n([sd]).evaluate(self.equation_system).val
                data[pp.PARAMETERS][self.n_flux_key].update({"darcy_flux": vals})
        self.discretize()

    def after_nonlinear_iteration(self, solution: np.ndarray) -> None:
        """Distribute solution variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.
        """
        # Distribute pressure and saturation variable. Secondary pressure variable is
        # distributed before the nonlinear iteration.
        self.equation_system.set_variable_values(
            solution,
            variables=[self.primary_pressure_var, self.saturation_var],
            iterate_index=0,
            additive=True,
        )
        self._nonlinear_iteration += 1

    # Ignore mypy complaining about uncompatible signature.
    def after_nonlinear_convergence(  # type: ignore
        self, solution: np.ndarray, errors: list[float], iteration_counter: int
    ) -> None:
        """Export and move to the next time step.

        When ``self._limit_saturation_change==True``, check if the wetting saturation
        has changed too much
        Parameters:
            solution: _description_
            errors: _description_
            iteration_counter: _description_
        """
        # If the saturation changes to much, decrease the time step and calculate again.
        if self._limit_saturation_change:
            new_saturation: np.ndarray = self.equation_system.get_variable_values(
                variables=[self.saturation_var], iterate_index=0
            )
            if (
                np.max(np.abs(new_saturation - self._prev_saturation))
                > self._max_saturation_change
            ):
                # This is set to false again in ``before_nonlinear_loop``.
                # NOTE: This is not a very nice solution, however, as of now I didn't
                # find a way to pass ``recompute_solution`` to
                # ``time_manager.compute_time_step()`` in ``run_time_dependent_model``
                # without the code getting really messy.
                self.time_manager._recomp_sol = True
                self.convergence_status = False
                logger.debug(
                    "Saturation grew to quickly. Trying again with a smaller time step."
                )
                return None
        # Distribute both pressure variables and the saturation variable.
        timestep_solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.set_variable_values(
            timestep_solution, time_step_index=0, additive=False
        )

        self.convergence_status = True
        self._export()
        logger.debug(f'{{"converged": {"true"}}}')

    # Ignore mypy complaining about uncompatible signature.
    def after_nonlinear_failure(  # type: ignore
        self, solution: np.ndarray, errors: list[float], iteration_counter: int
    ) -> None:
        logger.debug(f"Failed on timestep {self.time_manager.time_index}")
        logger.debug(f"Error {errors} nonlinear iteration {iteration_counter}")
        logger.debug(f'{{"converged": {"false"}}}')
        raise ValueError("nonlinear iterations did not converge")

    def after_simulation(self):
        pass


# Ignore mypy complaining about uncompatible signature between
# ``TwoPhaseFlowSolutionStrategy`` and ``pp.DataSavingMixin``.
class TwoPhaseFlowSetup(  # type: ignore
    TwoPhaseFlowEquations,
    TwoPhaseFlowVariables,
    TwoPhaseFlowBoundaryConditions,
    TwoPhaseFlowSolutionStrategy,
    #
    pp.ModelGeometry,
    #
    pp.DataSavingMixin,
):
    ...
