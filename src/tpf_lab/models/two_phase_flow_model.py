"""This module contains an implementation of a base model for two-phase flow problems.

The model uses the pressure-saturation formulation for the wetting fluid:
    .. math:
        -\nabla\cdot\left(\lambda_t\nabla p_w+\lambda_n\nabla p_c-\lambda_w\nabla\rho_w\bm{g}-\lambda_n\nabla\rho_n\bm{g}\right)=\bm{q}_t, 
        \phi\frac{\partial S_w}{\partial t}-\nabla\cdot\left(\lambda_w\nabla p_w-\lambda_w\nabla\rho_w\bm{g}\right)=\bm{q}_w,


TODO:
    - Denote Units for all parameters, values, variables.
    - Implement gravity.
    
Units:
    Collection of the SI units for all parameters. Not thoroughly checked if this all
    makes sense.
    saturation: dimensionless
    pressure: pascal=kg/(m*s^2)
    density: kg/m^3
    viscosity: kg/(m*s)
    permeability: m^2
    volumetric source terms: m^3/s
    mass flux: kg/s
"""

from __future__ import annotations

import logging
import os
import time
from functools import partial
from typing import Optional, Union

import numpy as np
import porepy as pp
import scipy.sparse as sps
from pythonjsonlogger import jsonlogger

from src.tpf_lab.numerics.ad.functions import pow


logger = logging.getLogger("__name__")


class _AdVariables:
    pressure_w: pp.ad.MergedVariable
    pressure_n: pp.ad.MergedVariable
    saturation: pp.ad.MergedVariable
    # Do we need the flux_discretization?
    flux_discretization: Union[pp.ad.MpfaAd, pp.ad.TpfaAd]
    subdomains: list[pp.Grid]


class TwoPhaseFlow(pp.models.abstract_model.AbstractModel):
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

    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        self._formulation: str
        """Choose which formulation of two-phase flow shall be run. Note, that his has
        (!!!) to be passed as a parameter. Changing it after initialization may result
        in wrong results. "
        
        Valid values:
            'w_pressure_w_saturation':
            'n_pressure_w_saturation':

        """
        if params is not None and "formulation" in params:
            self._formulation = params["formulation"]
        else:
            self._formulation = "w_pressure_w_saturation"
        # Variables
        self.pressure_w_var = "pressure_w"
        r"""Name for the wetting pressure variable :math:p_w . Depending on the chosen
        formulation, this is either a primary or a secondary variable."""
        self.pressure_n_var = "pressure_n"
        r"""Name for the nonwetting pressure variable :math:p_n . Depending on the
        chosen formulation, this is either a primary or a secondary variable."""
        self.saturation_var: str = "saturation_w"
        """Name for the wetting saturation variable :math:S_w ."""

        # Discretizations and params
        self.params_key: str = "params"
        """Keyword to define parameters that are not related to any discretization (i.e.
        porosity, total source and )"""
        self.w_flux_key: str = "wetting flux"
        """Keyword to define parameters and discretizations connected to the wetting
        phase."""
        self.n_flux_key: str = "nonwetting flux"
        """Keyword to define parameters and discretizations connected to the nonwetting
        phase."""
        self.cap_flux_key: str = "capillary flux"
        """Keyword to define parameters and discretizations connected to the capillary
        flux."""
        if self._formulation == "w_pressure_w_saturation":
            self.primary_pressure_var: str = self.pressure_w_var
            self.secondary_pressure_var: str = self.pressure_n_var
        elif self._formulation == "n_pressure_w_saturation":
            self.primary_pressure_var = self.pressure_n_var
            self.secondary_pressure_var = self.pressure_w_var

        # Some options
        self._use_ad: bool = True

        # Managers etc.
        self._ad = _AdVariables()
        self.exporter: pp.Exporter
        self.dof_manager: pp.DofManager
        self._eq_manager: pp.ad.EquationManager
        self.time_manager: pp.TimeManager

        # Time schedule.
        self._time_step: float = 0.2
        self._schedule: np.ndarray = np.array([0, 20.0])

        # Phase parameters
        self._w_viscosity: float = 1.0
        """Wetting fluid viscosity.

        SI Units: kg/(m*s)
        """
        self._n_viscosity: float = 1.0
        """Nonetting fluid viscosity.

        SI Units: kg/(m*s)
        """
        self._w_density: float = 1.0
        """Wetting fluid density.

        SI Units: kg/m^3
        """
        self._n_density: float = 1.0
        """Nonetting fluid density.

        SI Units: kg/m^3
        """

        # Parameters for the capillary pressure function
        self._cap_pressure_model: Optional[str] = "Brooks-Corey"
        # van Genuchten model
        self._n_g: float = 2.0
        self._m_g: float = 2 / 3
        self._beta_g: float = 1.0
        self._residual_w_saturation: float = 0.3
        self._residual_n_saturation: float = 0.0
        # Brooks Corey model
        self._entry_pressure: float = 0.1
        self._n_b: int = 1
        # linear model
        self._cap_pressure_linear: float = 1.0

        # Parameters for the relative permeability function
        self._rel_perm_model: str = "Brooks-Corey"
        self._n1: int = 2
        self._n2: int = int(1 + 2 / self._n_b)
        self._n3: int = 1
        # Parameters for the error function derivative
        self._yscale: float = 1.0
        self._xscale: float = 200
        self._offset: float = 0.5
        # Relative permeability limited below
        self._limit_rel_perm: bool = False
        # Note: the values get cubed!
        self._min_w_rel_perm: float = 0.01
        self._min_n_rel_perm: float = 0.01
        # Wetting saturation limited below
        self._limit_saturation: bool = False

        # Grid size
        self._grid_size: int = 20
        self._phys_size: int = 2

        # Limit saturation change
        self._limit_saturation_change: bool = False
        """If this is set to ``True``, the Newton method fails, if the final solution
        differs from the previous timestep by more than ``self._max_saturation_change``
        in any grid cell. The timestep is then shortened and recalculated.
        """
        self._max_saturation_change: float = 0.2

        # Setup logging.
        logger.handlers.clear()  # ? Why do we need this again?
        # Logging config
        try:
            os.makedirs(self.params["folder_name"])
        except OSError:
            pass
        fh = logging.FileHandler(
            os.path.join(
                self.params["folder_name"], ".".join([self.params["file_name"], "txt"])
            )
        )
        fh.setLevel(logging.DEBUG)
        # formatter = jsonlogger.JsonFormatter()
        # file_handler.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)
        logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)

    def prepare_simulation(self) -> None:
        self.create_grid()
        # Exporter initialization must be done after grid creation.
        self.exporter = pp.Exporter(
            self.mdg,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
        )
        self._assign_variables()
        self._create_dof_and_eq_manager()
        self._create_ad_variables()
        #
        self._initial_condition()
        self._set_parameters()
        self._assign_equations()
        #
        self._export()
        self._discretize()
        self._initialize_linear_solver()

    def _bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Neumann conditions on three sides; Dirichlet on one side to ensure existence
        of a unique solution."""
        south = self._domain_boundary_sides(g).south
        return pp.BoundaryCondition(g, south, "dir")

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values. Dirichlet pressure equals the initial state
        pressure"""
        array = np.zeros(g.num_faces)
        return array

    def _source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting source.

        In the default model there is no source term.

        SI Units: m^d/s
        """
        return np.zeros(g.num_cells)

    def _source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting source.

        SI Units: m^d/s
        """
        return np.zeros(g.num_cells)

    def _source_t(self, g: pp.Grid) -> np.ndarray:
        """Volumetric total source; sum of the wetting and nonetting source.

        SI Units: m^d/s
        """
        return self._source_w(g) + self._source_n(g)

    # More matrix and phase parameters.
    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Solid permeability.

        SI Units: m^2
        """
        return np.full(g.num_cells, 1)

    def _porosity(self, g: pp.Grid) -> np.ndarray:
        return np.full(g.num_cells, 1.0)

    # Cap pressure and relative permeability functions.
    def _cap_pressure(self) -> pp.ad.Operator:
        r"""Capillary pressure computed with the ... model.

        .. math::
            p_c(S_w)=\frac{(\hat{S}_w^{m_g}-1)^{-n_g}}{\beta_g}
            \text{for}
            \hat{S}_w=\frac{S_w-S_w^{min}}{S_w^{max}-S_w^{min}}\\

        """
        s = self._ad.saturation
        normalized_s = (s - self._residual_w_saturation) / (
            1 - self._residual_n_saturation - self._residual_w_saturation
        )
        if self._cap_pressure_model == "van Genuchten":
            # Setup pp.ad.functions.pow
            pow_func_1 = pp.ad.Function(partial(pow, exponent=self._m_g), "pow")
            pow_func_2 = pp.ad.Function(partial(pow, exponent=-self._n_g), "pow")
            return pow_func_2(pow_func_1(normalized_s) - 1) / self._beta_g
        elif self._cap_pressure_model == "Brooks-Corey":
            pow_func = pp.ad.Function(partial(pow, exponent=self._n_b), "pow")
            return pp.ad.Scalar(self._entry_pressure) * pow_func(normalized_s)
        elif self._cap_pressure_model == "linear":
            return self._cap_pressure_linear * normalized_s
        else:
            # TODO Make this look nicer (just return zero maybe? -> Doesn't work).
            return s * 0

    def _rel_perm_w(self) -> pp.ad.Operator:
        """Wetting phase relative permeability pressure computed with the ... model.

        .. math::
            k_{r,w}(s_w)=s_w^3
            or
            k_{r,w}(s_w)=\max\{s_w^3,0.01^3\},

        or whatever the minimum rel. perm. is.

        Returns:
            _description_
        """
        s = self._ad.saturation
        normalized_s = (s - self._residual_w_saturation) / (
            1 - self._residual_n_saturation - self._residual_w_saturation
        )
        if self._rel_perm_model == "Corey":
            # TODO: Add constant factor to power law.
            cube_func = pp.ad.Function(partial(pow, exponent=3), "cube")
            if self._limit_rel_perm:
                # TODO: Fix this so it applies to all subdomains.
                array = np.full(
                    self.mdg.subdomains()[0].num_cells, self._min_w_rel_perm
                )
                max_func = pp.ad.Function(
                    partial(pp.ad.functions.maximum, var1=array), "max"
                )
                return cube_func(max_func(s))
            else:
                return cube_func(s)
        elif self._rel_perm_model == "Brooks-Corey":
            power_func = pp.ad.Function(
                partial(pow, exponent=self._n1 + self._n2 * self._n3), "power"
            )
            return power_func(normalized_s)
        else:
            return s * 0

    def _rel_perm_n(self) -> pp.ad.Operator:
        """Non-wetting phase relative permeability pressure computed with the ... model.

        .. math::
            k_{r,n}(s_w)=(1-s_w)^3
            or
            k_{r,n}(s_w)=\max\{(1-s_w)^3,0.01^3\},

        or whatever the minimum rel. perm. is.

        Returns:
            _description_
        """
        s = self._ad.saturation
        normalized_s = (s - self._residual_w_saturation) / (
            1 - self._residual_n_saturation - self._residual_w_saturation
        )
        if self._rel_perm_model == "Corey":
            # TODO: Add constant factor to power law.
            cube_func = pp.ad.Function(partial(pow, exponent=3), "cube")
            if self._limit_rel_perm:
                # TODO: Fix this so it applies to all subdomains.
                min_value = np.full(
                    self.mdg.subdomains()[0].num_cells, self._min_n_rel_perm
                )
                max_func = pp.ad.Function(
                    partial(pp.ad.functions.maximum, var1=min_value), "max"
                )
                return cube_func(max_func(1 - s))
            else:
                return cube_func(1 - s)
        elif self._rel_perm_model == "Brooks-Corey":
            power_func1 = pp.ad.Function(partial(pow, exponent=self._n1), "power")
            power_func2 = pp.ad.Function(partial(pow, exponent=self._n2), "power")
            power_func3 = pp.ad.Function(partial(pow, exponent=self._n3), "power")

            return power_func1(1 - normalized_s) * power_func3(
                1 - power_func2(normalized_s)
            )
        else:
            return s * 0

    def create_grid(self) -> None:
        cell_dims: np.ndarray = np.array(
            [
                self._grid_size,
                self._grid_size,
            ]
        )
        phys_dims: np.ndarray = np.array(
            [
                self._phys_size,
                self._phys_size,
            ]
        )
        g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
        g_cart.compute_geometry()
        # g_tetra: pp.TetrahedralGrid = pp.TetrahedralGrid(g_cart.nodes)
        # g_tetra.compute_geometry()
        self.mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart]])
        self.box: dict = pp.bounding_box.from_points(
            np.array(
                [
                    [
                        0,
                        0,
                    ],
                    [
                        self._grid_size,
                        self._grid_size,
                    ],
                ]
            ).T
        )
        # logger.debug("Grid created")

    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameter fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            # Boundary conditions and parameters
            diffusivity = pp.SecondOrderTensor(self._permeability(sd))
            all_bf, *_ = self._domain_boundary_sides(sd)
            # Parameters that are not used for discretization.
            pp.initialize_data(
                sd,
                data,
                self.params_key,
                {
                    "source_t": self._source_t(sd),
                    "porosity": self._porosity(sd),
                },
            )
            # Parameters for wetting phase.
            pp.initialize_data(
                sd,
                data,
                self.w_flux_key,
                {
                    "source_w": self._source_w(sd),
                    "bc": self._bc_type(sd),
                    "bc_values": self._bc_values(sd),
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
                    "bc": self._bc_type(sd),
                    "bc_values": self._bc_values(sd),
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
                    # "bc": self._bc_type(sd),
                    "bc": pp.BoundaryCondition(sd, all_bf, "neu"),
                    "bc_values": np.zeros(sd.num_faces),
                    "second_order_tensor": diffusivity,
                },
            )

    def _initial_condition(self) -> None:
        """Set initial values for wetting pressure and saturation."""
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.set_state(data, {self.pressure_w_var: np.full(sd.num_cells, 0.0)})
            pp.set_state(data, {self.pressure_n_var: np.full(sd.num_cells, 0.0)})
            pp.set_state(data, {self.saturation_var: np.full(sd.num_cells, 0.5)})

    def _assign_variables(self) -> None:
        """
        Assign primary variables to subdomains and interfaces of the mixed-dimensional
        grid.
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            # One dof per cell for both variables
            data[pp.PRIMARY_VARIABLES] = {
                self.pressure_w_var: {"cells": 1},
                self.pressure_n_var: {"cells": 1},
                self.saturation_var: {"cells": 1},
            }

    def _create_dof_and_eq_manager(self) -> None:
        """Create a dof_manager and eq_manager based on a mixed-dimensional grid"""
        self.dof_manager = pp.DofManager(self.mdg)
        self._eq_manager = pp.ad.EquationManager(self.mdg, self.dof_manager)
        self.time_manager = pp.TimeManager(
            self._schedule, self._time_step, constant_dt=True
        )

    def _create_ad_variables(self) -> None:
        """Create merged variables for wetting pressure and saturation"""
        self._ad.pressure_w = self._eq_manager.merge_variables(
            [(sd, self.pressure_w_var) for sd in self.mdg.subdomains()]
        )
        self._ad.pressure_n = self._eq_manager.merge_variables(
            [(sd, self.pressure_n_var) for sd in self.mdg.subdomains()]
        )
        self._ad.saturation = self._eq_manager.merge_variables(
            [(sd, self.saturation_var) for sd in self.mdg.subdomains()]
        )

    def _assign_equations(self) -> None:
        """Define equations."""
        subdomains = [sd for sd in self.mdg.subdomains()]
        self._ad.subdomains = subdomains
        if len(list(self.mdg.subdomains(dim=self.mdg.dim_max()))) != 1:
            raise NotImplementedError("This will require further work")

        # Spatial discretization operators.
        div = pp.ad.Divergence(subdomains)
        cap_flux_mpfa = pp.ad.MpfaAd(self.cap_flux_key, subdomains)
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, subdomains)
        upwind_n = pp.ad.UpwindAd(self.n_flux_key, subdomains)

        # Ad saturation variables, pressure is not needed as the flux is computed in a
        # seperate method (``_flux``).
        s = self._ad.saturation
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s = pp.ad.time_derivatives.dt(s, dt)

        # Ad source
        source_ad_w = pp.ad.ParameterArray(self.w_flux_key, "source_w", subdomains)
        source_ad_t = pp.ad.ParameterArray(self.params_key, "source_t", subdomains)

        # Ad parameters
        viscosity_ad_w = pp.ad.Scalar(self._w_viscosity)
        viscosity_ad_n = pp.ad.Scalar(self._n_viscosity)
        porosity_ad = pp.ad.ParameterArray(self.params_key, "porosity", subdomains)

        # Compute cap pressure and relative permeabilities.
        p_cap = self._cap_pressure()
        p_cap_bc = pp.ad.ParameterArray(self.cap_flux_key, "bc_values", subdomains)
        mobility_w = upwind_w.upwind * (self._rel_perm_w() / viscosity_ad_w)
        mobility_n = upwind_n.upwind * (self._rel_perm_n() / viscosity_ad_n)
        # Add a small :math:\epsilon to the total mobility, to avoid division by zero.
        mobility_t = mobility_w + mobility_n + pp.ad.Scalar(1e-7)

        # Ad equations
        if self._formulation == "w_pressure_w_saturation":
            # Wetting fluid flux.
            flux_w = self._flux_w(subdomains)
            pressure_eq = (
                div
                * (
                    (mobility_t * flux_w)
                    + mobility_n
                    * (cap_flux_mpfa.flux * p_cap + cap_flux_mpfa.bound_flux * p_cap_bc)
                )
                - source_ad_t
            )
            saturation_eq = (
                porosity_ad * dt_s + (div * (mobility_w * flux_w)) - source_ad_w
            )
            pressure_eq.set_name("Wetting pressure equation")
            saturation_eq.set_name("Wetting saturation equation")
        elif self._formulation == "n_pressure_w_saturation":
            # Note, that for ``flux_t``, the mobility is already included.
            flux_t = self._flux_t(subdomains)
            flux_n = self._flux_n(subdomains)
            pressure_eq = div * flux_t - source_ad_t
            # pressure_eq = (
            #     div
            #     * (
            #         mobility_t
            #         * flux_n
            #         # - mobility_w
            #         # * (cap_flux_mpfa.flux * p_cap + cap_flux_mpfa.bound_flux * p_cap_bc)
            #     )
            #     - source_ad_t
            # )
            # This is kind of messy, but we do this to avoid inf values in the
            # equation system, which appear when using division.
            # invert_func = pp.ad.Function(partial(pow, exponent=-1), "invert")
            # fractional_flow_w = mobility_w * invert_func(mobility_t)
            fractional_flow_w = mobility_w / mobility_t
            saturation_eq = (
                porosity_ad * dt_s
                + div
                * (
                    fractional_flow_w * flux_t
                    - fractional_flow_w * mobility_n * (cap_flux_mpfa.flux * p_cap)
                )
                - source_ad_w
            )
            pressure_eq.set_name("Nonwetting pressure equation")
            saturation_eq.set_name("Wetting saturation equation")
        # Update the equation list.
        self._eq_manager.equations.update(
            {
                "pressure_eq": pressure_eq,
                "saturation_eq": saturation_eq,
            }
        )

    def _flux_w(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Wetting phase mass flux.

        SI Units: kg/s
        """
        p_w = self._ad.pressure_w
        p_w_bc = pp.ad.ParameterArray(self.w_flux_key, "bc_values", subdomains)
        # vector_source_subdomains = pp.ad.ParameterArray(
        #     param_keyword=self.flux_key,
        #     array_keyword="vector_source",
        #     subdomains=subdomains,
        # )
        flux_mpfa = pp.ad.MpfaAd(
            self.w_flux_key,
            subdomains,
        )
        flux: pp.ad.Operator = (
            flux_mpfa.flux * p_w
            + flux_mpfa.bound_flux * p_w_bc
            # + flux_mpfa.vector_source * vector_source_subdomains
        )
        flux.set_name("Wetting mass flux")
        return flux

    def _flux_n(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Nonwetting phase mass flux.

        SI Units: kg/s
        """
        p_n = self._ad.pressure_n
        p_n_bc = pp.ad.ParameterArray(self.n_flux_key, "bc_values", subdomains)
        # vector_source_subdomains = pp.ad.ParameterArray(
        #     param_keyword=self.flux_key,
        #     array_keyword="vector_source",
        #     subdomains=subdomains,
        # )
        flux_mpfa = pp.ad.MpfaAd(
            self.n_flux_key,
            subdomains,
        )
        flux: pp.ad.Operator = (
            flux_mpfa.flux * p_n
            + flux_mpfa.bound_flux * p_n_bc
            # + flux_mpfa.vector_source * vector_source_subdomains
        )
        flux.set_name("Nonwetting mass flux")
        return flux

    def _flux_t(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Total mass flux.

        This is always calculated in terms of the nonwetting pressure and the capillary
        pressure (i.e. in terms of the wetting Saturation). Note that, unlike in the
        phase flux functions, the mobilities are already included in this formulation.

        SI Units: kg/s
        """
        # Variables and parameters.
        p_n = self._ad.pressure_n
        p_bc = pp.ad.ParameterArray(self.n_flux_key, "bc_values", subdomains)
        p_cap_bc = pp.ad.ParameterArray(self.cap_flux_key, "bc_values", subdomains)
        viscosity_ad_w = pp.ad.Scalar(self._w_viscosity)
        viscosity_ad_n = pp.ad.Scalar(self._n_viscosity)
        # Spatial discretization operators.
        flux_mpfa = pp.ad.MpfaAd(self.n_flux_key, subdomains)
        cap_flux_mpfa = pp.ad.MpfaAd(self.cap_flux_key, subdomains)
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, subdomains)
        upwind_n = pp.ad.UpwindAd(self.n_flux_key, subdomains)
        # Compute cap pressure and relative permeabilities.
        p_cap = self._cap_pressure()
        mobility_w = upwind_w.upwind * (self._rel_perm_w() / viscosity_ad_w)
        mobility_n = upwind_n.upwind * (self._rel_perm_n() / viscosity_ad_n)
        # Add a small :math:\epsilon to the total mobility, to avoid division by zero.
        mobility_t = mobility_w + mobility_n + pp.ad.Scalar(1e-7)
        # Compute flux.
        flux_n: pp.ad.Operator = flux_mpfa.flux * p_n + flux_mpfa.bound_flux * p_bc
        flux_p_cap: pp.ad.Operator = (
            cap_flux_mpfa.flux * p_cap + cap_flux_mpfa.bound_flux * p_cap_bc
        )
        total_flux = mobility_t * flux_n - mobility_w * flux_p_cap
        total_flux.set_name("Total mass flux")
        return total_flux

    def _discretize(self) -> None:
        """Discretize all terms"""
        # t = time.time()
        self._eq_manager.discretize(self.mdg)
        # Fluid flux induced by the secondary variable. This needs to be discretized
        # once s.t. the Darcy flux can be computed for the secondary variable, s.t.
        # the upwind operator works correctly.
        if self._formulation == "w_pressure_w_saturation":
            flux_n = self._flux_n(self.mdg.subdomains())
            flux_n.discretize(self.mdg)
        elif self._formulation == "n_pressure_w_saturation":
            flux_w = self._flux_w(self.mdg.subdomains())
            flux_w.discretize(self.mdg)
        # logger.debug(f"Discretized in {time.time() - t} seconds")

    def assemble_linear_system(self) -> None:
        """Assemble the linearized system.

        The linear system is defined by the current state of the model.

        Attributes:
            linear_system is assigned.

        """
        t_0 = time.time()
        if self._use_ad:
            # Solve either for wetting or nonwetting pressure.
            if self.primary_pressure_var == self.pressure_w_var:
                variables = [self._ad.pressure_w, self._ad.saturation]
            elif self.primary_pressure_var == self.pressure_n_var:
                variables = [self._ad.pressure_n, self._ad.saturation]
            A, b = self._eq_manager.assemble_subsystem(
                eq_names=["pressure_eq", "saturation_eq"],
                variables=variables,
            )
        # A, b = self._eq_manager.assemble()
        self.linear_system = (A, b)
        # print(f"A: {A.todense()}")
        # print(f"b: {b}")
        logger.debug(f"Assembled linear system in {t_0-time.time():.2e} seconds.")

    # Newton loop.
    def before_newton_loop(self) -> None:
        """Set the starting estimate to the solution from the previous timestep."""
        logger.debug(
            f'{{"time step": {self.time_manager.time_index}, "time": {self.time_manager.time}}}'
        )
        self.time_manager._recomp_sol = False
        self._nonlinear_iteration = 0
        for sd, data in self.mdg.subdomains(return_data=True):
            assembled_variables = self.dof_manager.assemble_variable(
                grids=[sd],
                from_iterate=False,
            )
            pp.set_iterate(
                data, {self.pressure_w_var: assembled_variables[: sd.num_cells]}
            )
            pp.set_iterate(
                data,
                {
                    self.pressure_n_var: assembled_variables[
                        sd.num_cells : sd.num_cells * 2
                    ]
                },
            )
            pp.set_iterate(
                data,
                {self.saturation_var: assembled_variables[sd.num_cells * 2 :]},
            )
            # pp.set_iterate(
            #     data,
            #     {self.pressure_w_var: assembled_variables[: sd.num_cells]},
            # )
            # pp.set_iterate(
            #     data,
            #     {self.saturation_var: assembled_variables[sd.num_cells :]},
            # )
        if self._limit_saturation_change:
            self._prev_saturation: np.ndarray = self.dof_manager.assemble_variable(
                variables=[self.saturation_var], from_iterate=False
            )

    def before_newton_iteration(self) -> None:
        """Compute Darcy flux based on previous pressure solution to determine upstream
        direction.

        To evaluate the phase-mobilities separately, both the wetting as well as the
        nonwetting flux need to be computed.

        """
        # ? Does this suffice at each time step instead? -> Needs to happen at each
        # Newton iteration, because we are starting with a bad guess (previous timestep)
        # and improve towards the solution. We want to use discretization and Darcy flux
        # Evaluate the pressure of the secondary variable.
        secondary_pressure = self._ad.pressure_n - self._cap_pressure()
        secondary_pressure_sol = secondary_pressure.evaluate(self.dof_manager).val
        # Add padding for the two other variables.
        num_dof_one_var = self.mdg.subdomains()[0].num_cells
        if self.secondary_pressure_var == self.pressure_w_var:
            solution = np.insert(
                np.zeros(num_dof_one_var * 3), 0, secondary_pressure_sol
            )
        elif self.secondary_pressure_var == self.pressure_n_var:
            solution = np.insert(
                np.zeros(num_dof_one_var * 3), num_dof_one_var, secondary_pressure_sol
            )
        self.dof_manager.distribute_variable(
            solution,
            variables=[self.secondary_pressure_var],
            # Not additive, since we evaluate the current iterative variables to obtain
            # the solution.
            additive=False,
            to_iterate=True,
        )
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            keyword=self.w_flux_key,
            keyword_store=self.w_flux_key,
            p_name=self.pressure_w_var,
            from_iterate=True,
        )
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            keyword=self.n_flux_key,
            keyword_store=self.n_flux_key,
            p_name=self.pressure_n_var,
            from_iterate=True,
        )
        self._discretize()

    def after_newton_iteration(self, solution: np.ndarray) -> None:
        """Distribute solution variable.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.
        """
        # Insert the secondary pressure variable into the solution.
        num_dof_one_var = self.mdg.subdomains()[0].num_cells
        if self.secondary_pressure_var == self.pressure_w_var:
            solution = np.insert(solution, 0, np.zeros(num_dof_one_var))
        elif self.secondary_pressure_var == self.pressure_n_var:
            solution = np.insert(solution, num_dof_one_var, np.zeros(num_dof_one_var))
        # Distribute pressure and saturation variable.
        self.dof_manager.distribute_variable(
            solution,
            # variables=[self.primary_pressure_var, self.saturation_var],
            additive=True,
            to_iterate=True,
        )
        self._nonlinear_iteration += 1

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        """Check if the wetting saturation changed too much, export and move to the next
        time step.

        Parameters:
            solution: _description_
            errors: _description_
            iteration_counter: _description_
        """
        # If the saturation changes to much, decrease the time step and calculate again.
        if self._limit_saturation_change:
            new_saturation: np.ndarray = self.dof_manager.assemble_variable(
                variables=[self.saturation_var], from_iterate=False
            )
            if (
                np.max(np.abs(new_saturation - self._prev_saturation))
                > self._max_saturation_change
            ):
                # This is set to false again in ``before_newton_loop``. NOTE: This is
                # not a very nice solution, however, as of now I didn't find a way to
                # pass ``recompute_solution`` to ``time_manager.compute_time_step()`` in
                # ``run_time_dependent_model`` without the code getting really messy.
                self.time_manager._recomp_sol = True
                self.convergence_status = False
                logger.debug(
                    "Saturation grew to quickly. Trying again with a smaller time step."
                )
                return None

        # Distribute pressure and saturation variable.
        timestep_solution = self.dof_manager.assemble_variable(from_iterate=True)
        self.dof_manager.distribute_variable(timestep_solution, to_iterate=False)

        self.convergence_status = True
        self._export()
        logger.debug(f'{{"converged": {"true"}}}')

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        logger.debug(f"Failed on timestep {self.time_manager.time_index}")
        logger.debug(f"Error {errors} Newton iteration {iteration_counter}")
        logger.debug(f'{{"converged": {"false"}}}')
        raise ValueError("Newton iterations did not converge")

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu(
                # [self.pressure_w_var, self.pressure_n_var, self.saturation_var],
                # [self.pressure_n_var, self.saturation_var],
                [self.pressure_w_var, self.saturation_var],
                time_step=self.time_manager.time_index,
            )

    def _is_nonlinear_problem(self):
        return True

    def after_simulation(self):
        pass

    def _error_function_deriv(self) -> pp.ad.Operator:
        s = self._ad.saturation
        exp_func = pp.ad.Function(pp.ad.functions.exp, "exp")
        square_func = pp.ad.Function(partial(pow, exponent=2), "square")
        return self._yscale * exp_func(-self._xscale * square_func(s - self._offset))
