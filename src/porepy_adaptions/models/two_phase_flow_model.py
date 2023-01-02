"""This module contains an implementation of a base model for two-phase flow problems.

The model uses the pressure-saturation formulation for the wetting fluid:
    ..math:
        \nabla\cdot\mathbf{u}=
        \frac{\partial\phi S_w}{\partial t}+


Note, that the model is only implemented for domains without pressure.

TODO:
    - Denote units for all parameters, values, variables.
    - Implement gravity.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Union
from functools import partial
import numpy as np

import porepy as pp
from ..numerics.ad.functions import pow

# Setup logging.
logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


class _AdVariables:
    pressure: pp.ad.MergedVariable
    saturation: pp.ad.MergedVariable
    flux_discretization: Union[pp.ad.MpfaAd, pp.ad.TpfaAd]
    subdomains: list[pp.Grid]


class TwoPhaseFlow(pp.models.abstract_model.AbstractModel):
    """This is a shell class for two-phase  flow problems.

    This class is intended to provide a standardized setup, with all discretizations
    in place and reasonable parameter and boundary values. The intended use is to
    inherit from this class, and do the necessary modifications and specifications
    for the problem to be fully defined. The minimal adjustment needed is to
    specify the method create_grid(). The class also serves as parent for other
    model classes (CompressibleFlow).

    Public attributes:
        variable (str): Name assigned to the pressure variable in the
            highest-dimensional subdomain. Will be used throughout the simulations,
            including in ParaView export. The default variable name is 'p'.
        parameter_key (str): Keyword used to define parameters and discretizations.
        params (dict): Dictionary of parameters used to control the solution procedure.
            Some frequently used entries are file and folder names for export,
           mesh sizes...
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
        # Variables
        self.pressure_var: str = "p"
        self.saturation_var: str = "s"
        # Discretizations and params
        self.flux_key: str = "flux"
        self.cap_pressure_flux_key: str = "p_cap_flux"
        self.upwind_rel_perm_key: str = "rel_perm"
        self.params_key: str = "params"
        # Some options
        self._use_ad: bool = True
        self._mobility_bc: bool = True
        # Managers etc.
        self._ad = _AdVariables()
        self.exporter: pp.Exporter
        self.dof_manager: pp.DofManager
        self._eq_manager: pp.ad.EquationManager
        self.time_manager: pp.TimeManager
        # Time schedule.
        self._time_step: float = 0.2
        self._schedule: np.ndarray = np.array([0, 2.0])
        # Phase parameters
        self._w_viscosity_value: float = 1.0
        self._nw_viscosity_value: float = 1.0
        # Parameters for the capillary pressure function
        self._n_G: float = 1.0
        self._m_G: float = -2.0
        self._beta_G: float = 1.0
        # Relative permeability limited below
        self._limit_rel_perm: bool = False
        # Wetting saturation limited below
        self._limit_w_saturation: bool = False
        # Grid size
        self._grid_size: int = 20
        self._phys_size: int = 2

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
        """Homogeneous Neumann conditions on three sides, Dirichlet
        on one side to ensure existence of a unique solution."""
        all_bf, *_ = self._domain_boundary_sides(g)
        return pp.BoundaryCondition(g, all_bf[: self._grid_size], "dir")

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values. Dirichlet pressure equals the initial state
        pressure"""
        array = np.zeros(g.num_faces)
        array[0] = 0.3
        return array

    def _upwind_rel_perm_bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Boundary conditions for the saturation. We assume homogeneous Dirichlet bc
        for the saturation.

        Note that this is not necessarily needed and from a physical point of view, we
        can just assume no outflow and
        """
        return pp.BoundaryCondition(g, g.get_all_boundary_faces(), "dir")

    def _upwind_rel_perm_bc_values(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_faces)

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        """In the default model there is no source term."""
        return np.zeros(g.num_cells)

    def _nw_source(self, g: pp.Grid) -> np.ndarray:
        return np.zeros(g.num_cells)

    def _total_source(self, g: pp.Grid) -> np.ndarray:
        return self._w_source(g) + self._nw_source(g)

    # More matrix and phase parameters.
    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Unitary permeability.

        Units: m^2
        """
        return np.ones(g.num_cells)

    # ? Can viscosity be dynamic in a slightly compressible model?
    def _w_viscosity(self, g: pp.Grid) -> np.ndarray:
        return np.full(g.num_cells, self._w_viscosity_value)

    def _nw_viscosity(self, g: pp.Grid) -> np.ndarray:
        return np.full(g.num_cells, self._nw_viscosity_value)

    def _porosity(self, g: pp.Grid) -> np.ndarray:
        return np.ones(g.num_cells)

    # Cap pressure and relative permeability functions.
    def _cap_pressure(self, toggle_off: bool = False) -> pp.ad.Operator:
        """Capillary pressure computed with the ... model.

        .. math::
            p_c(s_w)=\frac{(s^{n_G})^{-m_G}-1}{\beta_G}

        Parameters:
            toggle_off: _description_, defaults to False

        Returns:
            _description_
        """
        s = self._ad.saturation
        if toggle_off:
            # TODO Make this look nicer (just return zero maybe? -> Doesn't work).
            return s * 0
        else:
            # Setup pp.ad.functions.pow
            pow_func = pp.ad.Function(
                partial(pow, exponent=-self._m_G * self._n_G), "pow"
            )
            return (pow_func(s) - 1) / self._beta_G

    def _w_rel_perm(self) -> pp.ad.Operator:
        """Wetting phase relative permeability pressure computed with the ... model.

        .. math::
            k_{r,ww}(s_w)=s_w^3
            or
            k_{r,w}(s_w)=\max\{s_w^3,0.0001^3\}

        Returns:
            _description_
        """
        s = self._ad.saturation
        cube_func = pp.ad.Function(partial(pow, exponent=3), "cube")
        # TODO: Fix this so it applies to all subdomains.
        if self._limit_rel_perm:
            array = np.full(self.mdg.subdomains()[0].num_cells, 0.0001)
            max_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var1=array), "max"
            )
            return cube_func(max_func(s))
        else:
            return cube_func(s)

    def _nw_rel_perm(self) -> pp.ad.Operator:
        """Non-wetting phase relative permeability pressure computed with the ... model.

        .. math::
            k_{r,nw}(s_w)=(1-s_w)^3
            or
            k_{r,nw}(s_w)=\max\{(1-s_w)^3,0.0001^3\}

        Returns:
            _description_
        """
        s = self._ad.saturation
        cube_func = pp.ad.Function(partial(pow, exponent=3), "cube")
        # TODO: Fix this so it applies to all subdomains.
        if self._limit_rel_perm:
            min_value = np.full(self.mdg.subdomains()[0].num_cells, 0.0001)
            max_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var1=min_value), "max"
            )
            return cube_func(max_func(1 - s))
        else:
            return cube_func(1 - s)

    # ! For later!
    # def _vector_source(self, g: Union[pp.Grid, pp.MortarGrid]) -> np.ndarray:
    #     """Zero vector source (gravity).

    #     To assign a gravity-like vector source, add a non-zero contribution in
    #     the last dimension:
    #         vals[-1] = - pp.GRAVITY_ACCELERATION * fluid_density
    #     """
    #     vals = np.zeros((self.mdg.dim_max(), g.num_cells))
    #     return vals

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
        logger.info("Grid created")

    def _set_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the flow problem.

        The parameter fields of the data dictionaries are updated for all
        subdomains and interfaces (of codimension 1).
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            # Boundary conditions and parameters
            diffusivity = pp.SecondOrderTensor(self._permeability(sd))
            pp.initialize_data(
                sd,
                data,
                self.flux_key,
                {
                    "bc": self._bc_type(sd),
                    "bc_values": self._bc_values(sd),
                    # ? Does this need to be total_source or wetting_source?
                    "source": self._w_source(sd),
                    "second_order_tensor": diffusivity,
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.upwind_rel_perm_key,
                {
                    "bc": self._upwind_rel_perm_bc_type(sd),
                    "bc_values": self._upwind_rel_perm_bc_values(sd),
                    # Initialize the darcy flux to be one on all cell interfaces.
                    "darcy_flux": np.ones(sd.num_faces),
                },
            )
            pp.initialize_data(
                sd,
                data,
                self.params_key,
                {
                    "porosity": self._porosity(sd),
                    "w_viscosity": self._w_viscosity(sd),
                    "nw_viscosity": self._nw_viscosity(sd),
                    "total_source": self._total_source(sd),
                },
            )
            # Set initial condition.

    def _initial_condition(self) -> None:
        """Set initial values for wetting pressure and saturation."""
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.set_state(data, {self.pressure_var: np.full(sd.num_cells, 0.3)})
            pp.set_state(data, {self.saturation_var: np.full(sd.num_cells, 0.3)})

    def _assign_variables(self) -> None:
        """
        Assign primary variables to subdomains and interfaces of the mixed-dimensional
        grid.
        """
        for sd, data in self.mdg.subdomains(return_data=True):
            # One dof per cell for both variables
            data[pp.PRIMARY_VARIABLES] = {
                self.pressure_var: {"cells": 1},
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
        self._ad.pressure = self._eq_manager.merge_variables(
            [(sd, self.pressure_var) for sd in self.mdg.subdomains()]
        )
        self._ad.saturation = self._eq_manager.merge_variables(
            [(sd, self.saturation_var) for sd in self.mdg.subdomains()]
        )

    def _assign_equations(self) -> None:
        """Define equations."""
        dt: float = self.time_manager.dt
        subdomains = [sd for sd in self.mdg.subdomains()]
        self._ad.subdomains = subdomains
        if len(list(self.mdg.subdomains(dim=self.mdg.dim_max()))) != 1:
            raise NotImplementedError("This will require further work")

        # Ad representation of discretizations
        div = pp.ad.Divergence(subdomains)
        flux_mpfa = pp.ad.MpfaAd(self.flux_key, subdomains)
        cap_pressure_flux_mpfa = pp.ad.MpfaAd(self.cap_pressure_flux_key, subdomains)
        upwind = pp.ad.UpwindAd(self.upwind_rel_perm_key, subdomains)

        # Ad saturation variable, p is not needed as the flux is computed in a seperate
        # method (``_flux``).

        # TODO: Fix this so it applies to all subdomains.
        if self._limit_w_saturation:
            min_s = np.full(self.mdg.subdomains()[0].num_cells, 0.0001)
            max_func = pp.ad.Function(
                partial(pp.ad.functions.maximum, var1=min_s), "max"
            )
            s = max_func(self._ad.saturation)
        else:
            s = self._ad.saturation
        s_prev = self._ad.saturation.previous_timestep()

        # Ad sorce
        w_source_ad = pp.ad.ParameterArray(self.flux_key, "source", subdomains)
        total_source_ad = pp.ad.ParameterArray(
            self.params_key, "total_source", subdomains
        )

        # Ad parameters
        w_viscosity_ad = pp.ad.ParameterArray(
            self.params_key, "w_viscosity", subdomains
        )
        nw_viscosity_ad = pp.ad.ParameterArray(
            self.params_key, "nw_viscosity", subdomains
        )
        porosity_ad = pp.ad.ParameterArray(self.params_key, "porosity", subdomains)

        # Option to cut the saturation s.t. it does not become negative.
        # zeros = np.full(mdg.subdomains()[0].num_cells, .0)
        # nonneg_func = pp.ad.Function(partial(pp.ad.functions.maximum, var1=zeros), 'nonneg')
        # s_nonneg = nonneg_func(s)

        # Compute cap pressure and relative permeabilities.
        p_cap = self._cap_pressure(toggle_off=False)
        w_mobility = self._w_rel_perm() / w_viscosity_ad
        nw_mobility = self._nw_rel_perm() / nw_viscosity_ad
        total_mobility = w_mobility + nw_mobility

        flux = self._flux(subdomains)

        # Mobility (saturation) bc. By default no bc on the saturation are applied.
        if self._mobility_bc:
            # TODO Make this more general, i.e. a real function instead of this helper
            # stuff.
            s_bc = pp.ad.ParameterArray(
                self.upwind_rel_perm_key, "bc_values", subdomains
            )

            w_mobility_bc = s_bc
            nw_mobility_bc = 1 - w_mobility_bc
            total_mobility_bc = w_mobility_bc + nw_mobility_bc

            # ! For now only homogeneous Dir. bc everywhere (on all mobilities, i.e. nothing
            # can flow out).
            w_mobility_with_bc = (
                upwind.upwind * w_mobility + upwind.bound_transport_dir * s_bc
            )
            nw_mobility_with_bc = (
                upwind.upwind * nw_mobility + upwind.bound_transport_dir * (1 - s_bc)
            )
            total_mobility_with_bc = (
                upwind.upwind * total_mobility
                + upwind.bound_transport_dir * (1 - s_bc + s_bc)
            )
        else:
            w_mobility_with_bc = upwind.upwind * w_mobility
            nw_mobility_with_bc = upwind.upwind * nw_mobility
            total_mobility_with_bc = upwind.upwind * total_mobility

        # Ad equations
        pressure_eq = (
            div
            * (
                total_mobility_with_bc * flux
                + nw_mobility_with_bc * (flux_mpfa.flux * p_cap)
            )
            - total_source_ad
        )
        saturation_eq = (
            porosity_ad * (s - s_prev)
            + dt * (div * (w_mobility_with_bc * flux))
            - dt * w_source_ad
        )
        pressure_eq.set_name("Total flow on subdomains.")
        saturation_eq.set_name("Wetting fluid flow on subdomains.")

        # Add to the equation list:
        self._eq_manager.equations.update(
            {"pressure_eq": pressure_eq, "saturation_eq": saturation_eq}
        )

    def _flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Fluid flux."""
        p = self._ad.pressure
        p_bc = pp.ad.ParameterArray(self.flux_key, "bc_values", subdomains)
        flux_mpfa = pp.ad.MpfaAd(self.flux_key, subdomains)
        flux: pp.ad.Operator = flux_mpfa.flux * p + flux_mpfa.bound_flux * p_bc
        flux.set_name("Total flux")
        return flux

    def _discretize(self) -> None:
        """Discretize all terms"""
        logger.info(f"Discretizing and assembling")
        t = time.time()
        self._eq_manager.discretize(self.mdg)
        logger.info(f"Discretized in {time.time() - t} seconds")

    # Newton loop.
    def before_newton_loop(self) -> None:
        self._nonlinear_iteration = 0
        for sd, data in self.mdg.subdomains(return_data=True):
            variables_assembled = self.dof_manager.assemble_variable(
                grids=[sd], from_iterate=False
            )
            pp.set_iterate(
                data, {self.pressure_var: variables_assembled[: sd.num_cells]}
            )
            pp.set_iterate(
                data, {self.saturation_var: variables_assembled[sd.num_cells :]}
            )

    def before_newton_iteration(self) -> None:
        """Compute Darcy flux based on previous pressure solution to determine upstream
        direction."""
        # ? Does this suffice at each time step instead?
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            keyword=self.flux_key,
            keyword_store=self.upwind_rel_perm_key,
            p_name=self.pressure_var,
            from_iterate=True,
        )
        self._discretize()

    def after_newton_iteration(self, solution: np.ndarray) -> None:
        """
        Scatters the solution vector for current iterate.

        Parameters:
            solution_vector (np.array): solution vector for the current iterate.

        """
        self.dof_manager.distribute_variable(
            solution,
            additive=True,
            to_iterate=True,
        )
        self._nonlinear_iteration += 1

    def after_newton_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:

        timestep_solution = self.dof_manager.assemble_variable(from_iterate=True)
        self.dof_manager.distribute_variable(timestep_solution, to_iterate=False)
        self.convergence_status = True
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        logger.info(f"Failed on timestep {self.time_manager.time_index}")
        raise ValueError("Newton iterations did not converge")

    def _export(self):
        if hasattr(self, "exporter"):
            self.exporter.write_vtu(
                [self.pressure_var, self.saturation_var],
                time_step=self.time_manager.time_index,
            )

    def _is_nonlinear_problem(self):
        return True

    def after_simulation(self):
        pass
