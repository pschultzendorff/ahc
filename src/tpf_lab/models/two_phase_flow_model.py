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
import time
from typing import Optional, Union
from functools import partial
import numpy as np

import porepy as pp
from src.tpf_lab.numerics.ad.functions import pow

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
        self._formulation: str = "w_pressure_w_saturation"
        # Variables
        self.pressure_var: str
        r"""Name for the pressure variable :math:p_\alpha . Depending on the chosen
         formulation, this is either the wetting or the nonetting pressure.
        """
        if self._formulation == "w_pressure_w_saturation":
            self.pressure_var = "w_pressure"
        elif self._formulation == "n_pressure_w_saturation":
            self.pressure_var = "n_pressure"
        self.saturation_var: str = "w_saturation"
        """Name for the wetting saturation variable :math:S_w ."""
        # Discretizations and params
        self.params_key: str = "params"
        self.cap_params_key: str = "cap_params"
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
        self._cap_pressure_model: str = "Brooks-Corey"
        # van Genuchten model
        self._n_g: float = 2.0
        self._m_g: float = 2 / 3
        self._beta_g: float = 1.0
        self._residual_w_saturation: float = 0.3
        self._residual_n_saturation: float = 0.0
        # Brooks Corey model
        self._entry_pressure: float = 0.1
        self._n_b: int = 1
        # Parameters for the relative permeability function
        self._rel_perm_model: str = "Brooks-Corey"
        self._n1: int = 2
        self._n2: int = int(1 + 2 / self._n_b)
        self._n3: int = 1
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
        # Limit saturation growth
        self._limit_saturation_change: bool = False
        """If this is set to ``True``, the Newton method fails, if the final solution
        differs from the previous timestep by more than ``self._max_saturation_change``
        in any grid cell. The timestep is then shortened and recalculated.
        """
        self._max_saturation_change: float = 0.2

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
        south = self._domain_boundary_sides(g).south
        return pp.BoundaryCondition(g, south, "dir")

    def _bc_values(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous boundary values. Dirichlet pressure equals the initial state
        pressure"""
        array = np.zeros(g.num_faces)
        return array

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting source.

        In the default model there is no source term.

        SI Units: m^d/s
        """
        return np.zeros(g.num_cells)

    def _n_source(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting source.

        SI Units: m^d/s
        """
        return np.zeros(g.num_cells)

    def _total_source(self, g: pp.Grid) -> np.ndarray:
        """Volumetric total source; sum of the wetting and nonetting source.

        SI Units: m^d/s
        """
        return self._w_source(g) + self._n_source(g)

    # More matrix and phase parameters.
    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Solid permeability.

        SI Units: m^2
        """
        return np.full(g.num_cells, 1)

    def _porosity(self, g: pp.Grid) -> np.ndarray:
        return np.full(g.num_cells, 1.0)

    # Cap pressure and relative permeability functions.
    def _cap_pressure(self, toggle: bool = True) -> pp.ad.Operator:
        r"""Capillary pressure computed with the ... model.

        .. math::
            p_c(S_w)=\frac{(\hat{S}_w^{m_g}-1)^{-n_g}}{\beta_g}
            \text{for}
            \hat{S}_w=\frac{S_w-S_w^{min}}{S_w^{max}-S_w^{min}}\\

        Parameters:
            toggle: _description_, defaults to False
        """
        s = self._ad.saturation
        normalized_s = (s - self._residual_w_saturation) / (
            1 - self._residual_n_saturation - self._residual_w_saturation
        )
        if not toggle:
            # TODO Make this look nicer (just return zero maybe? -> Doesn't work).
            return s * 0
        elif self._cap_pressure_model == "van Genuchten":
            # Setup pp.ad.functions.pow
            pow_func_1 = pp.ad.Function(partial(pow, exponent=self._m_g), "pow")
            pow_func_2 = pp.ad.Function(partial(pow, exponent=-self._n_g), "pow")
            return pow_func_2(pow_func_1(normalized_s) - 1) / self._beta_g
        elif self._cap_pressure_model == "Brooks-Corey":
            pow_func = pp.ad.Function(partial(pow, exponent=-self._n_b), "pow")
            return pp.ad.Scalar(self._entry_pressure) * pow_func(normalized_s)
        else:
            return s * 0

    def _w_rel_perm(self) -> pp.ad.Operator:
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
        # TODO: Fix this so it applies to all subdomains.
        if self._rel_perm_model == "Corey":
            # TODO: Add constant factor to power law.
            cube_func = pp.ad.Function(partial(pow, exponent=3), "cube")
            if self._limit_rel_perm:
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

    def _n_rel_perm(self) -> pp.ad.Operator:
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
        # TODO: Fix this so it applies to all subdomains.
        if self._rel_perm_model == "Corey":
            # TODO: Add constant factor to power law.
            cube_func = pp.ad.Function(partial(pow, exponent=3), "cube")
            if self._limit_rel_perm:
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

    def _w_vector_source(
        self, g: Union[pp.Grid, pp.MortarGrid], toggle=False
    ) -> np.ndarray:
        """Zero vector source (gravity).

        # TODO: fix this! Right now it doesn't change anything.
        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * fluid_density
        """
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        if toggle:
            vals[-1] = -pp.GRAVITY_ACCELERATION * self._w_density
        return vals

    def _n_vector_source(
        self, g: Union[pp.Grid, pp.MortarGrid], toggle=False
    ) -> np.ndarray:
        """Zero vector source (gravity).

        # TODO: fix this! Right now it doesn't change anything.
        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * fluid_density
        """
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        if toggle:
            vals[-1] = -pp.GRAVITY_ACCELERATION * self._n_density
        return vals

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
            gravity = self._w_vector_source(sd)
            pp.initialize_data(
                sd,
                data,
                self.params_key,
                {
                    "bc": self._bc_type(sd),
                    "bc_values": self._bc_values(sd),
                    # ? Does this need to be total_source or wetting_source?
                    "wetting_source": self._w_source(sd),
                    "vector_source": gravity.ravel("F"),
                    "second_order_tensor": diffusivity,
                    "porosity": self._porosity(sd),
                    "total_source": self._total_source(sd),
                    "darcy_flux": np.ones(sd.num_faces),
                },
            )
            all_bf, *_ = self._domain_boundary_sides(sd)
            pp.initialize_data(
                sd,
                data,
                self.cap_params_key,
                {
                    "bc": pp.BoundaryCondition(sd, all_bf, "neu"),
                    "bc_values": np.zeros(sd.num_faces),
                    "second_order_tensor": diffusivity,
                },
            )

    def _initial_condition(self) -> None:
        """Set initial values for wetting pressure and saturation."""
        for sd, data in self.mdg.subdomains(return_data=True):
            pp.set_state(data, {self.pressure_var: np.full(sd.num_cells, 0.0)})
            pp.set_state(data, {self.saturation_var: np.full(sd.num_cells, 0.5)})

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
        subdomains = [sd for sd in self.mdg.subdomains()]
        self._ad.subdomains = subdomains
        if len(list(self.mdg.subdomains(dim=self.mdg.dim_max()))) != 1:
            raise NotImplementedError("This will require further work")

        # Ad representation of discretizations
        div = pp.ad.Divergence(subdomains)
        flux_mpfa = pp.ad.MpfaAd(self.params_key, subdomains)
        upwind = pp.ad.UpwindAd(self.params_key, subdomains)
        cap_flux_mpfa = pp.ad.MpfaAd(self.cap_params_key, subdomains)

        # Ad saturation variables, pressure is not needed as the flux is computed in a
        # seperate method (``_flux``).
        s = self._ad.saturation
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s = pp.ad.time_derivatives.dt(s, dt)

        # Ad source
        w_source_ad = pp.ad.ParameterArray(
            self.params_key, "wetting_source", subdomains
        )
        total_source_ad = pp.ad.ParameterArray(
            self.params_key, "total_source", subdomains
        )

        # Ad parameters
        w_viscosity_ad = pp.ad.Scalar(self._w_viscosity)
        n_viscosity_ad = pp.ad.Scalar(self._n_viscosity)
        porosity_ad = pp.ad.ParameterArray(self.params_key, "porosity", subdomains)

        # Compute cap pressure and relative permeabilities.
        p_cap = self._cap_pressure(toggle=False)
        w_mobility = upwind.upwind * (self._w_rel_perm() / w_viscosity_ad)
        n_mobility = upwind.upwind * (self._n_rel_perm() / n_viscosity_ad)
        total_mobility = w_mobility + n_mobility

        # Wetting fluid flux.
        flux = self._wetting_flux(subdomains)

        # Ad equations
        if self._formulation == "w_pressure_w_saturation":
            pressure_eq = (
                div
                * (total_mobility * flux + n_mobility * (cap_flux_mpfa.flux * p_cap))
                - total_source_ad
            )
            saturation_eq = (
                porosity_ad * dt_s + (div * (w_mobility * flux)) - w_source_ad
            )
            pressure_eq.set_name("Pressure equation")
            saturation_eq.set_name("Saturation equation")
        elif self._formulation == "n_pressure_w_saturation":
            # TODO: Implement this.
            pass
        pressure_eq.set_name("Pressure equation")
        saturation_eq.set_name("Saturation equation")
        # Add to the equation list.
        self._eq_manager.equations.update(
            {"pressure_eq": pressure_eq, "saturation_eq": saturation_eq}
        )

    def _wetting_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Wetting fluid mass flux.

        SI Units: kg/s
        """
        p = self._ad.pressure
        p_bc = pp.ad.ParameterArray(self.params_key, "bc_values", subdomains)
        vector_source_subdomains = pp.ad.ParameterArray(
            param_keyword=self.params_key,
            array_keyword="vector_source",
            subdomains=subdomains,
        )
        flux_mpfa = pp.ad.MpfaAd(self.params_key, subdomains)
        flux: pp.ad.Operator = (
            flux_mpfa.flux * p
            + flux_mpfa.bound_flux * p_bc
            # + flux_mpfa.vector_source * vector_source_subdomains
        )
        flux.set_name("Wetting mass flux")
        return flux

    def _total_flux(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        p = self._ad.pressure
        w_viscosity_ad = pp.ad.ParameterArray(
            self.params_key, "w_viscosity", subdomains
        )
        n_viscosity_ad = pp.ad.ParameterArray(
            self.params_key, "n_viscosity", subdomains
        )
        upwind = pp.ad.UpwindAd(self.params_key, subdomains)
        # Compute cap pressure and relative permeabilities.
        p_cap = self._cap_pressure(toggle=False)
        w_mobility = upwind.upwind * (self._w_rel_perm() / w_viscosity_ad)
        n_mobility = upwind.upwind * (self._n_rel_perm() / n_viscosity_ad)
        total_mobility = w_mobility + n_mobility
        p_bc = pp.ad.ParameterArray(self.params_key, "bc_values", subdomains)
        flux_mpfa = pp.ad.MpfaAd(self.params_key, subdomains)
        w_flux: pp.ad.Operator = flux_mpfa.flux * p + flux_mpfa.bound_flux * p_bc
        cap_flux: pp.ad.Operator = flux_mpfa.flux * p_cap
        total_flux = total_mobility * w_flux - w_mobility * cap_flux
        total_flux.set_name("Total mass flux")
        return total_flux

    def _discretize(self) -> None:
        """Discretize all terms"""
        logger.info(f"Discretizing and assembling")
        t = time.time()
        self._eq_manager.discretize(self.mdg)
        logger.info(f"Discretized in {time.time() - t} seconds")

    # Newton loop.
    def before_newton_loop(self) -> None:
        """Set the starting estimate to the solution from the previous timestep."""
        logger.info(
            f"Time step {self.time_manager.time_index} at time \
                {self.time_manager.time:.1e} of {self.time_manager.time_final:.1e} \
                with time step {self.time_manager.dt:.1e}"
        )
        self.time_manager._recomp_sol = False
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
        if self._limit_saturation_change:
            self._prev_saturation: np.ndarray = self.dof_manager.assemble_variable(
                variables=[self.saturation_var], from_iterate=False
            )

    def before_newton_iteration(self) -> None:
        """Compute Darcy flux based on previous pressure solution to determine upstream
        direction."""
        # ? Does this suffice at each time step instead? -> Needs to happen at each
        # Newton iteration, because we are starting with a bad guess (previous timestep)
        # and improve towards the solution. We want to use discretization and Darcy flux
        # based on the best approximation.
        pp.fvutils.compute_darcy_flux(
            self.mdg,
            keyword=self.params_key,
            keyword_store=self.params_key,
            p_name=self.pressure_var,
            from_iterate=True,
        )
        self._discretize()

    def after_newton_iteration(self, solution: np.ndarray) -> None:
        """Distribute solution variable.

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
                logger.info(
                    "Saturation grew to quickly. Trying again with a smaller time step."
                )
                return None
        timestep_solution = self.dof_manager.assemble_variable(from_iterate=True)
        self.dof_manager.distribute_variable(timestep_solution, to_iterate=False)
        self.convergence_status = True
        self._export()

    def after_newton_failure(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        logger.info(f"Failed on timestep {self.time_manager.time_index}")
        logger.info(f"Error {errors} Newton iteration {iteration_counter}")
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
