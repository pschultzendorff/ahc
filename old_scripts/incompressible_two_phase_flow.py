from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import numpy as np
import scipy.sparse.linalg as spla

import porepy as pp

TIME_STEP: float = 0.2
SCHEDULE: np.ndarray = np.array([0, 20.0])

PERMEABILITY: float = 1.0
WETTING_VISCOSITY: float = 1.0
NONWETTING_VISCOSITY: float = 1.0

TOL: float = 0.00000001

GRID_SIZE: int = 20
PHYS_SIZE: int = 2

# Setup logging.
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# Create a tetrahedral grid on a simple box.
logger.info("creating grid")
cell_dims: np.ndarray = np.array(
    [
        GRID_SIZE,
        GRID_SIZE,
    ]
)
phys_dims: np.ndarray = np.array(
    [
        PHYS_SIZE,
        PHYS_SIZE,
    ]
)
g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
g_cart.compute_geometry()
# g_tetra: pp.TetrahedralGrid = pp.TetrahedralGrid(g_cart.nodes)
# g_tetra.compute_geometry()
mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart]])
box: dict = pp.bounding_box.from_points(
    np.array(
        [
            [
                0,
                0,
            ],
            [
                GRID_SIZE,
                GRID_SIZE,
            ],
        ]
    ).T
)
subdomains = [sd for sd in mdg.subdomains()]

# Define parameters and discretization.
flux_key: str = "flux"
cap_pressure_flux_key: str = "cap_pressure_flux"
upwind_rel_perm_key: str = "rel_perm"
params_key: str = "params"
# Pressure and the saturation of the wetting fluid.
pressure_var: str = "pressure"
saturation_var: str = "saturation"

# Boundary conditions for the pressure. We assume the pressure to be zero on the
# boundary.
def bc_type(g: pp.Grid) -> pp.BoundaryCondition:
    return pp.BoundaryCondition(g, g.get_all_boundary_faces(), "dir")


def bc_values(g: pp.Grid) -> np.ndarray:
    return np.zeros(g.num_faces)


# Boundary conditions for the saturation. We assume the transmissibility to be one on
# the boundary (i.e. Neumann type conditions for the saturation).
def upwind_rel_perm_bc_type(g: pp.Grid) -> pp.BoundaryCondition:
    return pp.BoundaryCondition(g, g.get_all_boundary_faces(), "dir")


def upwind_rel_perm_bc_values(g: pp.Grid) -> np.ndarray:
    return np.zeros(g.num_faces)


# Source terms. There is a source for the wetting fluid somewhere inside the domain.
def w_source(g: pp.Grid) -> np.ndarray:
    array = np.zeros(g.num_cells)
    array[189:191] = 1
    array[209:211] = 1
    return array


def nw_source(g: pp.Grid) -> np.ndarray:
    return np.zeros(g.num_cells)


def total_source(g: pp.Grid) -> np.ndarray:
    return w_source(g) + nw_source(g)


# More model parameters.
def porosity(g: pp.Grid) -> np.ndarray:
    return np.ones(g.num_cells)


def w_viscosity(g: pp.Grid) -> np.ndarray:
    return np.full(g.num_cells, WETTING_VISCOSITY)


def nw_viscosity(g: pp.Grid) -> np.ndarray:
    return np.full(g.num_cells, NONWETTING_VISCOSITY)


def ones(g: pp.Grid) -> np.ndarray:
    """Helper function to generate ``np.ones`` for grids."""
    return np.ones(g.num_cells)


def cap_pressure(s: pp.ad.MergedVariable, toggle_off: bool = False) -> pp.ad.Operator:
    if toggle_off:
        # TODO Make this look nicer (just return zero maybe? -> Doesn't work).
        return s * 0
    else:
        N_G: float = 1.0
        M_G: float = -2.0
        BETA_G: float = 1.0
        # Setup pp.ad.functions.pow
        pow_func = pp.ad.Function(
            partial(pp.ad.functions.pow, exponent=-M_G * N_G), "pow"
        )
        # TODO Check that this works in all situations. -> See pp.ad.functions.pow for
        # details.
        # ! Otherwise use the inbuilt __pow__ function in Ad_array.
        # return (s**(-M_G*N_G) - 1)/BETA_G
        return (pow_func(s) - 1) / BETA_G


# Set variables and bc for all subdomains.
logger.info("setting parameters")
for sd, data in mdg.subdomains(return_data=True):
    # One dof per cell for both variables
    data[pp.PRIMARY_VARIABLES] = {
        pressure_var: {"cells": 1},
        saturation_var: {"cells": 1},
    }
    # Boundary conditions and parameters
    # TODO Change diffusity to a function s.t. it can depend on the rock.
    # ! This must depend on viscosity and permeability.
    # diffusivity = pp.SecondOrderTensor(np.ones(sd.num_cells))
    diffusivity_array = np.ones(sd.num_cells)
    diffusivity_array[109:111] = 0.05
    diffusivity = pp.SecondOrderTensor(diffusivity_array)
    pp.initialize_data(
        sd,
        data,
        flux_key,
        {
            "bc": bc_type(sd),
            "bc_values": bc_values(sd),
            # ? Does this need to be total_source or wetting_source?
            "source": w_source(sd),
            "second_order_tensor": diffusivity,
        },
    )
    pp.initialize_data(
        sd,
        data,
        upwind_rel_perm_key,
        {
            "bc": upwind_rel_perm_bc_type(sd),
            "bc_values": upwind_rel_perm_bc_values(sd),
            "darcy_flux": -np.ones(sd.num_faces),
        },
    )
    pp.initialize_data(
        sd,
        data,
        params_key,
        {
            "porosity": porosity(sd),
            "w_viscosity": w_viscosity(sd),
            "nw_viscosity": nw_viscosity(sd),
            "total_source": total_source(sd),
            "ones": ones(sd),
        },
    )
    # Set initial condition.
    pp.set_state(data, {pressure_var: np.full(sd.num_cells, 0.3)})
    pp.set_state(data, {saturation_var: np.full(sd.num_cells, 0.3)})

logger.info("initializing ad objects")
dof_manager: pp.DofManager = pp.DofManager(mdg)
eq_manager: pp.ad.EquationManager = pp.ad.EquationManager(mdg, dof_manager)
time_manager: pp.TimeManager = pp.TimeManager(SCHEDULE, TIME_STEP, constant_dt=True)

# Merge variables. The merged variables can then be used in equations.
p: pp.ad.MergedVariable = eq_manager.merge_variables(
    [(sd, pressure_var) for sd in mdg.subdomains()]
)
s: pp.ad.MergedVariable = eq_manager.merge_variables(
    [(sd, saturation_var) for sd in mdg.subdomains()]
)

# Define div and MPFA and upwind operator.
div = pp.ad.Divergence(mdg.subdomains())
flux_mpfa = pp.ad.MpfaAd(flux_key, mdg.subdomains())
cap_pressure_flux_mpfa = pp.ad.MpfaAd(cap_pressure_flux_key, mdg.subdomains())
upwind = pp.ad.UpwindAd(upwind_rel_perm_key, mdg.subdomains())

# Source and bc.
w_source_ad = pp.ad.ParameterArray(flux_key, "source", subdomains)
total_source_ad = pp.ad.ParameterArray(params_key, "total_source", subdomains)
p_bc = pp.ad.ParameterArray(flux_key, "bc_values", subdomains)
s_bc = pp.ad.ParameterArray(upwind_rel_perm_key, "bc_values", subdomains)

# Time loop.
while time_manager.time < time_manager.time_final:
    # TODO makes this more general for mdgs with multiple subdomains.
    logger.info(f"time step {time_manager.time_index} time {time_manager.time}")
    dt = time_manager.compute_time_step(iterations=1)

    for sd, data in mdg.subdomains(return_data=True):
        variables_assembled = dof_manager.assemble_variable(
            grids=[sd], from_iterate=False
        )
        pp.set_iterate(data, {pressure_var: variables_assembled[: sd.num_cells]})
        pp.set_iterate(data, {saturation_var: variables_assembled[sd.num_cells :]})

    # TODO Do the equations need to be defined at every iteration? At every time
    # step? How to handle ``dt``?

    # Equation.
    # Get parameters
    porosity_ad = pp.ad.ParameterArray(params_key, "porosity", subdomains)
    w_viscosity_ad = pp.ad.ParameterArray(params_key, "w_viscosity", subdomains)
    nw_viscosity_ad = pp.ad.ParameterArray(params_key, "nw_viscosity", subdomains)
    ones_ad = pp.ad.ParameterArray(params_key, "ones", subdomains)

    # Option to cut the saturation s.t. it does not become negative.
    # zeros = np.full(mdg.subdomains()[0].num_cells, .0)
    # nonneg_func = pp.ad.Function(partial(pp.ad.functions.maximum, var1=zeros), 'nonneg')
    # s_nonneg = nonneg_func(s)
    # s_prev = s.previous_timestep()
    p_cap = cap_pressure(s, toggle_off=False)
    cube_func = pp.ad.Function(partial(pp.ad.functions.pow, exponent=3), "cube")
    w_rel_perm = cube_func(s)
    # Do I need to use ones here or does ``1-s_nonneg`` suffice?
    # TODO change the functions to be close to zero, if the saturation is negative.
    nw_rel_perm = cube_func(ones_ad - s)
    w_mobility = w_rel_perm / w_viscosity_ad
    nw_mobility = nw_rel_perm / nw_viscosity_ad
    total_mobility = w_mobility + nw_mobility

    flux = flux_mpfa.flux * p + flux_mpfa.bound_flux * p_bc

    # TODO Make this more general, i.e. a real function instead of this helper stuff.
    w_mobility_bc = s_bc
    nw_mobility_bc = np.ones_like(w_mobility_bc) - w_mobility_bc
    total_mobility_bc = w_mobility_bc + nw_mobility_bc

    # ! For now only homogeneous Dir. bc everywhere (on all mobilities, i.e. nothing can
    # move out).
    w_mobility_with_bc = upwind.upwind * w_mobility + upwind.bound_transport_dir * s_bc
    nw_mobility_with_bc = upwind.upwind * nw_mobility + upwind.bound_transport_dir * (
        1 - s_bc
    )
    total_mobility_with_bc = (
        upwind.upwind * total_mobility + upwind.bound_transport_dir * (1 - s_bc + s_bc)
    )

    # TODO Add total mobility to the boundary flux -> Calculate the equations for
    # this.
    # TODO Fix the saturation bc. Do we need different bc for the different mobilities?
    # ? Can the total mobility even be zero as a boundary condition.
    # ! Boundary condition is zero flux in normal direction on all boundaries, not
    # explicitely on p and s.
    # If we have homogeneous Dir. bc for s, then ``nw_rel_perm`` equals 1 on the
    # boundary. ->
    pressure_eq = (
        div
        * (
            total_mobility_with_bc * flux
            + nw_mobility_with_bc * (flux_mpfa.flux * p_cap)
        )
        - total_source_ad
    )
    saturation_eq = (
        porosity_ad * (s - s.previous_timestep())
        + dt * (div * (w_mobility_with_bc * flux))
        - dt * w_source_ad
    )

    eqs = {"pressure_eq": pressure_eq, "saturation_eq": saturation_eq}
    eq_manager.equations.update(eqs)

    # Newton loop.
    logger.info(f"\t Newton loop")
    converged: bool = False
    iteration_count: int = 0
    # Discretize once, s.t. the darcy_flux can be computed.
    eq_manager.discretize(mdg)

    while not converged:
        # Before Newton iteration compute the new Darcy flux and setup the equation.
        # TODO This is done explicitly! Try a smaller time step.
        pp.fvutils.compute_darcy_flux(
            mdg, keyword="flux", keyword_store=upwind_rel_perm_key, from_iterate=True
        )

        # Discretize and assemble the equations.
        logger.info(f"\t discretizing and assembling")
        eq_manager.discretize(mdg)
        A, b = eq_manager.assemble()
        # A, b = eq_manager.assemble_subsystem(["pressure_eq"], variables=[p])
        logger.info(f"\t solving")
        solution = spla.spsolve(A, b)

        # Distribute the variable to the grid. Since we are in a Newton loop, use
        # additive.
        dof_manager.distribute_variable(
            values=solution,
            variables=[pressure_var, saturation_var],
            additive=True,
            to_iterate=True,
        )

        # After Newton iteration
        iteration_count += 1
        is_linear = False
        if is_linear:
            converged = True
        error: float = np.linalg.norm(b)
        logger.info(f"\t\t Newton iteration {iteration_count}; error {error}")
        if error <= TOL:
            converged = True

    # Distribute the variable to the grid. Since we are using the Newton method, use
    # ``additive``.
    timestep_solution = dof_manager.assemble_variable(
        from_iterate=True,
        variables=[pressure_var, saturation_var],
    )
    timestep_pressure_solution = dof_manager.assemble_variable(
        from_iterate=True,
        variables=[pressure_var, saturation_var],
    )
    # print(timestep_pressure_solution)
    dof_manager.distribute_variable(
        timestep_solution, variables=[pressure_var, saturation_var], to_iterate=False
    )
    # TODO Add some postprocess function to calculate the nonwetting pressure &
    # saturation.

    # Visualize.
    exporter = pp.Exporter(
        mdg, f"two_phase_flow_full", folder_name="/home/peter/saves/"
    )
    exporter.write_vtu(
        [pressure_var, saturation_var], time_step=time_manager.time_index
    )

    # Increase time.
    time_manager.increase_time()
    time_manager.increase_time_index()
