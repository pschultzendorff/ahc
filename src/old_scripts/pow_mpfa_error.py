from __future__ import annotations

import logging
from functools import partial
from typing import Optional

import numpy as np
import scipy.sparse.linalg as spla

import porepy as pp


# Setup logging.
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# Create a tetrahedral grid on a simple box.
logger.info("creating grid")
cell_dims: np.ndarray = np.array([3, 3, 1])
phys_dims_1: np.ndarray = np.array([3, 3, 1])
phys_dims_2: np.ndarray = np.array([3, 3, 1])
g_cart_1: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims_1)
g_cart_2: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims_2)
g_cart_1.compute_geometry()
g_cart_2.compute_geometry()
# g_tetra: pp.TetrahedralGrid = pp.TetrahedralGrid(g_cart.nodes)
# g_tetra.compute_geometry()
mdg_1: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart_1]])
mdg_2: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart_2]])
box_1: dict = pp.bounding_box.from_points(np.array([[0, 0, 0], [3, 3, 1]]).T)
box_2: dict = pp.bounding_box.from_points(np.array([[0, 0, 0], [3, 3, 1]]).T)
subdomains_1 = [sd for sd in mdg_1.subdomains()]
subdomains_2 = [sd for sd in mdg_2.subdomains()]

# Define parameters and discretization
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


def total_source(g: pp.Grid) -> np.ndarray:
    array = np.zeros(g.num_cells)
    array[0] = 1
    return array


# Set variables and bc for all subdomains
logger.info("setting parameters")
for sd, data in mdg_1.subdomains(return_data=True):
    # One dof per cell for both variables
    data[pp.PRIMARY_VARIABLES] = {
        pressure_var: {"cells": 1},
    }
    # Boundary conditions and parameters
    diffusivity = pp.SecondOrderTensor(np.ones(sd.num_cells))
    pp.initialize_data(
        sd,
        data,
        flux_key,
        {
            "bc": bc_type(sd),
            "bc_values": bc_values(sd),
            "source": total_source(sd),
            "second_order_tensor": diffusivity,
        },
    )
    # Set initial condition.
    pp.set_state(data, {pressure_var: np.full(sd.num_cells, 0.3)})

logger.info("initializing ad objects")
dof_manager: pp.DofManager = pp.DofManager(mdg_1)
eq_manager: pp.ad.EquationManager = pp.ad.EquationManager(mdg_1, dof_manager)

# Merge variables. The merged variables can then be used in equations.
p: pp.ad.MergedVariable = eq_manager.merge_variables(
    [(sd, pressure_var) for sd in mdg_1.subdomains()]
)

# Define div and MPFA and upwind operator.
div = pp.ad.Divergence(mdg_1.subdomains())
flux_mpfa = pp.ad.TpfaAd(flux_key, mdg_1.subdomains())

# Source and bc.
total_source_ad = pp.ad.ParameterArray(flux_key, "source", subdomains_1)
p_bc = pp.ad.ParameterArray(flux_key, "bc_values", subdomains_1)

# Equation.
flux = flux_mpfa.flux * p + flux_mpfa.bound_flux * p_bc
pressure_eq = div * flux - total_source_ad

eqs = {"pressure_eq": pressure_eq}
eq_manager.equations.update(eqs)

# Discretize and solve.
logger.info(f"Solve")

logger.info(f"discretizing and assembling")
eq_manager.discretize(mdg_1)
A, b = eq_manager.assemble()
logger.info(f"\t solving")
solution = spla.spsolve(A, b)
print(f"A {A.todense()}")
print(f"b {b}")
print(f"sol {solution}")
