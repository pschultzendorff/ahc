"""Implementation of the Buckley-Leverett assumptions in the two-phase flow model."""

import logging
import math
import os
from datetime import date
from typing import Any

import numpy as np
import porepy as pp

from src.tpf_lab.models.run_models import run_time_dependent_model
from src.tpf_lab.models.two_phase_flow import TwoPhaseFlow
from src.tpf_lab.utils import logging_redirect_tqdm, rm_out_padding

# Angle of the tube.
ANGLE: float = math.pi / 4
# Total influx at the left.
FLUX: float = 1.0
# Cap. pressure is ignored.
CAP_PRESSURE_MODEL = None

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BuckleyLeverett(TwoPhaseFlow):
    def create_grid(self) -> None:
        cell_dims: np.ndarray = np.array([self._grid_size])
        phys_dims: np.ndarray = np.array([self._phys_size])
        g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
        g_cart.compute_geometry()
        self.mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart]])
        self.domain = pp.Domain(
            bounding_box={
                "xmin": 0,
                "xmax": phys_dims[0],
                "ymin": 0,
                "ymax": 0,
            }
        )
        logger.debug("Grid created")

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting vector source. Corresponds to the wetting buoyancy flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the x dimension. This is scaled by the angle of the domain.
            vals[-1] = - pp.GRAVITY_ACCELERATION * self._w_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[0] = pp.GRAVITY_ACCELERATION * math.cos(ANGLE) * self._density_w
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric nonwetting vector source. Corresponds to the nonwetting buoyancy
        flow.

        Defined by gravity times :math:`cos(angle)` of the tube.

        To assign a gravity-like vector source, add a non-zero contribution in
        the x dimension. This is scaled by the angle of the domain.
            vals[-1] = - pp.GRAVITY_ACCELERATION * self._n_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[0] = pp.GRAVITY_ACCELERATION * math.cos(ANGLE) * self._density_n
        return vals.ravel()

    def _bc_type_pressure_w(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Wetting pressure boundary conditions.

        Neumann conditions on the left, Dirichlet conditions on the right.

        """
        east = self._domain_boundary_sides(g).east
        return pp.BoundaryCondition(g, east, "dir")

    def _bc_type_pressure_n(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Nonwetting pressure boundary conditions.

        Dirichlet bc on both sides.

        """
        all_bf = self._domain_boundary_sides(g).all_bf
        return pp.BoundaryCondition(g, all_bf, "dir")

    def _bc_type_pressure_c(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Capillary pressure boundary conditions.

        Neumann bc on the left. Dirichlet conditions on the right.

        """
        east = self._domain_boundary_sides(g).east
        return pp.BoundaryCondition(g, east, "dir")

    def _bc_values_pressure_w(self, g: pp.Grid) -> np.ndarray:
        """Injection at the left."""
        array = np.zeros(g.num_faces)
        # For some reason the east boundary has index 0 and the west boundary has index
        # -1.
        array[0] = 0
        array[-1] = FLUX
        return array

    def _bc_values_pressure_n(self, g: pp.Grid) -> np.ndarray:
        """Constant pressure on both sides."""
        array = np.zeros(g.num_faces)
        return array

    def _initial_condition(self) -> None:
        super()._initial_condition()
        initial_saturation = np.full(self._grid_size, 1 - self._residual_saturation_n)
        initial_saturation[: int(self._grid_size / 2)] = self._residual_saturation_w
        self.equation_system.set_variable_values(
            initial_saturation,
            [self._ad.saturation],
            time_step_index=self.time_manager.time_index,
        )

    def _mobility_w(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        mobility_w = super()._mobility_w(subdomains)
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, subdomains)
        return mobility_w + upwind_w.upwind @ self._error_function_deriv()


i = 1


yscales = np.arange(0, 0.6, 0.1)
densities = np.arange(1.0, 10.0, 2.0)

yscales = [0.0]
densities = [1.0]

# We set the model up with the same values as for the Buckley-Leverett analytical
# solution.

for yscale in yscales:
    for density in densities:
        params: dict[str, Any] = {
            "formulation": "n_pressure_w_saturation",
            "file_name": f"yscale_{yscale}_density_w_{density}",
            "folder_name": os.path.join(
                "results",
                "buckley_leverett",
            ),
        }
    model = BuckleyLeverett(params)

    model._grid_size = 1000
    model._phys_size = 10
    model._time_step = 0.001
    model._schedule = np.array([0, 0.1])

    model._density_ = density

    model._rel_perm_model = "Corey"
    model._rel_perm_linear_param = 1.0

    model._cap_pressure_model = CAP_PRESSURE_MODEL

    model._yscale = yscale
    model._xscale = 1000

    model.prepare_simulation()

    with logging_redirect_tqdm([logger]):
        run_time_dependent_model(
            model, {"nl_convergence_tol": 1e-10, "max_iterations": 100}
        )
