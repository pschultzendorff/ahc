"""Implementation of the Buckley-Leverett assumptions in the two-phase flow model."""

import logging
import math
import os
from datetime import date

import numpy as np
import porepy as pp

from src.tpflab.models.run_models import run_time_dependent_model
from src.tpflab.models.two_phase_flow import TwoPhaseFlow
from src.tpflab.utils import logging_redirect_tqdm, rm_out_padding

# Angle of the tube.
ANGLE: float = 0.0
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
                "xmin": -10,
                "xmax": -10 + phys_dims[0],
                "ymin": 0,
                "ymax": 0,
            }
        )
        logger.debug("Grid created")

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting vector source. Corresponds to the wetting buoyancy flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * self._w_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[-1] = pp.GRAVITY_ACCELERATION * math.cos(ANGLE) * self._density_w._value
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric nonwetting vector source. Corresponds to the nonwetting buoyancy
        flow.

        Defined by gravity times :math:`cos(angle)` of the tube.

        To assign a gravity-like vector source, add a non-zero contribution in
        the last dimension:
            vals[-1] = - pp.GRAVITY_ACCELERATION * self._n_density
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[-1] = pp.GRAVITY_ACCELERATION * math.cos(ANGLE) * self._density_n._value
        return vals.ravel()

    def _bc_type_pressure_w(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Wetting pressure boundary conditions.

        Neumann conditions on the left, Dirichlet conditions on the right.

        """
        east = self._domain_boundary_sides(g).east
        return pp.BoundaryCondition(g, east, "dir")

    def _bc_type_pressure_n(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Nonwetting pressure boundary conditions.

        Neumann conditions on the left, Dirichlet conditions on the right.

        """
        east = self._domain_boundary_sides(g).east
        return pp.BoundaryCondition(g, east, "dir")

    def _bc_values_pressure(self, g: pp.Grid) -> pp.ad.DenseArray:
        """Injection at the left."""
        array = np.zeros(g.num_faces)
        # For some reason the east boundary has index 0 and the west boundary has index
        # -1.
        array[0] = 0
        array[-1] = FLUX
        return pp.ad.DenseArray(array)

    def _initial_condition(self) -> None:
        super()._initial_condition()
        initial_saturation = np.full(
            self._grid_size, 1 - self._residual_saturation_n._value
        )
        initial_saturation[
            : int(self._grid_size / 2)
        ] = self._residual_saturation_w._value
        self.equation_system.set_variable_values(
            initial_saturation,
            [self._ad.saturation],
            time_step_index=self.time_manager.time_index,
        )


i = 1

params = {
    "formulation": "n_pressure_w_saturation",
    "file_name": f"{date.today().strftime('%Y-%m-%d')}_run_{i}",
    "folder_name": os.path.join(
        "results",
        "buckley_leverett",
    ),
}

model = BuckleyLeverett(params)

model._grid_size = 1000
model._phys_size = 10
model._cap_pressure_model = CAP_PRESSURE_MODEL
model._time_step = 0.001
model._schedule = np.array([0, 1.0])
model.prepare_simulation()


with logging_redirect_tqdm([logger]):
    run_time_dependent_model(
        model, {"nl_convergence_tol": 1e-10, "max_iterations": 100}
    )
