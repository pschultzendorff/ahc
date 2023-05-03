"""Implementation of the Buckley-Leverett assumptions in the two-phase flow model."""

import logging
import math
import os
from datetime import date
from typing import Any

import numpy as np
import porepy as pp

logger = logging.Logger("single_phase_flow")


class SinglePhaseFlow1D:
    def set_domain(self) -> None:
        self._domain = pp.Domain(
            bounding_box={
                "xmin": 0,
                "xmax": 20,
                "ymin": 0,
                "ymax": 0,
            }
        )

    def meshing_arguments(self) -> dict:
        mesh_args: dict[str, float] = {"cell_size": 0.25 / self.units.m}
        return mesh_args

    def bc_type_darcy(self, g: pp.Grid) -> pp.BoundaryCondition:
        east = self._domain_boundary_sides(g).east
        return pp.BoundaryCondition(g, east, "dir")

    def bc_values_darcy(self, g: pp.Grid) -> np.ndarray:
        """Injection at the left."""
        array = np.zeros(g.num_faces)
        # The west boundary has index -1 and the east boundary has index 0.
        array[-1] = 1.0
        return array


class SinglePhaseFlowGeometry(SinglePhaseFlow1D, pp.fluid_mass_balance.SinglePhaseFlow):
    ...


params = {}
model = SinglePhaseFlowGeometry(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8))
