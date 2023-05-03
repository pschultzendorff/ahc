"""Implementation of the Buckley-Leverett model in the two-phase flow model."""

import logging
import math

import numpy as np
import porepy as pp

from tpf_lab.models.two_phase_flow import TwoPhaseFlow


# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class BuckleyLeverett(TwoPhaseFlow):
    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        # Angle of the tube.
        if params is None:
            params = {}
        self._angle: float = params.get("angle", math.pi / 4)
        # For PorePy reasons we want the flux to be negative (this equals inflow).
        self._influx: float = params.get("influx", -1.0)
        # Cap pressure is ignored.
        self._cap_pressure_model = None

    def create_grid(self) -> None:
        # 1d grid
        cell_dims: np.ndarray = np.array([self._grid_size])
        phys_dims: np.ndarray = np.array([self._phys_size])
        g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
        g_cart.compute_geometry()
        self.mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_cart]])
        self.domain = pp.Domain(
            bounding_box={
                "xmin": -10,
                "xmax": phys_dims[0] - 10,
                "ymin": 0,
                "ymax": 0,
            }
        )
        logger.debug("Grid created")

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting vector source. Corresponds to the wetting buoyancy flow.

        To assign a gravity-like vector source, add a non-zero contribution in
        the x dimension. This is scaled by the angle of the domain.
            vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) *
            self._density_w
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) * self._density_w
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Volumetric nonwetting vector source. Corresponds to the nonwetting buoyancy
        flow.

        Defined by gravity times :math:`cos(angle)` of the tube.

        To assign a gravity-like vector source, add a non-zero contribution in
        the x dimension. This is scaled by the angle of the domain.
            vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) *
            self._density_n
        """
        vals = np.zeros((g.num_cells, self.mdg.dim_max()))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * math.cos(self._angle) * self._density_n
        return vals.ravel()

    def _bc_type_pressure_w(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Wetting pressure boundary conditions.

        Neumann conditions on the left, Dirichlet conditions on the right.

        """
        lhs = np.array([0])
        return pp.BoundaryCondition(g, lhs, "dir")

    def _bc_type_pressure_n(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Nonwetting pressure boundary conditions.

        Neumann conditions on the left (index -1), Dirichlet conditions on the right
        (index 0).

        """
        lhs = np.array([0])
        return pp.BoundaryCondition(g, lhs, "dir")

    def _bc_type_pressure_c(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Capillary pressure boundary conditions.

        Neumann bc on both sides.

        """
        return pp.BoundaryCondition(g)

    def _dirichlet_bc_values_pressure_n(self, g: pp.Grid) -> np.ndarray:
        """Zero pressure at the right (index 0)."""
        array = np.zeros(g.num_faces)
        return array

    def _dirichlet_bc_values_pressure_w(self, g: pp.Grid) -> np.ndarray:
        """Wetting pressure equals the nonwetting pressure."""
        array = np.zeros(g.num_faces)
        return array

    def _neumann_bc_values_flux_n(self, g: pp.Grid) -> np.ndarray:
        """Injection at the left (index -1). Note that this equals the total flux."""
        array = np.zeros(g.num_faces)
        array[-1] = self._influx
        return array

    def _bc_values_mobility_w(self, g: pp.Grid) -> np.ndarray:
        array = super()._bc_values_mobility_w(g)
        # Set the wetting mobility at the boundaries to the wetting mobility at residual
        # saturations.
        # For PorePy reasons the values needs to be negative.
        array[-1] = -((1 - 0.5) ** 3) * self._rel_perm_linear_param
        array[0] = 0.01
        return array

    def _bc_values_mobility_n(self, g: pp.Grid) -> np.ndarray:
        array = super()._bc_values_mobility_n(g)
        # Set the nonwetting mobility at the boundaries to the nonwetting mobility at
        # residual saturations.
        # For PorePy reasons the values needs to be negative.
        array[-1] = 0.01
        array[0] = -((1 - 0.5) ** 3) * self._rel_perm_linear_param
        return array

    def _initial_condition(self) -> None:
        """Residual nonwetting saturation in the left side of the domain. Residual
        wetting saturation in the right side of the domain. A transition zone in the
        middle."""
        super()._initial_condition()
        initial_saturation = np.full(self._grid_size, 1 - self._residual_saturation_n)
        initial_saturation[: int(self._grid_size / 2)] = self._residual_saturation_w
        initial_saturation[
            int(self._grid_size / 2) - 10 : int(self._grid_size / 2) + 10
        ] = np.linspace(
            self._residual_saturation_w, 1 - self._residual_saturation_n, 20
        )

        self.equation_system_full.set_variable_values(
            initial_saturation,
            [self._ad.saturation],
            time_step_index=self.time_manager.time_index,
        )

    def _export(self):
        """Export only each 50th time step."""
        if self.time_manager.time_index % 50 == 0:
            super()._export()
