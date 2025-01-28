import pathlib
import typing
import warnings
from typing import Callable

import numpy as np
import porepy as pp
import tpf
from porepy.viz.exporter import DataInput
from tpf.models.phase import FluidPhase
from tpf.models.protocol import TPFProtocol
from tpf.spe10.fluid_values import (
    BHP,
    INITIAL_PRESSURE,
    INITIAL_SATURATION,
    PRODUCTION_WELL_SIZE,
    oil,
    water,
)
from tpf.spe10.geometry import X_LENGTH, Y_LENGTH, load_spe10_data
from tpf.utils.constants_and_typing import FEET, NONWETTING, WETTING


class EquationsSPE10(TPFProtocol):
    """Mixin class to provide the SPE10 model equations and data.

    Takes care of:
    Updates the two-phase flow equations to include:
    - the SPE10 porosity field.
    - the SPE10 permeability field.
    - A volumetric source term for the water phase in the center cell.
    - Production wells in the corner cells.

    """

    _permeability: np.ndarray
    """Provided by :class:`SolutionStrategySPE10`."""
    _porosity: np.ndarray
    """Provided by :class:`SolutionStrategySPE10`."""

    def permeability(self, g: pp.Grid) -> dict[str, np.ndarray]:
        """Solid permeability. Chosen layer of the SPE10 model. Units are set by
        :attr:`self.solid`."""
        return {
            dim: self.solid.convert_units(perm, "m^2")
            for dim, perm in zip(["kxx", "kyy", "kzz"], self._permeability)
        }

    def porosity(self, g: pp.Grid) -> np.ndarray:
        """Solid porosity. Chosen layer of the SPE10 model.

        The layers may have zero porosity, which results in the linear solver failing to
        solve the transport equation. To avoid this, we add a small epsilon.

        """
        return np.maximum(self._porosity, np.full_like(self._porosity, 1e-5))

    def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:  # type: ignore
        r"""Volumetric phase source term. Given as volumetric flux.

        Five-spot setup. Water (wetting) injection in the center, oil (nonwetting)
        production in the four corners.

        NOTE: This is the average value per grid cell, i.e., it gets scaled with the
        cell volume in the equation.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        if phase.name == self.wetting.name:
            array: np.ndarray = super().phase_fluid_source(g, phase)
            array[self.center_cell_id(g)] = phase.convert_units(
                87.5, "m^3"
            ) / phase.convert_units(
                pp.DAY, "s"
            )  # 87.5 m^3/day in [m^3/s]
            return array
        elif phase.name == self.nonwetting.name:
            return super().phase_fluid_source(g, phase)

    @staticmethod
    def center_cell_id(g: pp.Grid) -> np.intp:
        """Identify the center cell of the grid.

        Parameters:
            g: Grid.

        Returns:
            corner: Index of the center cell.

        """
        # Ignore z-values of the grid.
        cell_centers = g.cell_centers[:2, :]
        min_x, min_y = np.min(cell_centers, axis=1)
        max_x, max_y = np.max(cell_centers, axis=1)
        center = np.argmin(
            np.sum(
                (
                    cell_centers
                    - np.array([[(max_x - min_x) / 2], [(max_y - min_y) / 2]])
                )
                ** 2,
                axis=0,
            )
        )
        return center

    @staticmethod
    def corner_faces_id(g: pp.Grid, height: float, width: float) -> np.ndarray:
        """Identify the boundary faces in the corners of the grid corresponding to
        production wells.

        Parameters:
            g: Grid.

        Returns:
            corners: Indices of the boundary faces in the corners.

        """
        # Ignore z-values of the grid.
        boundary_faces: np.ndarray = g.get_boundary_faces()
        boundary_face_centers: np.ndarray = g.face_centers[:2, boundary_faces]
        # Find indices of faces in the corners.
        indices: list[np.ndarray] = []
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] == 0,
                    boundary_face_centers[1] <= PRODUCTION_WELL_SIZE,
                ),
            )
        )
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] == 0,
                    boundary_face_centers[1] >= height - PRODUCTION_WELL_SIZE,
                )
            )
        )
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] == width,
                    boundary_face_centers[1] <= PRODUCTION_WELL_SIZE,
                )
            )
        )
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] == width,
                    boundary_face_centers[1] >= height - PRODUCTION_WELL_SIZE,
                )
            )
        )
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] <= PRODUCTION_WELL_SIZE,
                    boundary_face_centers[1] == 0,
                )
            )
        )
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] >= width - PRODUCTION_WELL_SIZE,
                    boundary_face_centers[1] == 0,
                )
            )
        )
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] <= PRODUCTION_WELL_SIZE,
                    boundary_face_centers[1] == height,
                )
            )
        )
        indices.append(
            np.argwhere(
                np.logical_and(
                    boundary_face_centers[0] >= width - PRODUCTION_WELL_SIZE,
                    boundary_face_centers[1] == height,
                )
            )
        )
        return boundary_faces[np.concatenate(indices).flatten()]


class ModifiedBoundarySPE10(TPFProtocol):

    corner_faces_id: Callable[[pp.Grid, float, float], np.ndarray]

    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann).

        We assign Neumann conditions for all faces. The four corner cells get prescribed
        a pressure explicitely, which acts as a Dirichlet condition.

        """
        height: float = (
            (Y_LENGTH / 2) if self.params["spe10_quarter_domain"] else Y_LENGTH
        )
        width: float = X_LENGTH / 2 if self.params["spe10_quarter_domain"] else X_LENGTH
        corner_faces_id: np.ndarray = self.corner_faces_id(g, height, width)
        return pp.BoundaryCondition(g, corner_faces_id, "dir")

    def _bc_dirichlet_pressure_values(
        self, g: pp.Grid, phase: FluidPhase
    ) -> np.ndarray:
        """Dirichle pressure values.

        We assign Neumann conditions for all faces. The boundaries in all corners get
        prescribed pressure explicitely and act as wells.

        """
        if phase == self.nonwetting:
            height: float = (
                Y_LENGTH / 2 if self.params["spe10_quarter_domain"] else Y_LENGTH
            )
            width: float = (
                X_LENGTH / 2 if self.params["spe10_quarter_domain"] else X_LENGTH
            )
            corner_faces_id: np.ndarray = self.corner_faces_id(g, height, width)
            bc: np.ndarray = np.zeros(g.num_faces)
            bc[corner_faces_id] = BHP
            return bc
        else:
            raise NotImplementedError(
                "Dirichlet pressure values not implemented for the wetting phase."
            )

    def _bc_dirichlet_saturation_values(
        self, g: pp.Grid, phase: FluidPhase
    ) -> np.ndarray:
        if phase.name == self.wetting.name:
            s_bc: np.ndarray = np.full(g.num_faces, INITIAL_SATURATION)
        elif phase.name == self.nonwetting.name:
            s_bc = np.ones(g.num_faces) - self._bc_dirichlet_saturation_values(
                g, self.wetting
            )
        return s_bc


class SolutionStrategySPE10(TPFProtocol):
    """Mixin class to provide the SPE10 model data.

    Takes care of:
    - Loading the SPE10 fluids, i.e., oil and water.
    - Loading the SPE10 geometry, i.e., permeability and porosity.
    - Exporting the SPE10 geometry.

    Requires the following model parameters to be set:
        - "spe10_layer" (int): The layer of the SPE10 model to use.
        - "spe10_isotropic_perm" (bool): Whether to use isotropic permeability.

    """

    corner_cell_ids: Callable[[pp.Grid], list[np.intp]]

    def set_phases(self) -> None:
        self.phases: dict[str, FluidPhase] = {}
        for phase_name, constants in zip([WETTING, NONWETTING], [water, oil]):
            phase = FluidPhase(constants)
            phase.set_units(self.units)
            setattr(self, phase_name, phase)
            self.phases[phase_name] = phase

    def load_spe10_model(self, g: pp.Grid) -> None:
        """Load porosity and permeability of the SPE10 layer specified in the model
        parameters.

        Note:
        - If the model domain is within the horizontal extend of an SPE10 layer, ...
        - Requires "self.params["spe10_layer"]" and "self.params["isotropic_perm"]"
        to be set.

        Parameters:
            g (pp.Grid): Grid to load the data for.

        Raises:
            ValueError: If the cell size is larger than the SPE10 model cell size.

        """
        cell_size = self.params["meshing_arguments"]["cell_size"]
        assert isinstance(cell_size, float)
        if cell_size > 20 * FEET:
            raise ValueError(
                "The cell size is larger than the SPE10 model cell size. "
                + "This is not supported yet."
            )
        layer: int = self.params.get("spe10_layer", 1)
        isotropic_perm: bool = self.params.get("spe10_isotropic_perm", True)
        for param_name, value in zip(
            ["spe10_layer", "spe10_isotropic_perm"], [layer, isotropic_perm]
        ):
            if param_name not in self.params:
                warnings.warn(
                    f"The model parameter '{param_name}' is not set."
                    + f" Continuing with default value {value}"
                )

        perm, poro = load_spe10_data(pathlib.Path(__file__).parent / "data")
        if isotropic_perm:
            self._permeability: np.ndarray = np.zeros((1, g.num_cells))
        else:
            self._permeability = np.zeros((g.dim, g.num_cells))
        self._porosity: np.ndarray = np.zeros((g.num_cells,))
        for i in range(g.num_cells):
            # TODO Average over all SPE10 cells instead of using the center of the
            # cell. This fix applies only for coarse resolutions.
            coors: np.ndarray = g.cell_centers[:, i]
            # One cell in the original SPE10 model is 20 ft x 10 ft x 2 ft.
            x_ind: int = int(coors[0] // (20 * FEET))
            y_ind: int = int(coors[1] // (10 * FEET))
            for j in range(self._permeability.shape[0]):
                self._permeability[j, i] = perm[j, layer, y_ind, x_ind]
            self._porosity[i] = poro[layer, y_ind, x_ind]

    def add_constant_spe10_data(self) -> None:
        """Save the SPE10 data to the exporter."""
        data: list[DataInput] = []
        g: pp.Grid = self.mdg.subdomains()[0]
        for dim, perm in zip(["kxx", "kyy", "kzz"], self._permeability):
            data.append((g, "permeability_" + dim, perm))
        data.append((g, "porosity", self.porosity(g)))
        self.exporter.add_constant_data(data)

        # For convenience, add the porosity and permeability to the iteration exporter
        # if it exists.
        if hasattr(self, "iteration_exporter"):
            self.iteration_exporter.add_constant_data(data)

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation.

        The corner cells get prescibed the right values immediately. Inside the
        reservoir, the initial pressure is higher. The initial saturation is set to the
        residual wetting saturation + 0.1 inside the reservoir.

        """
        g: pp.Grid = self.mdg.subdomains()[0]
        # corner_cell_ids: list[np.intp] = self.corner_cell_ids(g)

        initial_pressure = np.full(g.num_cells, INITIAL_PRESSURE)
        # initial_pressure[corner_cell_ids] = BHP
        initial_saturation = np.full(g.num_cells, INITIAL_SATURATION)
        # initial_saturation[corner_cell_ids] = 1 - self.wetting.residual_saturation
        self.equation_system.set_variable_values(
            np.concatenate([initial_pressure, initial_pressure]),
            [self.wetting.p, self.nonwetting.p],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )
        self.equation_system.set_variable_values(
            np.concatenate([initial_saturation, 1 - initial_saturation]),
            [self.wetting.s, self.nonwetting.s],
            time_step_index=0,
            hc_index=0,
            iterate_index=0,
        )

    def prepare_simulation(self) -> None:
        self.set_materials()
        self.set_geometry()
        # Initialize permeability and porosity now. Must be done after setting the
        # geometry but before setting equations.
        self.load_spe10_model(self.mdg.subdomains()[0])
        # Continue with the simulation preparation. This will run ``set_geometry`` and
        # ``set_materials`` again, which is not an issue.
        super().prepare_simulation()
        # Save porosity and permeability only after the exporter is initialized. Else,
        # they would be overwritten.
        self.add_constant_spe10_data()


class ModelGeometrySPE10(TPFProtocol):

    def set_domain(self) -> None:
        r"""Single layer of the SPE10 problem 2 model. Extend of the full domain is
        :math:`\qty{1200 x 2200 x 170}{\feet}`. A single layer is
        :math:`\qty{1200 x 2200}{\feet}`.

        """
        quarter_domain: bool = self.params.get("spe10_quarter_domain", False)
        if "spe10_quarter_domain" not in self.params:
            warnings.warn(
                "The model parameter 'spe10_quarter_domain' is not set."
                + f" Continuing with default value {quarter_domain}"
            )
        if quarter_domain:
            bounding_box: dict[str, pp.number] = {
                "xmin": 0,
                "xmax": X_LENGTH / 2,
                "ymin": 0,
                "ymax": Y_LENGTH / 2,
            }
        else:
            bounding_box = {
                "xmin": 0,
                "xmax": X_LENGTH,
                "ymin": 0,
                "ymax": Y_LENGTH,
            }
        self._domain = pp.Domain(bounding_box)

    def set_fractures(self) -> None:
        # Use fractures as constraints to ensure that the grid is conforming to the well
        # boundaries.
        self._fractures = [
            pp.LineFracture(
                np.array([[0, X_LENGTH], [PRODUCTION_WELL_SIZE, PRODUCTION_WELL_SIZE]])
            ),
            pp.LineFracture(
                np.array(
                    [
                        [0, X_LENGTH],
                        [
                            Y_LENGTH - PRODUCTION_WELL_SIZE,
                            Y_LENGTH - PRODUCTION_WELL_SIZE,
                        ],
                    ]
                )
            ),
            pp.LineFracture(
                np.array([[PRODUCTION_WELL_SIZE, PRODUCTION_WELL_SIZE], [0, Y_LENGTH]])
            ),
            pp.LineFracture(
                np.array(
                    [
                        [
                            X_LENGTH - PRODUCTION_WELL_SIZE,
                            X_LENGTH - PRODUCTION_WELL_SIZE,
                        ],
                        [0, Y_LENGTH],
                    ]
                )
            ),
        ]

    def meshing_kwargs(self) -> dict:
        """Keyword arguments for md-grid creation.

        Returns:
            Keyword arguments compatible with pp.create_mdg() method.

        """
        meshing_kwargs = self.params.get("meshing_kwargs", None)
        if meshing_kwargs is None:
            meshing_kwargs = {}
        meshing_kwargs.update(
            {"constraints": np.array(list(range(len(self.fractures))))}
        )
        return meshing_kwargs


# The various protocols define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SPE10Mixin(EquationsSPE10, ModifiedBoundarySPE10, SolutionStrategySPE10, ModelGeometrySPE10): ...  # type: ignore
