import pathlib
import typing
import warnings
from typing import Callable, Optional, override

import numpy as np
import porepy as pp
import tpf
from porepy.viz.exporter import DataInput
from tpf.models.flow_and_transport import EquationsTPF, SolutionStrategyTPF
from tpf.models.phase import FluidPhase
from tpf.spe10.fluid_values import BHP, INITIAL_PRESSURE, INITIAL_SATURATION, oil, water
from tpf.spe10.geometry import load_spe10_data
from tpf.utils.constants_and_typing import FEET, NONWETTING, PSI, WETTING


class EquationsSPE10(EquationsTPF):
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

    @typing.override
    def permeability(self, g: pp.Grid) -> dict[str, np.ndarray]:
        """Solid permeability. Chosen layer of the SPE10 model. Units are set by
        :attr:`self.solid`."""
        return {
            dim: self.solid.convert_units(perm, "m^2")
            for dim, perm in zip(["kxx", "kyy", "kzz"], self._permeability)
        }

    @typing.override
    def porosity(self, g: pp.Grid) -> np.ndarray:
        """Solid permeability. Chosen layer of the SPE10 model."""
        return self._porosity

    @typing.override
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
        cell_centers = g.cell_centers
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
    def corner_cell_ids(g: pp.Grid) -> list[np.intp]:
        """Identify the four corner cells of the grid.

        Parameters:
            g: Grid.

        Returns:
            corners: Indices of the corner cells.

        """
        cell_centers = g.cell_centers
        min_x, min_y = np.min(cell_centers, axis=1)
        max_x, max_y = np.max(cell_centers, axis=1)
        corners = [
            np.argmin(
                np.sum((cell_centers - np.array([[min_x], [min_y]])) ** 2, axis=0)
            ),
            np.argmin(
                np.sum((cell_centers - np.array([[min_x], [max_y]])) ** 2, axis=0)
            ),
            np.argmin(
                np.sum((cell_centers - np.array([[max_x], [min_y]])) ** 2, axis=0)
            ),
            np.argmin(
                np.sum((cell_centers - np.array([[max_x], [max_y]])) ** 2, axis=0)
            ),
        ]
        return corners

    def corner_masks(self, g: pp.Grid) -> tuple[pp.ad.DenseArray, pp.ad.DenseArray]:
        """Create masks that hide and single out the corner cells."""
        corner_cell_ids: list[np.intp] = self.corner_cell_ids(g)
        corner_mask_ndarray: np.ndarray = np.zeros((g.num_cells))
        corner_mask_ndarray[corner_cell_ids] = 1
        corner_mask = pp.ad.DenseArray(corner_mask_ndarray)
        corner_mask_inverse = pp.ad.DenseArray(1 - corner_mask_ndarray)
        corner_mask.set_name("Corner mask")
        corner_mask_inverse.set_name("Corner mask inverse")
        return corner_mask, corner_mask_inverse

    @typing.override
    def set_equations(self, equation_names: Optional[dict[str, str]] = None) -> None:
        """Modify the equations s.t. the corner cells get prescibed a pressure and
        saturation explicitly. This simulates production wells.

        """
        super().set_equations(equation_names)

        # Prescribe the corner cell values directly. This resembles Dirichlet
        # conditions.
        g: pp.Grid = self.mdg.subdomains()[0]
        flow_equation: pp.ad.Operator = self.equation_system.equations["Flow equation"]
        transport_equation: pp.ad.Operator = self.equation_system.equations[
            "Transport equation"
        ]

        corner_mask, corner_mask_inverse = self.corner_masks(g)

        # Subdivide new equations in 3 parts for easier debugging.
        old_flow_equation_masked: pp.ad.Operator = corner_mask_inverse * flow_equation
        old_transport_equation_masked: pp.ad.Operator = (
            corner_mask_inverse * transport_equation
        )
        old_flow_equation_masked.set_name("Old flow equation masked")
        old_transport_equation_masked.set_name("Old transport equation masked")

        explicit_pressure: pp.ad.Operator = corner_mask * (
            self.nonwetting.p - pp.ad.Scalar(BHP)
        )
        explicit_saturation: pp.ad.Operator = corner_mask * (
            self.wetting.s - pp.ad.Scalar(1 - self.wetting.residual_saturation)
        )
        explicit_pressure.set_name("Explicit pressure")
        explicit_saturation.set_name("Explicit saturation")

        # Attach the pressure values to the flow equation and the saturation values to
        # the transport equation.
        new_flow_equation: pp.ad.Operator = old_flow_equation_masked + explicit_pressure
        new_transport_equation: pp.ad.Operator = (
            old_transport_equation_masked + explicit_saturation
        )
        new_flow_equation.set_name("Flow equation")
        new_transport_equation.set_name("Transport equation")

        self.equation_system.equations["Flow equation"] = new_flow_equation
        self.equation_system.equations["Transport equation"] = new_transport_equation


class SolutionStrategySPE10(SolutionStrategyTPF):
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

    @typing.override
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


        """
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
            self._permeability: np.ndarray = np.zeros((g.dim, g.num_cells))
        self._porosity: np.ndarray = np.zeros((g.num_cells,))
        for i in range(g.num_cells):
            # FIXME Average over all SPE10 cells instead of using the center of the
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

    @typing.override
    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation.

        The corner cells get prescibed the right values immediately. Inside the
        reservoir, the initial pressure is higher. The initial saturation is set to the
        residual wetting saturation + 0.1 inside the reservoir.

        """
        g: pp.Grid = self.mdg.subdomains()[0]
        corner_cell_ids: list[np.intp] = self.corner_cell_ids(g)

        initial_pressure = np.full(g.num_cells, INITIAL_PRESSURE)
        initial_pressure[corner_cell_ids] = BHP
        initial_saturation = np.full(g.num_cells, INITIAL_SATURATION)
        initial_saturation[corner_cell_ids] = 1 - self.wetting.residual_saturation
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

    @typing.override
    def prepare_simulation(self) -> None:
        self.set_materials()
        self.set_geometry()
        # Initialize permeability and porosity now. Must be done after setting the
        # geometry but before setting equations.
        self.load_spe10_model(self.mdg.subdomains()[0])
        # Initialize data saving, s.t. ``self.exporter`` is instantiated and save
        # porosity and permeability.
        self.initialize_data_saving()
        self.add_constant_spe10_data()
        # Continue with the simulation preparation. This will run ``set_geometry`` and
        # ``set_materials`` again, which is not an issue.
        super().prepare_simulation()


class ModelGeometrySPE10(pp.ModelGeometry):

    @typing.override
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
                "xmax": 600 * FEET,
                "ymin": 0,
                "ymax": 1100 * FEET,
            }
        else:
            bounding_box = {
                "xmin": 0,
                "xmax": 1200 * FEET,
                "ymin": 0,
                "ymax": 2200 * FEET,
            }
        self._domain = pp.Domain(bounding_box)


class SPE10(EquationsSPE10, SolutionStrategySPE10, tpf.TwoPhaseFlow): ...
