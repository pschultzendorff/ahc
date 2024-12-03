import pathlib
from typing import Callable, Optional

import numpy as np
import porepy as pp
from porepy.viz.exporter import DataInput
from tpf_lab.constants_and_typing import NONWETTING, WETTING
from tpf_lab.models.phase import FluidPhase
from tpf_lab.spe10.fluid_values import oil, water
from tpf_lab.spe10.geometry import load_spe10_data


class SPE10Mixin:
    """Mixin class to provide the SPE10 model data.

    Takes care of:
    - Loading the SPE10 geometry, i.e., permeability and porosity.
    - Loading the SPE10 fluids, i.e., oil and water.
    - Exporting the SPE10 geometry.

    Requires the following parameters to be set:
        - "spe10_layer" (int): The layer of the SPE10 model to use.
        - "spe10_isotropic_perm" (bool): Whether to use isotropic permeability.

    """

    params: dict[str, int]
    """Normally provided by a mixin of instance :class:`~porepy.SolutionStrategy`."""
    units: pp.Units
    """Normally provided by a mixin of instance :class:`~porepy.SolutionStrategy`."""
    set_materials: Callable[[], None]
    """Normally provided by a mixin of instance :class:`~porepy.SolutionStrategy`."""

    mdg: pp.MixedDimensionalGrid
    """Normally provided by a mixin of instance :class:`~porepy.ModelGeometry`."""
    set_geometry: Callable[[], None]
    """Normally provided by a mixin of instance :class:`~porepy.ModelGeometry`."""

    def permeability(self, g: pp.Grid) -> dict[str, np.ndarray]:
        """Solid permeability. Chosen layer of the SPE10 model. Units are set by
        :attr:`self.solid`."""
        # TODO Include unit!
        return {
            dim: perm for dim, perm in zip(["kxx", "kyy", "kzz"], self._permeability)
        }

    def porosity(self, g: pp.Grid) -> np.ndarray:
        """Solid permeability. Chosen layer of the SPE10 model."""
        return self._porosity

    def load_spe10_model(self, g: pp.Grid) -> None:
        """Load porosity and permeability of the SPE10 layer specified in the model
        parameters.

        If the model domain is within the horizontal extend of an SPE10 layer,
        Requires "self.params["spe10_layer"]" and "self.params["isotropic_perm"]" to be set.

        Parameters:
            g (pp.Grid): Grid to load the data for.

        Raises:
            ValueError: If the parameters "spe10_layer" or "isotropic_perm" are not
            set.

        """
        if (
            "spe10_layer" not in self.params
            or "spe10_isotropic_perm" not in self.params
        ):
            raise ValueError(
                "The parameters 'spe10_layer' and `spe10_isotropic_perm` must be set to"
                + " load the SPE10 model."
            )
        perm, poro = load_spe10_data(pathlib.Path(__file__).parent / "data")
        if self.params["spe10_isotropic_perm"]:
            self._permeability: np.ndarray = np.zeros((1, g.num_cells))
        else:
            self._permeability: np.ndarray = np.zeros((g.dim, g.num_cells))
        self._porosity: np.ndarray = np.zeros((g.num_cells,))
        for i in range(g.num_cells):
            # FIXME Average over all SPE10 cells instead of using the center of the
            # cell.
            coors: np.ndarray = g.cell_centers[:, i]
            # One cell in the original SPE10 model is 10 ft x 20 ft x 2 ft, i.e.,
            # 3.048 m x 6.096 m x 0.6096 m.
            x_ind: int = int(coors[0] // 3.048)
            y_ind: int = int(coors[1] // 6.096)
            for j in range(self._permeability.shape[0]):
                self._permeability[j, i] = perm[
                    j, self.params["spe10_layer"], y_ind, x_ind
                ]
            self._porosity[i] = poro[0, y_ind, x_ind]

    def set_phases(self) -> None:
        self.phases: dict[str, FluidPhase] = {}
        for phase_name, constants in zip([WETTING, NONWETTING], [water, oil]):
            phase = FluidPhase(constants)
            phase.set_units(self.units)
            setattr(self, phase_name, phase)
            self.phases[phase_name] = phase

    def prepare_simulation(self) -> None:
        self.set_materials()
        self.set_geometry()
        # Initialize permeability and porosity now. Must be done after setting the
        # geometry but before setting equations.
        self.load_spe10_model(self.mdg.subdomains()[0])
        # Continue with the simulation preparation. This will run ``set_geometry`` and
        # ``set_materials`` again, which is not an issue.
        super().prepare_simulation()

    # TODO Mark this data as constant
    def _data_to_export(
        self, time_step_index: Optional[int] = None, iterate_index: Optional[int] = None
    ) -> list[DataInput]:
        """Append porosity and permeability to the exported data."""
        # Get data from ``super()`` and append porosity and permeability.
        data: list[DataInput] = super()._data_to_export(
            time_step_index=time_step_index, iterate_index=iterate_index
        )
        g: pp.Grid = self.mdg.subdomains()[0]
        for dim, perm in zip(["kxx", "kyy", "kzz"], self._permeability):
            data.append((g, "permeability_" + dim, perm))
        data.append((g, "porosity", self.porosity(g)))
        return data

    # TODO Add initial conditions and phase sources. Code something to automatically
    # find the corner cells.
