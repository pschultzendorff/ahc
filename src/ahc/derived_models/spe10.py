"""This module defines a model for the 10th SPE Comparative Solution Project (SPE10),
case 2A.

[M. A. Christie and M. J. Blunt, “Tenth SPE Comparative Solution Project: A
Comparison of Upscaling Techniques,” SPE Reservoir Evaluation & Engineering, vol. 4, no.
04, pp. 308–317, Aug. 2001, doi: 10.2118/72469-PA.]

https://www.spe.org/web/csp/datasets/set02.htm

Additionally, the module provides util functions to download and prepare geometric data
for the model.

"""

import logging
import pathlib
import warnings
import zipfile
from typing import Any, cast

import numpy as np
import porepy as pp
import requests
from porepy.viz.exporter import DataInput

from ahc.derived_models.fluid_values import oil as _oil
from ahc.derived_models.fluid_values import water as _water
from ahc.derived_models.utils import center_cell_id, corner_faces_id
from ahc.models.phase import FluidPhase
from ahc.models.protocol import TPFProtocol
from ahc.utils.constants_and_typing import FEET, NONWETTING, PSI, WETTING

logger = logging.getLogger(__name__)

DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve() / "spe10_data"
ZIP_FILENAME: str = "por_perm_case2a.zip"
URL: str = "https://www.spe.org/web/csp/datasets/por_perm_case2a.zip"

# region MODEL_PARAMETERS
WIDTH: float = 1200 * FEET
HEIGHT: float = 2200 * FEET

INITIAL_SATURATION: float = 0.3  # [-], initial saturation.

water: dict[str, Any] = _water.copy()
water.update(
    {
        "residual_saturation": 0.2,  # [-], residual saturation.
    }
)
oil: dict[str, Any] = _oil.copy()
oil.update(
    {
        "residual_saturation": 0.2,  # [-], residual saturation.
    }
)

BHP: float = 4000 * PSI  # [psi], bottom hole pressure.
PRODUCTION_WELL_SIZE: float = 100 * FEET
"""Size of the production wells in the corner cells. The boundary in in vicinity of the
corners is prescribed Dirichlet conditions corresponding to a production well, i.e.,
fixed BHP."""
INJECTION_RATE: float = 87.5  # [m^3/day], constant water injection rate.

INITIAL_PRESSURE: float = BHP
# [psi], initial pressure; not really relevant except for Newton's initial guess.


# endregion


def download_spe10_data(
    data_dir: pathlib.Path, zip_filepath: pathlib.Path = DATA_DIR / ZIP_FILENAME
) -> None:
    """Download the SPE10 porosity and permeability data."""
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the ZIP file containing the SPE10 dataset.
    logger.info(f"Downloading dataset from {URL}")
    response = requests.get(URL)
    response.raise_for_status()

    with zip_filepath.open("wb") as f:
        f.write(response.content)
    logger.info("Download completed.")

    # Extract the ZIP file.
    extracted_files: list[str] = []
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(data_dir)
        extracted_files = zip_ref.namelist()
    logger.info(f"Extracted files: {extracted_files}")

    # Locate the .dat files for permeability and porosity.
    perm_file: pathlib.Path | None = None
    poro_file: pathlib.Path | None = None
    for filename in extracted_files:
        if "perm" in filename.lower():
            perm_file = data_dir / filename
        elif "phi" in filename.lower():
            poro_file = data_dir / filename

    if perm_file is None or poro_file is None:
        raise FileNotFoundError(
            "Could not locate permeability or porosity data files in the downloaded"
            + " contents."
        )

    # Clean up.
    zip_filepath.unlink()
    logger.info("Downloaded files cleaned up.")


def load_spe10_data(data_dir: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the SPE10 data into a :class:`~numpy.ndarray`.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Permeability data array.
            - np.ndarray: Porosity data array.

    """
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    perm_file: pathlib.Path | None = None
    poro_file: pathlib.Path | None = None

    # In lieu of a goto statement, run the following loop at maximum twice.
    i: int = 0
    while True:
        for filename in data_dir.iterdir():
            if "perm" in str(filename).lower():
                perm_file = data_dir / filename
            elif "phi" in str(filename).lower():
                poro_file = data_dir / filename
        if perm_file is None or poro_file is None:
            if i >= 1:
                raise FileNotFoundError(
                    "Permeability and porosity data files not found. Perhaps, they were"
                    + " not downloaded correctly."
                )
            logger.info(
                "Permeability and porosity data files not found. Downloading..."
            )
            download_spe10_data(data_dir)
        else:
            break
        i += 1

    logger.info("Loading permeability and porosity data.")
    perm_data = np.loadtxt(str(perm_file)).reshape(3, 85, 220, 60)  # unit: [mD]

    # Convert permeability which comes in mD to m^2.
    perm_data *= pp.MILLIDARCY
    poro_data = np.loadtxt(str(poro_file)).reshape(85, 220, 60)  # unit: [-]

    return perm_data, poro_data


# NOTE All SPE10 classes are purely mixins, i.e., the only superclasses are protocols.
# This ensures that the SPE10 model can be added on top of both adaptive homotopy
# continuation and adaptive Newton.
# For example, super().prepare_simulation calls either
# SolutionStrategyHC.prepare_simulation or EstimatesSolutionStrategy.prepare_simulation.
# If SPE10SolutionStrategyMixin would subclass EstimatesSolutionStrategy, this could not
# be solved dynamically.
# On the downside, mypy complains about missing methods or calls to abstract methods
# with trivial body in the (protocol) superclass. We ingore these complaints.


class SPE10EquationsMixin(TPFProtocol):
    """Mixin class to provide the SPE10 model equations and data.

    Takes care of:
    Updates the two-phase flow equations to include:
    - the SPE10 porosity field.
    - the SPE10 permeability field.
    - source terms.

    Two setups for the source terms are implemented:
    - Pure gravity segretation:
        No source terms for either phase.
    - Five-spot setup:
        A volumetric source term for the water phase in the center cell, which
        corresponds to a water injection well. No source term for the oil phase.

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

    def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
        r"""Volumetric phase source term. Given as volumetric flux.

        - Gravity segretation: No source terms for either phase.
        - Five-spot setup: Water (wetting) injection in the center. Mixture production
          in the corners via Dirichlet bc.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        if self.params["spe10_case"] == "gravity_segretation":
            source: np.ndarray = np.zeros(self.g.num_cells)
        elif self.params["spe10_case"] == "five_spot":
            source = np.zeros(self.g.num_cells)
            if phase.name == self.wetting.name:
                # Center well injecting water.
                source[center_cell_id(g)] = phase.convert_units(
                    INJECTION_RATE, "m^3"
                ) / phase.convert_units(pp.DAY, "s")  # 87.5 m^3/day in [m^3/s]
                return source
            # No injection wells for oil.
        else:
            raise ValueError(
                f"Unknown SPE10 case '{self.params['spe10_case']}'."
                + " Supported cases are 'gravity_segretation' and 'five_spot'."
            )

        return source


class SPE10ModifiedBoundaryMixin(TPFProtocol):
    """Mixin specifying the boundary conditions for the SPE10 model.

    Two setups are implemented:
    - Pure gravity segretation:
        Neumann conditions for all faces.
    - Five-spot setup:
        We assign Dirichlet conditions for the corner cells, which act as production
        wells. Neumann conditions for all other faces.

    """

    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann)."""
        if self.params["spe10_case"] == "gravity_segretation":
            domain_sides = self.domain_boundary_sides(g)
            bc = pp.BoundaryCondition(g, domain_sides.north, "dir")
        elif self.params["spe10_case"] == "five_spot":
            height: float = (
                (HEIGHT / 2) if self.params["spe10_quarter_domain"] else HEIGHT
            )
            width: float = WIDTH / 2 if self.params["spe10_quarter_domain"] else WIDTH
            corner_faces: np.ndarray = corner_faces_id(
                g, height, width, PRODUCTION_WELL_SIZE
            )
            bc = pp.BoundaryCondition(g, corner_faces, "dir")
        else:
            raise ValueError(
                f"Unknown SPE10 case '{self.params['spe10_case']}'."
                + " Supported cases are 'gravity_segretation' and 'five_spot'."
            )
        return bc

    def _bc_dirichlet_pressure_values(
        self, g: pp.Grid, phase: FluidPhase
    ) -> np.ndarray:
        """Dirichlet pressure values."""
        if self.params["spe10_case"] == "gravity_segretation":
            bc: np.ndarray = np.zeros(g.num_faces)
        elif self.params["spe10_case"] == "five_spot":
            height: float = (
                (HEIGHT / 2) if self.params["spe10_quarter_domain"] else HEIGHT
            )
            width: float = WIDTH / 2 if self.params["spe10_quarter_domain"] else WIDTH
            corner_faces: np.ndarray = corner_faces_id(
                g, height, width, PRODUCTION_WELL_SIZE
            )
            bc = np.zeros(g.num_faces)
            bc[corner_faces] = phase.convert_units(BHP, "kg*m^-1*s^-2")
        else:
            raise ValueError(
                f"Unknown SPE10 case '{self.params['spe10_case']}'."
                + " Supported cases are 'gravity_segretation' and 'five_spot'."
            )
        return bc


class SPE10SolutionStrategyMixin(TPFProtocol):
    """Mixin class to provide the SPE10 model data.

    Takes care of:
    - Loading the SPE10 fluids, i.e., oil and water.
    - Loading the SPE10 geometry, i.e., permeability and porosity.
    - Exporting the SPE10 geometry.

    Requires the following model parameters to be set:
        - "spe10_layer" (int): The layer of the SPE10 model to use.
        - "spe10_isotropic_perm" (bool): Whether to use isotropic permeability.

    """

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
            NotImplementedError: If the cell size is larger than the SPE10 model cell
                size.

        """
        # Get model parameters and throw some warnings/errors for the user.
        cell_size = self.params["meshing_arguments"]["cell_size"]
        assert isinstance(cell_size, float)
        if cell_size > 20 * FEET:
            raise NotImplementedError(
                "The cell size is larger than the SPE10 model cell size. "
                + "This is not supported yet."
            )
        layer: int = self.params.get("spe10_layer", 1) - 1
        isotropic_perm: bool = self.params.get("spe10_isotropic_perm", True)
        for param_name, value in zip(
            ["spe10_layer", "spe10_isotropic_perm"], [layer, isotropic_perm]
        ):
            if param_name not in self.params:
                warnings.warn(
                    f"The model parameter '{param_name}' is not set."
                    + f" Continuing with default value {value}"
                )

        perm, poro = load_spe10_data(DATA_DIR)
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
        for dim, perm in cast(dict, self.permeability(self.g)).items():
            data.append((self.g, "permeability_" + dim, perm))
        data.append((self.g, "porosity", self.porosity(self.g)))
        self.exporter.add_constant_data(data)

        # For convenience, add the porosity and permeability to the iteration exporter
        # if it exists.
        if hasattr(self, "iteration_exporter"):
            self.iteration_exporter.add_constant_data(data)

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation.

        - Gravity segretation: The upper half of the domain is fully saturated with the
          more dense phase (water), lower half is fully saturated with the less dense
          phase (oil).
        - Five-spot setup: The saturation in the full domain is set to
          ``INITIAL_SATURATION``.

        """

        if self.params["spe10_case"] == "gravity_segretation":
            initial_pressure = np.full(self.g.num_cells, 0.0)
            height: float = (
                (HEIGHT / 2) if self.params["spe10_quarter_domain"] else HEIGHT
            )
            width: float = WIDTH / 2 if self.params["spe10_quarter_domain"] else WIDTH
            # self.g.cell_centers has shape=(ambient_dimension, num_cells)
            # Choose initial saturation depending on whether cell is above or below
            # slanted line
            initial_saturation = np.array(
                [
                    INITIAL_SATURATION
                    if cell[1] >= height / 2 + 10 * (1 - 2 * cell[0] / width)
                    else 1.0 - INITIAL_SATURATION
                    for cell in np.swapaxes(self.g.cell_centers, 0, 1)
                ]
            )
        elif self.params["spe10_case"] == "five_spot":
            initial_pressure = np.full(self.g.num_cells, INITIAL_PRESSURE)
            initial_saturation = np.full(self.g.num_cells, INITIAL_SATURATION)
        else:
            raise ValueError(
                f"Unknown SPE10 case '{self.params['spe10_case']}'."
                + " Supported cases are 'gravity_segretation' and 'five_spot'."
            )

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
        # Ignore mypy. When mixed in with a concrete class, self.set_materials and
        # super().prepare_simulation exist.

        # Initialize materials and geometry early so the grid is available.
        self.set_materials()  # type: ignore
        self.set_geometry()

        # SPE10 porosity/permeability depend on the grid, so load them after geometry
        # creation and before equation setup.
        self.load_spe10_model(self.mdg.subdomains()[0])

        # Continue with the standard simulation setup. This calls set_materials and
        # set_geometry again, which is harmless here.
        super().prepare_simulation()  # type: ignore

        # Export SPE10 fields after exporter initialization.
        self.add_constant_spe10_data()


class SPE10ModelGeometryMixin(TPFProtocol):
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
                "xmax": WIDTH / 2,
                "ymin": 0,
                "ymax": HEIGHT / 2,
            }
        else:
            bounding_box = {
                "xmin": 0,
                "xmax": WIDTH,
                "ymin": 0,
                "ymax": HEIGHT,
            }
        self._domain = pp.Domain(bounding_box)

    def set_fractures(self) -> None:
        """Use fractures as constraints to ensure that the grid is conforming at

        - Gravity segretation: the slanted line at half the height of the domain,
          which separates the two phases in the initial condition.
        - Five spot setup: the production well boundaries.

        """
        if self.params["spe10_case"] == "gravity_segretation":
            height = (HEIGHT / 2) if self.params["spe10_quarter_domain"] else HEIGHT
            width = WIDTH / 2 if self.params["spe10_quarter_domain"] else WIDTH
            self._fractures = [
                pp.LineFracture(
                    np.array([[0, width], [height / 2 + 10, height / 2 - 10]])
                )
            ]

        elif self.params["spe10_case"] == "five_spot":
            self._fractures = [
                pp.LineFracture(
                    np.array([[0, WIDTH], [PRODUCTION_WELL_SIZE, PRODUCTION_WELL_SIZE]])
                ),
                pp.LineFracture(
                    np.array(
                        [
                            [0, WIDTH],
                            [
                                HEIGHT - PRODUCTION_WELL_SIZE,
                                HEIGHT - PRODUCTION_WELL_SIZE,
                            ],
                        ]
                    )
                ),
                pp.LineFracture(
                    np.array(
                        [[PRODUCTION_WELL_SIZE, PRODUCTION_WELL_SIZE], [0, HEIGHT]]
                    )
                ),
                pp.LineFracture(
                    np.array(
                        [
                            [
                                WIDTH - PRODUCTION_WELL_SIZE,
                                WIDTH - PRODUCTION_WELL_SIZE,
                            ],
                            [0, HEIGHT],
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


class SPE10Mixin(
    SPE10EquationsMixin,
    SPE10ModifiedBoundaryMixin,
    SPE10SolutionStrategyMixin,
    SPE10ModelGeometryMixin,
): ...
