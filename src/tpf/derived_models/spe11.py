"""Provide a 5-spot model based on the geometry of the 11th SPE Comparative Solution
Project (SPE11), case A:

[J. M. Nordbotten, M. A. Ferno, B. Flemisch, A. R. Kovscek, and K.-A. Lie, “The 11th
Society of Petroleum Engineers Comparative Solution Project: Problem Definition,” SPE
Journal, vol. 29, no. 05, pp. 2507–2524, May 2024, doi: 10.2118/218015-PA.]

https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/spe_csp11_description.pdf

Additionally, the module provides util functions to download and prepare geometric data
for the model.

"""

import copy
import logging
import pathlib
from typing import Any

# Ignore type checking gmsh due to missing stubs.
import gmsh  # type: ignore
import numpy as np
import porepy as pp
import requests
from numpy.typing import ArrayLike
from porepy.fracs.fracture_importer import dfm_from_gmsh
from porepy.grids.partition import extract_subgrid
from porepy.viz.exporter import DataInput

from tpf.derived_models.fluid_values import co2_reservoir as _co2_reservoir
from tpf.derived_models.fluid_values import co2_surface as _co2_surface
from tpf.derived_models.fluid_values import water as _water
from tpf.derived_models.utils import well_cell_id
from tpf.models.constitutive_laws_tpf import CapPressConstants
from tpf.models.phase import FluidPhase
from tpf.models.protocol import SPE11Protocol, TPFProtocol
from tpf.utils.constants_and_typing import NONWETTING, WETTING

logger = logging.getLogger(__name__)

DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve() / "spe11_data"
ZIP_FILENAME: str = "por_perm_case2a.zip"
URL_CASE_A: str = "https://raw.githubusercontent.com/Simulation-Benchmarks/11thSPE-CSP/refs/heads/main/geometries/spe11a.geo"
URL_CASE_B: str = "https://raw.githubusercontent.com/Simulation-Benchmarks/11thSPE-CSP/refs/heads/main/geometries/spe11b.geo"

GEO_FILE_CASE_A: str = "spe11a.geo"
GEO_FILE_CASE_B: str = "spe11b.geo"

# region MODEL_PARAMETERS
ATM: float = 0.0  # [Pa], atmospheric pressure
RESERVOIR_PRESSURE: float = 3e7

# region case A
case_A: dict[str, Any] = {
    "CASE_NAME": "A",
    "WIDTH": 2.8,  # [m]
    "HEIGHT": 1.2,  # [m]
    "INITIAL_PRESSURE": ATM,  # [Pa]
    "INITIAL_SATURATION": 0.8,  # [-], initial saturation. Domain filled with water.
    "INJECTION_RATE": 1.7e-7 / _co2_surface["density"],  # 1.7x10^- 7kg/s
    "WELL_SIZE": 0.02,  # [m]
    "WELL_1_POS": (0.9, 0.3),  # [m]
    "WELL_2_POS": (
        1.7,
        0.66,
    ),  # [m], lower than in the benchmark so it's approx. in the center of facies 4.
    "MAX_CAP_PRESS": 9.5e4,  # [Pa], upper limit on capillary pressure.
    "PERMEABILITY": {  # Permeability in [m^2]
        "facies 1": 4e-11,
        "facies 2": 5e-10,
        "facies 3": 1e-9,
        "facies 4": 2e-9,
        "facies 5": 4e-9,
        "facies 6": 1e-8,
        "facies 7": 0.0,
    },
    "POROSITY": {  # Porosity in [-]
        "facies 1": 0.44,
        "facies 2": 0.43,
        "facies 3": 0.44,
        "facies 4": 0.45,
        "facies 5": 0.43,
        "facies 6": 0.46,
        "facies 7": 0.0,
    },
    "ENTRY_PRESSURE": {  # Entry pressures in [Pa]
        "facies 1": 1500.0,
        "facies 2": 300.0,
        "facies 3": 100.0,
        "facies 4": 25.0,
        "facies 5": 10.0,
        "facies 6": 1.0,
        "facies 7": 1e-20,  # Epsilon to avoid ill-defined problem.
    },
    "SCALE_FACTOR_X": 1.0,
    "SCALE_FACTOR_Y": 1.0,
}
# endregion

# region case B
case_B: dict[str, Any] = {
    "CASE_NAME": "B",
    "WIDTH": 8400,  # [m]
    "HEIGHT": 1200,  # [m]
    "INITIAL_PRESSURE": RESERVOIR_PRESSURE,  # [Pa], specified only in the center of well 1. Without
    # calculating an equilibrium, we just assume this holds for the full domain.
    "INITIAL_SATURATION": 0.9,  # [-], initial saturation. Domain filled with water.
    "INJECTION_RATE": 0.07 / _co2_reservoir["density"],  # 0.035 kg/s
    "WELL_1_POS": (2700, 300),  # [m]
    "WELL_2_POS": (5100, 660),  # [m]
    "WELL_SIZE": 0.2,  # [m]
    "MAX_CAP_PRESS": 3e7,  # [Pa], upper limit on capillary pressure.
    "PERMEABILITY": {  # Permeability in [m^2]
        "facies 1": 1e-16,
        "facies 2": 1e-13,
        "facies 3": 2e-13,
        "facies 4": 5e-13,
        "facies 5": 1e-12,
        "facies 6": 2e-12,
        "facies 7": 0,
    },
    "POROSITY": {  # Porosity in [-]
        "facies 1": 0.1,
        "facies 2": 0.2,
        "facies 3": 0.2,
        "facies 4": 0.2,
        "facies 5": 0.25,
        "facies 6": 0.35,
        "facies 7": 0,
    },
    "REFINEMENT_FACTOR_BASE": 4000.0,
    "SCALE_FACTOR_X": 3000.0,
    "SCALE_FACTOR_Y": 1000.0,
}
# endregion


water: dict[str, Any] = _water.copy()
water.update(
    {
        # In the SPE11 cases, the residual saturation varies between 0.1 and 0.32,
        # depending on the facies. We use a simplified homogeneous value.
        "residual_saturation": 0.15,  # [-], residual saturation.
    }
)
co2_surface: dict[str, Any] = _co2_surface.copy()
co2_reservoir: dict[str, Any] = _co2_reservoir.copy()

for co2 in [co2_surface, co2_reservoir]:
    co2.update(
        {
            # Same residual gas saturation for case A and B.
            "residual_saturation": 0.1,  # [-], residual saturation.
        }
    )

# endregion


def download_spe11_data(data_dir: pathlib.Path) -> None:
    """Download the SPE11 geometric data and store it."""
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the ZIP file.
    for url, file in zip([URL_CASE_A, URL_CASE_B], [GEO_FILE_CASE_A, GEO_FILE_CASE_B]):
        logger.info(f"Downloading dataset from {url}")
        response = requests.get(url)
        response.raise_for_status()

        with (data_dir / file).open("wb") as f:
            f.write(response.content)

    logger.info("Download completed.")


def read_refinement_factor(geo_file: pathlib.Path) -> float:
    """Read the refinement factor in the SPE11 geometric information."""
    with geo_file.open("r") as f:
        lines: list[str] = f.readlines()
    line_idx = 3 if geo_file.name == GEO_FILE_CASE_A else 4
    return float(lines[line_idx][36:-3])


def write_refinement_factor(
    geo_file: pathlib.Path, refinement_factor: float = 1.0
) -> None:
    """Adjust the refinement factor in the SPE11 geometric information."""
    with geo_file.open("r+") as f:
        lines: list[str] = f.readlines()
        line_idx = 3 if geo_file.name == GEO_FILE_CASE_A else 4
        lines[line_idx] = (
            f"DefineConstant[ refinement_factor = {refinement_factor} ];\n"
        )
        # Replace the current content of the file.
        f.seek(0)
        f.write("".join(lines))
        f.truncate()


def write_well_positions(geo_file: pathlib.Path, case: dict[str, Any]) -> None:
    """Add the well positions in the SPE11 geometric information.

    Note: The line numbers are hardcoded and should only changed with care. If something
        is added or removed, they have to be changed.

    """
    # Data is written to the `spe11a.geo` file and upscaled to case B size if necessary.
    # All positions are therefore in case A scale.

    scale_factor_x = case["SCALE_FACTOR_X"]
    scale_factor_y = case["SCALE_FACTOR_Y"]

    well_1_pos_x = case["WELL_1_POS"][0] / scale_factor_x
    well_1_pos_y = case["WELL_1_POS"][1] / scale_factor_y
    well_2_pos_x = case["WELL_2_POS"][0] / scale_factor_x
    well_2_pos_y = case["WELL_2_POS"][1] / scale_factor_y
    well_size_x = case["WELL_SIZE"] / scale_factor_x
    well_size_y = case["WELL_SIZE"] / scale_factor_y

    NUM_SQUARES: int = 4

    # Additional range
    with geo_file.open("r+") as f:
        lines: list[str] = f.readlines()
        pts_lines = lines[:325].copy()
        lns_lines = lines[325:643].copy()
        surface_lines = lines[643:].copy()

        # NOTE The following is done already now, so we don't have to keep track of line
        # counts when we add lines to ``surface_lines``

        # Remove larges squares (to be defined later) from existing surfaces.
        surface_lines[17] = (
            "Plane Surface(2) = {2, " + f"{300 + NUM_SQUARES * 100 + 20}" + "};\n"
        )
        surface_lines[43] = (
            "Plane Surface(11) = {11, " + f"{300 + NUM_SQUARES * 100 + 21}" + "};\n"
        )

        # Add all squares as physical surfaces.
        surface_lines[26] = (
            'Physical Surface("Facies 5", 5) = {2, 3, 4, 5, 6, '
            + ", ".join([f"{400 + i * 100 + 22}" for i in range(NUM_SQUARES)])
            + "};\n"
        )
        surface_lines[54] = (
            'Physical Surface("Facies 4", 4) = {10, 11, 12, 13, 14, 15, 22, '
            + ", ".join([f"{400 + i * 100 + 23}" for i in range(NUM_SQUARES)])
            + "};\n"
        )

        # Add wells plus 3 boxes of increasing size around the wells to ensure
        # well-conditioned cells. Without the boxes, the large size difference between
        # the wells and other features can create ill-conditioned cells.
        for i in range(NUM_SQUARES):
            # Base entity ID > 400 to avoid collisions with existing entities.
            base_id = 400 + 100 * i

            # Four corner points per square.
            well_1_pts: list[str] = [
                f"Point({base_id}) = {{{well_1_pos_x - well_size_x * (2 ** (i - 1))}, {well_1_pos_y - well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
                f"Point({base_id + 1}) = {{{well_1_pos_x + well_size_x * (2 ** (i - 1))}, {well_1_pos_y - well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
                f"Point({base_id + 2}) = {{{well_1_pos_x - well_size_x * (2 ** (i - 1))}, {well_1_pos_y + well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
                f"Point({base_id + 3}) = {{{well_1_pos_x + well_size_x * (2 ** (i - 1))}, {well_1_pos_y + well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
            ]
            well_2_pts: list[str] = [
                f"Point({base_id + 4}) = {{{well_2_pos_x - well_size_x * (2 ** (i - 1))}, {well_2_pos_y - well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
                f"Point({base_id + 5}) = {{{well_2_pos_x + well_size_x * (2 ** (i - 1))}, {well_2_pos_y - well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
                f"Point({base_id + 6}) = {{{well_2_pos_x - well_size_x * (2 ** (i - 1))}, {well_2_pos_y + well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
                f"Point({base_id + 7}) = {{{well_2_pos_x + well_size_x * (2 ** (i - 1))}, {well_2_pos_y + well_size_y * (2 ** (i - 1))}, 0, cl__1}};\n",
            ]

            # Sides of the square.
            well_1_lns: list[str] = [
                f"Line({base_id + 10}) = {{{base_id}, {base_id + 2}}};\n",
                f"Line({base_id + 11}) = {{{base_id + 2}, {base_id + 3}}};\n",
                f"Line({base_id + 12}) = {{{base_id + 3}, {base_id + 1}}};\n",
                f"Line({base_id + 13}) = {{{base_id + 1}, {base_id}}};\n",
            ]
            well_2_lns: list[str] = [
                f"Line({base_id + 14}) = {{{base_id + 4}, {base_id + 6}}};\n",
                f"Line({base_id + 15}) = {{{base_id + 6}, {base_id + 7}}};\n",
                f"Line({base_id + 16}) = {{{base_id + 7}, {base_id + 5}}};\n",
                f"Line({base_id + 17}) = {{{base_id + 5}, {base_id + 4}}};\n",
            ]

            # Squares oriented in both directions.
            line_loops: list[str] = [
                f"Line Loop({base_id + 20}) = {{{base_id + 10}, {base_id + 11}, {base_id + 12}, {base_id + 13}}};\n",
                f"Line Loop({base_id + 21}) = {{{base_id + 14}, {base_id + 15}, {base_id + 16}, {base_id + 17}}};\n",
                f"Line Loop({base_id + 22}) = {{-{base_id + 10}, -{base_id + 11}, -{base_id + 12}, -{base_id + 13}}};\n",
                f"Line Loop({base_id + 23}) = {{-{base_id + 14}, -{base_id + 15}, -{base_id + 16}, -{base_id + 17}}};\n",
            ]

            if i == 0:
                # Add the smallest square.
                surfaces: list[str] = [
                    f"Plane Surface({base_id + 22}) = {{{base_id + 22}}};\n",
                    f"Plane Surface({base_id + 23}) = {{{base_id + 23}}};\n",
                ]
            else:
                # Add square "rings" with the next-smallest square as a hole.
                surfaces = [
                    f"Plane Surface({base_id + 22}) = {{{base_id + 22}, {base_id - 100 + 20}}};\n",
                    f"Plane Surface({base_id + 23}) = {{{base_id + 23}, {base_id - 100 + 21}}};\n",
                ]

            # Add to existing entities.
            pts_lines.extend(well_1_pts)
            pts_lines.extend(well_2_pts)
            lns_lines.extend(well_1_lns)
            lns_lines.extend(well_2_lns)

            # Insert add the start of ``surface_lines``, because the later lines refer
            # to the added entities (see above). Insert in order of the squares, because
            # the squares refer to each other.
            # NOTE ``surfaces``, then ``line_loops`` so line loops come FIRST in the
            # final file. Order is important, because the surfaces refer to the loops.
            for line in surfaces + line_loops:
                surface_lines.insert(i * 6, line)

        # Replace the current content of the file.
        new_lines = pts_lines + lns_lines + surface_lines
        f.seek(0)
        f.write("".join(new_lines))
        f.truncate()


def fix_face_normals(
    gmsh_file: pathlib.Path, mesh_normal: np.ndarray = np.array([0, 0, 1])
) -> pathlib.Path:
    """Fix the SPE11 mesh s.t. all face normals point in the same direction.

    Writes the fixed mesh to a `.msh` file with the same name as the input file.

    """
    if gmsh_file.suffix == ".msh":
        out_file: pathlib.Path = gmsh_file
        gmsh.open(str(gmsh_file))
    else:
        out_file = gmsh_file.with_suffix(".msh")

        # Generate the mesh.
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 3)
        # Use Frontal-Delaunay meshing algorithm for high quality elements.
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.merge(str(gmsh_file))
        gmsh.model.mesh.generate(dim=2)
        gmsh.model.mesh.createGeometry()

    gmsh.write(str(out_file))

    # Get all entities
    entities: list[tuple[int, int]] = gmsh.model.getEntities(2)
    points: np.ndarray = gmsh.model.mesh.getNodes(-1, -1)[1].reshape(-1, 3)

    # Function to calculate the normal of a triangle
    def calculate_normal(triangle: np.ndarray) -> np.ndarray:
        p1, p2, p3 = points[triangle]
        normal: np.ndarray = np.cross(p2 - p1, p3 - p1)
        return normal / np.linalg.norm(normal)

    # Fix normals
    num_false_normals: int = 0
    num_cells: int = 0

    for entity in entities:
        element_types, element_tags, node_tags = gmsh.model.mesh.get_elements(
            entity[0], entity[1]
        )
        for elem_type, elem_tags, nodes in zip(element_types, element_tags, node_tags):
            if elem_type == 2:
                # Convert to zero-based index
                triangles: np.ndarray = nodes.reshape(-1, 3) - 1

                # Loop through all simplices and swap order of vertices if normal points
                # in the wrong direction.
                for i, triangle in enumerate(triangles):
                    normal: np.ndarray = calculate_normal(triangle)
                    num_cells += 1
                    if np.dot(normal, mesh_normal) < 0:
                        triangles[i] = [triangle[1], triangle[0], triangle[2]]
                        num_false_normals += 1
                gmsh.model.mesh.remove_elements(entity[0], entity[1])
                gmsh.model.mesh.add_elements(
                    entity[0],
                    entity[1],
                    [elem_type],
                    [elem_tags],
                    [triangles.flatten() + 1],
                )

    logger.info(f"Fixed {num_false_normals} of {num_cells} normals.")

    gmsh.write(str(out_file))
    gmsh.finalize()
    return out_file


def load_spe11_data(
    data_dir: pathlib.Path,
    case: dict[str, Any] = case_A,
    refinement_factor: float = 1.0,
) -> pp.MixedDimensionalGrid:
    """Load the SPE11 data into a :class:`~numpy.ndarray`.

    Parameters:
        data_dir: The directory containing the .geo file.

    Returns:
        mdg: PorePy grid.

    """
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    geo_file: pathlib.Path = data_dir / (
        GEO_FILE_CASE_A if case["CASE_NAME"] == "A" else GEO_FILE_CASE_B
    )

    # Assure that the `.geo` file exists. In lieu of a goto statement, run the
    # following loop at maximum twice.
    i: int = 0
    while True:
        if geo_file.exists():
            break
        if i >= 1:
            raise FileNotFoundError(
                "Could not locate the .geo file. Perhaps, the download failed"
            )
        logger.info(".geo file not found. Downloading again ...")
        download_spe11_data(data_dir)
        i += 1

    if read_refinement_factor(geo_file) != refinement_factor:
        logger.info(
            "Refinement factor in the .geo file is wrong. Adjusting and"
            + " recomputing mesh ..."
        )
        write_refinement_factor(geo_file, refinement_factor)

    logger.warning(
        "Well positions may be wrong if the files were downloaded previously and you"
        + " switch between cases. Delete all .geo files and rerun. "
    )

    # The well positions always gets written to the case A geo file.
    with (data_dir / GEO_FILE_CASE_A).open("r") as f:
        lines: list[str] = f.readlines()
        # NOTE Line number must align with ``write_well_positions``. 823 is valid for
        # NUM_SQUARES=4.
        if len(lines) != 823:
            logger.info(
                "Well positions not included in the .geo file. Adjusting and"
                + " recomputing mesh ..."
            )
            write_well_positions(data_dir / GEO_FILE_CASE_A, case)

    gmsh_file: pathlib.Path = fix_face_normals(geo_file)

    logger.info("Loading mesh.")
    mdg = dfm_from_gmsh(str(gmsh_file), dim=2)
    return mdg


def LeverettJfunction(permeability: ArrayLike, porosity: ArrayLike) -> np.ndarray:
    r"""Calculate the Leverett J-function.

    .. math::
        p_e = \sqrt{\frac{\phi}{k_x}} \cdot 6.12 \times 10^{-3} \text{N}/\text{m}

    """
    permeability = np.asarray(permeability)
    porosity = np.asarray(porosity)
    return np.sqrt(porosity / permeability) * 6.12e-3


# NOTE All SPE11 classes are purely mixins, i.e., the only superclasses are protocols.
# This ensures that the SPE11 model can be added on top of both adaptive homotopy
# continuation and adaptive Newton.
# For example, super().prepare_simulation calls either
# SolutionStrategyHC.prepare_simulation or EstimatesSolutionStrategy.prepare_simulation.
# If SPE11SolutionStrategyMixin would subclass EstimatesSolutionStrategy, this could not
# be solved dynamically.
# On the downside, mypy complains about missing methods or calls to abstract methods
# with trivial body in the (protocol) superclass. We ingore these complaints.


class SPE11CapillaryPressureMixin(SPE11Protocol, TPFProtocol):
    """Spatially heterogeneous capillary pressure and upper limit on capillary pressure."""

    def set_cap_press_constants(self) -> None:
        updated_constants = {"max": self.spe11_params["MAX_CAP_PRESS"], "limit": True}

        if self.uses_hc:
            constants: dict[str, dict] = self.params.get("cap_press_constants", {})
            cap_press_1_constants: dict[str, Any] = constants.get("model_1", {})
            cap_press_2_constants: dict[str, Any] = constants.get("model_2", {})
            cap_press_1_constants.update(updated_constants)
            cap_press_2_constants.update(updated_constants)
            self._cap_press_constants_1 = CapPressConstants(**cap_press_1_constants)
            self._cap_press_constants_2 = CapPressConstants(**cap_press_2_constants)

        else:
            cap_press_constants = self.params.get("cap_press_constants", {})
            cap_press_constants.update(updated_constants)
            self._cap_press_constants = CapPressConstants(**cap_press_constants)

    def entry_pressure(
        self,
        g: pp.Grid | pp.BoundaryGrid,
        cap_press_constants: CapPressConstants | None = None,
    ) -> pp.ad.Operator:
        r"""Entry pressure function.

        Parameters:
            g: Grid object.
            cap_press_constants: Capillary pressure constants. If set, overrides
                :attr:`self._cap_press_constants`. Default is ``None``.

        Returns:
            Entry pressure.

        """
        if self.params.get("spe11_heterogeneous_cap_pressure", True):
            # Ignore mypy. g is a pp.Grid and self.entry_pressure_np returns an array in
            # this case.
            return pp.ad.DenseArray(
                self.entry_pressure_np(g, cap_press_constants)  # type: ignore
            )
        else:
            return pp.ad.Scalar(self.params["spe11_entry_pressure"])

    def entry_pressure_np(
        self, g: pp.Grid, cap_press_constants: CapPressConstants | None = None
    ) -> float | np.ndarray:
        r"""Entry pressure function for numpy.

        Parameters:
            g: Grid object.
            cap_press_constants: Capillary pressure constants. If set, overrides
                :attr:`self._cap_press_constants`. Default is ``None``.

        Returns:
            Entry pressure.

        """
        if self.params.get("spe11_heterogeneous_cap_pressure", True):
            if self.spe11_case == "A":
                entry_pressure: np.ndarray = np.zeros(g.num_cells)
                for facies, ep in self.spe11_params["ENTRY_PRESSURE"].items():
                    entry_pressure[g.tags[facies + "_simplices"]] = ep
            elif self.spe11_case == "B":
                entry_pressure = LeverettJfunction(
                    self.permeability(g)["kxx"], self.porosity(g)
                )

            return entry_pressure
        else:
            return self.params["spe11_entry_pressure"]


class SPE11EquationsMixin(SPE11Protocol, TPFProtocol):
    """Mixin class to provide the SPE11 model inspired equations and data.

    Takes care of:
    Updates the two-phase flow equations to include:
    - the SPE11 porosity field.
    - the SPE11 permeability field.
    - A volumetric source term for the water phase in the center cell.
    - Production wells in the corner cells.

    """

    def permeability(self, g: pp.Grid) -> np.ndarray | dict[str, np.ndarray]:
        """Solid permeability. Units are set by :attr:`self.solid`."""
        permeability: np.ndarray = np.zeros(g.num_cells)
        for facies, perm in self.spe11_params["PERMEABILITY"].items():
            permeability[g.tags[facies + "_simplices"]] = perm
        return {"kxx": permeability}

    def porosity(self, g: pp.Grid) -> np.ndarray:
        """Solid porosity."""
        porosity: np.ndarray = np.zeros(g.num_cells)
        for facies, por in self.spe11_params["POROSITY"].items():
            porosity[g.tags[facies + "_simplices"]] = por
        return porosity

    def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
        r"""Volumetric phase source term. Given as volumetric flux.

        Two CO2 injection wells are placed. One in the upper section of the domain and
        one in the lower section.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        array = np.zeros(g.num_cells)

        if phase.name == self.nonwetting.name:
            # Two wells injecting CO2.

            # Lower well.
            well_1_ids: np.ndarray = well_cell_id(
                g,
                np.array(
                    [
                        [
                            self.spe11_params["WELL_1_POS"][0]
                            - self.spe11_params["WELL_SIZE"] / 2,
                            self.spe11_params["WELL_1_POS"][1]
                            - self.spe11_params["WELL_SIZE"] / 2,
                        ],
                        [
                            self.spe11_params["WELL_1_POS"][0]
                            + self.spe11_params["WELL_SIZE"] / 2,
                            self.spe11_params["WELL_1_POS"][1]
                            + self.spe11_params["WELL_SIZE"] / 2,
                        ],
                    ]
                ),
            )
            array[well_1_ids] = phase.convert_units(
                self.spe11_params["INJECTION_RATE"], "m^3"
            ) / (phase.convert_units(pp.SECOND, "s") * len(well_1_ids))
            # Upper well.
            well_2_ids: np.ndarray = well_cell_id(
                g,
                np.array(
                    [
                        [
                            self.spe11_params["WELL_2_POS"][0]
                            - self.spe11_params["WELL_SIZE"] / 2,
                            self.spe11_params["WELL_2_POS"][1]
                            - self.spe11_params["WELL_SIZE"] / 2,
                        ],
                        [
                            self.spe11_params["WELL_2_POS"][0]
                            + self.spe11_params["WELL_SIZE"] / 2,
                            self.spe11_params["WELL_2_POS"][1]
                            + self.spe11_params["WELL_SIZE"] / 2,
                        ],
                    ]
                ),
            )
            array[well_2_ids] = phase.convert_units(
                self.spe11_params["INJECTION_RATE"], "m^3"
            ) / (phase.convert_units(pp.SECOND, "s") * len(well_2_ids))
            return array

        # No water is injected.
        return array


class SPE11ModifiedBoundaryMixin(SPE11Protocol, TPFProtocol):
    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann).

        Dirichlet bc at the left and right. No flow Neumann bc at the bottom and top.

        """
        domain_sides = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(
            g, faces=np.logical_or(domain_sides.west, domain_sides.east), cond="dir"
        )

    def _bc_dirichlet_pressure_values(
        self, g: pp.Grid, phase: FluidPhase
    ) -> np.ndarray:
        """Dirichle pressure values."""
        domain_sides = self.domain_boundary_sides(g)
        bc_values: np.ndarray = np.zeros(g.num_faces)
        # Boundary pressure values are identical to initial pressure.
        bc_values[np.logical_or(domain_sides.west, domain_sides.east)] = (
            phase.convert_units(self.spe11_params["INITIAL_PRESSURE"], "kg*m^-1*s^-2")
        )
        return bc_values


class SPE11ModelGeometryMixin(SPE11Protocol, TPFProtocol):
    def set_domain(self) -> None:
        """Set domain of the problem."""
        box: dict[str, pp.number] = {
            "xmax": self.spe11_params["WIDTH"],
            "ymax": self.spe11_params["HEIGHT"],
        }
        self._domain = pp.Domain(box)

    def set_geometry(self) -> None:
        self.set_domain()
        # Load full domain including inactive cells, i.e., areas with porosity zero.
        self.extended_domain = load_spe11_data(
            DATA_DIR,
            self.spe11_params,
            self.params["meshing_arguments"].get("spe11_refinement_factor", 10.0),
        )
        extended_domain_g = self.extended_domain.subdomains()[0]

        # Extract subgrid corresponding to the active cells.
        self.active_cells = np.nonzero(self.porosity(extended_domain_g))[0]
        active_g, _, _ = extract_subgrid(extended_domain_g, self.active_cells)

        # Copy facies tags.
        for key, value in extended_domain_g.tags.items():
            if key.startswith("facies"):
                active_g.tags[key] = value[self.active_cells]

        # Set subgrid as domain.
        self.mdg = copy.deepcopy(self.extended_domain)
        self.mdg.remove_subdomain(self.mdg.subdomains()[0])
        self.mdg.add_subdomains(active_g)
        self.nd: int = self.mdg.dim_max()

        # Check that the domain size is correct.
        height: float = np.max(extended_domain_g.nodes[1, :]) - np.min(
            extended_domain_g.nodes[1, :]
        )
        width: float = np.max(extended_domain_g.nodes[0, :]) - np.min(
            extended_domain_g.nodes[0, :]
        )
        assert np.isclose(height, self.spe11_params["HEIGHT"])
        assert np.isclose(width, self.spe11_params["WIDTH"])


class SPE11SolutionStrategyMixin(SPE11Protocol, TPFProtocol):
    """Mixin class to provide the SPE11 model data.

    Takes care of:
    - Loading the SPE11 fluids, i.e., CO2 and water.
    - Loading the SPE11 geometry, i.e., permeability and porosity.
    - Exporting the SPE11 geometry.

    """

    def __init__(self, params: dict | None) -> None:
        # self.spe11_case has to be set before self.set_phases is called.
        if params is None:
            params = {}
        self.spe11_case: str = params.get("spe11_case", "A")

        if self.spe11_case == "A":
            self.spe11_params = case_A
        elif self.spe11_case == "B":
            self.spe11_params = case_B
        else:
            raise ValueError(
                f"Unknown SPE11 case {self.spe11_case}. "
                + "Please choose either 'A' or 'B'."
            )

        # Ignore mypy. When mixed in with a concrete class, super().__init__ takes
        # params.
        super().__init__(params)  # type: ignore

    def set_phases(self) -> None:
        self.phases: dict[str, FluidPhase] = {}
        co2 = co2_surface if self.spe11_case == "A" else co2_reservoir
        for phase_name, constants in zip([WETTING, NONWETTING], [water, co2]):
            phase = FluidPhase(constants)
            phase.set_units(self.units)
            setattr(self, phase_name, phase)
            self.phases[phase_name] = phase

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        initial_pressure = np.full(
            self.g.num_cells, self.spe11_params["INITIAL_PRESSURE"]
        )
        initial_saturation = np.full(
            self.g.num_cells, self.spe11_params["INITIAL_SATURATION"]
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
        self.set_materials()  # type: ignore
        self.set_geometry()
        super().prepare_simulation()  # type: ignore
        self.add_constant_spe11_data()

    def add_constant_spe11_data(self) -> None:
        """Save the SPE11 data to the exporter."""
        data: list[DataInput] = []

        # Add zero values on inactive cells for proper visualization.
        extended_domain_g = self.extended_domain.subdomains()[0]
        full_array = np.zeros(extended_domain_g.num_cells)

        # Ignore mypy. Permeability is always dict.
        for dim, perm in self.permeability(self.g).items():  # type: ignore
            extended_perm = full_array.copy()
            extended_perm[self.active_cells] = perm
            data.append((extended_domain_g, "permeability_" + dim, extended_perm))

        extended_porosity = full_array.copy()
        extended_porosity[self.active_cells] = self.porosity(self.g)
        data.append((extended_domain_g, "porosity", extended_porosity))

        # Entry pressure is only a distribution if it is heterogeneous.
        if self.params["spe11_heterogeneous_cap_pressure"]:
            extended_entry_pressure = full_array.copy()
            extended_entry_pressure[self.active_cells] = self.entry_pressure_np(self.g)
            data.append((extended_domain_g, "entry_pressure", extended_entry_pressure))

        self.exporter.add_constant_data(data)

        # For convenience, add the porosity and permeability to the iteration exporter
        # if it exists.
        if hasattr(self, "iteration_exporter"):
            self.iteration_exporter.add_constant_data(data)


class SPE11DataSavingEstMixin(SPE11Protocol, TPFProtocol):
    def _data_to_export(
        self, time_step_index: int | None = None, iterate_index: int | None = None
    ) -> list[DataInput]:

        # Ignore mypy. When mixed in, super()._data_to_export exists.
        data: list[tuple[pp.Grid, str, np.ndarray]] = super()._data_to_export(  # type: ignore
            time_step_index, iterate_index
        )
        updated_data: list[DataInput] = []

        # Add zero values on inactive cells for proper visualization.
        extended_domain_g = self.extended_domain.subdomains()[0]
        for _, name, array in data:
            full_array = np.zeros(extended_domain_g.num_cells)
            full_array[self.active_cells] = array
            array = full_array
            updated_data.append((extended_domain_g, name, array))

        return updated_data

    def initialize_data_saving(self) -> None:
        self.exporter = pp.Exporter(
            self.extended_domain,
            self.params["file_name"],
            folder_name=self.params["folder_name"],
            export_constants_separately=self.params.get(
                "export_constants_separately", False
            ),
            length_scale=self.units.m,
        )

        if "solver_statistics_file_name" in self.params:
            self.nonlinear_solver_statistics.path = (
                pathlib.Path(self.params["folder_name"])
                / self.params["solver_statistics_file_name"]
            )


class SPE11Mixin(
    SPE11CapillaryPressureMixin,
    SPE11EquationsMixin,
    SPE11ModifiedBoundaryMixin,
    SPE11SolutionStrategyMixin,
    SPE11ModelGeometryMixin,
    SPE11DataSavingEstMixin,
): ...
