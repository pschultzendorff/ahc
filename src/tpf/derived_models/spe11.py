"""Provide a 5-spot model based on the geometry of the 11th SPE Comparative Solution
Project (SPE11), case A:

[J. M. Nordbotten, M. A. Ferno, B. Flemisch, A. R. Kovscek, and K.-A. Lie, “The 11th
Society of Petroleum Engineers Comparative Solution Project: Problem Definition,” SPE
Journal, vol. 29, no. 05, pp. 2507–2524, May 2024, doi: 10.2118/218015-PA.]

https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/spe_csp11_description.pdf

Additionally, the module provides util functions to download and prepare geometric data
for the model.

"""

import logging
import pathlib
from functools import partial
from typing import Any

import gmsh
import numpy as np
import porepy as pp
import requests
from numpy.typing import ArrayLike
from porepy.fracs.fracture_importer import dfm_from_gmsh
from porepy.viz.exporter import DataInput

from tpf.derived_models.fluid_values import co2 as _co2
from tpf.derived_models.fluid_values import water as _water
from tpf.derived_models.utils import cell_id_position
from tpf.models.constitutive_laws_tpf import CapPressConstants
from tpf.models.phase import FluidPhase
from tpf.models.protocol import SPE11Protocol, TPFProtocol
from tpf.utils.constants_and_typing import NONWETTING, WETTING

logger = logging.getLogger(__name__)

DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent / "spe11_data"
ZIP_FILENAME: str = "por_perm_case2a.zip"
URL: str = "https://raw.githubusercontent.com/Simulation-Benchmarks/11thSPE-CSP/refs/heads/main/geometries/spe11a.geo"

# region MODEL_PARAMETERS
ATM: float = 101325.0  # [Pa], atmospheric pressure

# region case A
case_A: dict[str, Any] = {
    "WIDTH": 2.8,  # [m]
    "HEIGHT": 1.2,  # [m]
    "INITIAL_PRESSURE": ATM,  # [Pa]
    "INITIAL_SATURATION": 0.9,  # [-], initial saturation. Domain filled with water.
    "INJECTION_RATE": 1.75e-7 / _co2["density"],  # 1.7x10^-5 kg/s
    "WELL_1_POS": (0.9, 0.3),  # [m]
    "WELL_2_POS": (1.7, 0.7),  # [m]
    "MAX_CAP_PRESS": 9.5e4,  # [Pa], upper limit on capillary pressure.
    "PERMEABILITY": {  # Permeability in [m^2]
        "facies 1": 4e-11,
        "facies 2": 5e-10,
        "facies 3": 1e-9,
        "facies 4": 2e-9,
        "facies 5": 4e-9,
        "facies 6": 1e-8,
        "facies 7": 1e-30,  # Epsilon to avoid division by zero when calculating face transmissibilities.
    },
    "POROSITY": {  # Porosity in [-]
        "facies 1": 0.44,
        "facies 2": 0.43,
        "facies 3": 0.44,
        "facies 4": 0.45,
        "facies 5": 0.43,
        "facies 6": 0.46,
        "facies 7": 1e-20,  # Epsilon to avoid ill-defined problem.
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
}
# endregion

# region case B
case_B: dict[str, Any] = {
    "WIDTH": 8400,  # [m]
    "HEIGHT": 1200,  # [m]
    "INITIAL_PRESSURE": 3e7,  # [Pa], specified only in the center of well 1. Without
    # calculating an equilibrium, we just assume this holds for the full domain.
    "INITIAL_SATURATION": 0.9,  # [-], initial saturation. Domain filled with water.
    "INJECTION_RATE": 1.75e-7 / _co2["density"],  # 1.7x10^-5 kg/s
    "WELL_1_POS": (2700, 300),  # [m]
    "WELL_2_POS": (5100, 700),  # [m]
    "MAX_CAP_PRESS": 3e7,  # [Pa], upper limit on capillary pressure.
    "PERMEABILITY": {  # Permeability in [m^2]
        "facies 1": 1e-16,
        "facies 2": 1e-13,
        "facies 3": 2e-3,
        "facies 4": 5e-13,
        "facies 5": 1e-12,
        "facies 6": 2e-12,
        "facies 7": 1e-30,  # Epsilon to avoid division by zero when calculating face transmissibilities.
    },
    "POROSITY": {  # Porosity in [-]
        "facies 1": 0.1,
        "facies 2": 0.2,
        "facies 3": 0.2,
        "facies 4": 0.2,
        "facies 5": 0.25,
        "facies 6": 0.35,
        "facies 7": 1e-20,  # Epsilon to avoid ill-defined problem.
    },
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
co2: dict[str, Any] = _co2.copy()
co2.update(
    {
        # Same residual gas saturation for case A and B.
        "residual_saturation": 0.1,  # [-], residual saturation.
    }
)

# endregion


def download_spe11_data(data_dir: pathlib.Path) -> None:
    """Download the SPE11 geometric information and store them locally."""
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the ZIP file.
    logger.info(f"Downloading dataset from {URL}")
    response = requests.get(URL)
    response.raise_for_status()

    with (data_dir / "spe11a.geo").open("wb") as f:
        f.write(response.content)
    logger.info("Download completed.")


def read_refinement_factor(geo_file: pathlib.Path) -> float:
    """Read the refinement factor in the SPE11 geometric information."""
    with geo_file.open("r") as f:
        lines: list[str] = f.readlines()
    return float(lines[3][36:-3])


def write_refinement_factor(
    geo_file: pathlib.Path, refinement_factor: float = 1.0
) -> None:
    """Adjust the refinement factor in the SPE11 geometric information."""
    with geo_file.open("r+") as f:
        lines: list[str] = f.readlines()
        lines[3] = f"DefineConstant[ refinement_factor = {refinement_factor} ];\n"
        # Replace the current content of the file.
        f.seek(0)
        f.write("".join(lines))
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
        gmsh.merge(str(gmsh_file))
        gmsh.model.mesh.generate(dim=2)
        gmsh.model.mesh.createGeometry()

    # Get all entities
    entities: list[tuple[int, int]] = gmsh.model.getEntities(2)
    points: np.ndarray = gmsh.model.mesh.getNodes(-1, -1)[1].reshape(-1, 3)  # type: ignore

    # Function to calculate the normal of a triangle
    def calculate_normal(triangle: np.ndarray) -> np.ndarray:
        nonlocal points
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
                triangles: np.ndarray = nodes.reshape(-1, 3) - 1  # type: ignore
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
    data_dir: pathlib.Path, refinement_factor: float = 1.0
) -> pp.MixedDimensionalGrid:
    """Load the SPE11 data into a :class:`~numpy.ndarray`.

    Parameters:
        data_dir: The directory containing the .geo file.

    Returns:
        tuple: A tuple containing:

    """
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Assure that the `.geo` file is available. In lieu of a goto statement, run the
    # following loop at maximum twice.
    i: int = 0
    geo_file: pathlib.Path | None = None
    while True:
        for filename in data_dir.iterdir():
            if filename.suffix == ".geo":
                geo_file = data_dir / filename
        if geo_file is None:
            if i >= 1:
                raise FileNotFoundError(
                    "Could not locate the .geo file. Perhaps, the download failed"
                )
            logger.info(".geo file not found. Downloading again ...")
            download_spe11_data(data_dir)
        else:
            break
        i += 1

    if read_refinement_factor(geo_file) != refinement_factor:
        logger.info(
            "Refinement factor in the .geo file is wrong. Adjusting and"
            + " recomputing mesh ..."
        )
        write_refinement_factor(geo_file, refinement_factor)

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


class CapillaryPressureSPE11(SPE11Protocol, TPFProtocol):
    """Spatially heterogeneous capillary pressure and upper limit on capillary pressure."""

    def entry_pressure(
        self,
        g: pp.Grid | pp.BoundaryGrid,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs,
    ) -> pp.ad.Operator:
        r"""Entry pressure function.

        Parameters:
            g: Grid object.
            cap_press_constants: Capillary pressure constants. If set, overrides
                :attr:`self._cap_press_constants`. Default is ``None``.
            **kwargs: May contain the following keywords:
                - faces: Flag indicating whether the entry pressure is evaluated on cell
                    faces (instead of the default cell centers). Default is False.

        Returns:
            Entry pressure.

        """
        # We know that the entry pressure is a np.ndarray.
        return pp.ad.DenseArray(
            self.entry_pressure_np(g, cap_press_constants, **kwargs)  # type: ignore
        )

    def entry_pressure_np(
        self, g: pp.Grid, cap_press_constants: CapPressConstants | None = None, **kwargs
    ) -> float | np.ndarray:
        r"""Entry pressure function for numpy.

        Parameters:
            g: Grid object.
            cap_press_constants: Capillary pressure constants. If set, overrides
                :attr:`self._cap_press_constants`. Default is ``None``.
            **kwargs: May contain the following keywords:
                - faces: Flag indicating whether the entry pressure is evaluated on cell
                    faces (instead of the default cell centers). Default is False.

        Returns:
            Entry pressure.

        """
        if self.params["spe11_case"] == "A":
            entry_pressure: np.ndarray = np.zeros(g.num_cells)
            for facies, ep in self.spe11_params["ENTRY_PRESSURE"].items():
                entry_pressure[g.tags[facies + "_simplices"]] = ep
        elif self.params["spe11_case"] == "B":
            entry_pressure: np.ndarray = LeverettJfunction(
                self.permeability(g)["kxx"], self.porosity(g)
            )

        # Map to faces.
        if kwargs.get("faces", False):
            cells_to_faces = g.cell_faces
            entry_pressure = cells_to_faces @ entry_pressure
            # Divide inner faces by 2 to obtain the averaged value.
            entry_pressure[g.get_internal_faces()] /= 2

        return entry_pressure

    def cap_press(
        self,
        saturation_w: pp.ad.Operator,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs,
    ) -> pp.ad.Operator:
        cap_press: pp.ad.Operator = super().cap_press(
            saturation_w, cap_press_constants, **kwargs
        )
        # Limit the capillary pressure to a maximum value.
        maximum_func = pp.ad.Function(
            partial(
                pp.ad.maximum,
                var_1=self.spe11_params["CAP_PRESS_LIMIT"],
            ),
            "max",
        )

        cap_press = maximum_func(cap_press)  # type: ignore
        return cap_press

    def cap_press_np(
        self,
        saturation_w: np.ndarray,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs,
    ) -> np.ndarray:
        cap_press: np.ndarray = super().cap_press_np(
            saturation_w, cap_press_constants, **kwargs
        )
        # Limit the capillary pressure to a maximum value.
        cap_press = np.maximum(cap_press, self.spe11_params["CAP_PRESS_LIMIT"])
        return cap_press

    # TODO
    # def cap_press_deriv(
    #     self,
    #     saturation_w: pp.ad.Operator,
    #     cap_press_constants: CapPressConstants | None = None,
    #     **kwargs,
    # ) -> pp.ad.Operator:
    #     cap_press: pp.ad.Operator = super().cap_press_deriv(
    #         saturation_w, cap_press_constants, **kwargs
    #     )
    #     cap_press_deriv: pp.ad.Operator = super().cap_press_deriv(
    #         saturation_w, cap_press_constants, **kwargs
    #     )
    #     cap_press_deriv =

    def cap_press_deriv_np(
        self,
        saturation_w: np.ndarray,
        cap_press_constants: CapPressConstants | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        cap_press: np.ndarray = super().cap_press_np(
            saturation_w, cap_press_constants, **kwargs
        )
        cap_press_deriv: np.ndarray = super().cap_press_deriv_np(
            saturation_w, cap_press_constants, **kwargs
        )
        # Limit the capillary pressure to a maximum value.
        cap_press_deriv = np.where(
            cap_press > self.spe11_params["CAP_PRESS_LIMIT"], 0.0, cap_press_deriv
        )
        return cap_press_deriv


class EquationsSPE11(SPE11Protocol, TPFProtocol):
    """Mixin class to provide the SPE11 model inspired equations and data.

    Takes care of:
    Updates the two-phase flow equations to include:
    - the SPE11 porosity field.
    - the SPE11 permeability field.
    - A volumetric source term for the water phase in the center cell.
    - Production wells in the corner cells.

    """

    def permeability(self, g: pp.Grid) -> dict[str, np.ndarray]:
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

    def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:  # type: ignore
        r"""Volumetric phase source term. Given as volumetric flux.

        Two injection wells are placed. One in the upper section of the domain and one
        in the lower section.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        if phase.name == self.nonwetting.name:
            array: np.ndarray = super().phase_fluid_source(g, phase)
            # Upper well.
            array[
                cell_id_position(
                    g,
                    self.spe11_params["WELL_1_POS"][0],
                    self.spe11_params["WELL_1_POS"][1],
                    percentages=False,
                )
            ] = phase.convert_units(
                self.spe11_params["INJECTION_RATE"], "m^3"
            ) / phase.convert_units(pp.SECOND, "s")
            # Lower well.
            array[
                cell_id_position(
                    g,
                    self.spe11_params["WELL_2_POS"][0],
                    self.spe11_params["WELL_2_POS"][1],
                    percentages=False,
                )
            ] = phase.convert_units(
                self.spe11_params["INJECTION_RATE"], "m^3"
            ) / phase.convert_units(pp.SECOND, "s")
            return array
        elif phase.name == self.wetting.name:
            return super().phase_fluid_source(g, phase)


class ModifiedBoundarySPE11(SPE11Protocol, TPFProtocol):
    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann).

        Dirichlet bc at the left and right. No flow Neumann bc at the bottom and top.

        """
        domain_sides = self.domain_boundary_sides(g)
        # return pp.BoundaryCondition(
        #     g, faces=np.logical_or(domain_sides.west, domain_sides.east), cond="dir"
        # )
        return pp.BoundaryCondition(g, faces=domain_sides.north, cond="dir")

    def _bc_dirichlet_pressure_values(
        self, g: pp.Grid, phase: FluidPhase
    ) -> np.ndarray:
        """Dirichle pressure values."""
        if phase == self.nonwetting:
            domain_sides = self.domain_boundary_sides(g)
            # bc: np.ndarray = np.zeros(g.num_faces)
            # bc[np.logical_or(domain_sides.west, domain_sides.east)] = (
            #     phase.convert_units(ATM, "kg*m^-1*s^-2")
            # )
            bc: np.ndarray = np.zeros(g.num_faces)
            bc[domain_sides.north] = phase.convert_units(ATM, "kg*m^-1*s^-2")
            return bc
        else:
            raise NotImplementedError(
                "Dirichlet pressure values not implemented for the wetting phase."
            )

    def _bc_dirichlet_saturation_values(
        self, g: pp.Grid, phase: FluidPhase
    ) -> np.ndarray:
        if phase.name == self.wetting.name:
            s_bc: np.ndarray = np.full(
                g.num_faces, self.spe11_params["INITIAL_SATURATION"]
            )
        elif phase.name == self.nonwetting.name:
            s_bc = np.ones(g.num_faces) - self._bc_dirichlet_saturation_values(
                g, self.wetting
            )
        return s_bc


class ModelGeometrySPE11(SPE11Protocol, TPFProtocol):
    def set_domain(self) -> None:
        """Set domain of the problem."""
        box: dict[str, pp.number] = {
            "xmax": self.spe11_params["WIDTH"],
            "ymax": self.spe11_params["HEIGHT"],
        }
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        pass

    def set_geometry(self) -> None:
        self.set_domain()
        self.mdg = load_spe11_data(
            DATA_DIR,
            self.params["meshing_arguments"].get("spe11_refinement_factor", 10.0),
        )
        self.nd: int = self.mdg.dim_max()
        g: pp.Grid = self.mdg.subdomains()[0]

        # Check that the domain size is correct.
        height: float = np.max(g.nodes[1, :]) - np.min(g.nodes[1, :])
        width: float = np.max(g.nodes[0, :]) - np.min(g.nodes[0, :])
        assert np.isclose(height, self.spe11_params["HEIGHT"])
        assert np.isclose(width, self.spe11_params["WIDTH"])


class SolutionStrategySPE11(TPFProtocol):
    """Mixin class to provide the SPE11 model data.

    Takes care of:
    - Loading the SPE11 fluids, i.e., CO2 and water.
    - Loading the SPE11 geometry, i.e., permeability and porosity.
    - Exporting the SPE11 geometry.

    """

    def __init__(self, params: dict | None) -> None:
        super().__init__(params)  # type: ignore
        self.spe11_case: str = self.params.get("spe11_case", "A")
        if self.spe11_case == "A":
            self.spe11_params = case_A
        elif self.spe11_case == "B":
            self.spe11_params = case_B
        else:
            raise ValueError(
                f"Unknown SPE11 case {self.spe11_case}. "
                + "Please choose either 'A' or 'B'."
            )

    def set_phases(self) -> None:
        self.phases: dict[str, FluidPhase] = {}
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

    # For the next two methods, ignore type errors due do unknown methods/attributes and
    # the wrong type for self.permeability.
    def prepare_simulation(self) -> None:
        self.set_materials()  # type: ignore
        self.set_geometry()
        super().prepare_simulation()  # type: ignore
        self.add_constant_spe11_data()

    def add_constant_spe11_data(self) -> None:
        """Save the SPE11 data to the exporter."""
        data: list[DataInput] = []
        for dim, perm in self.permeability(self.g).items():  # type: ignore
            data.append((self.g, "permeability_" + dim, perm))
        data.append((self.g, "porosity", self.porosity(self.g)))
        data.append((self.g, "entry_pressure", self.entry_pressure_np(self.g)))  # type: ignore
        self.exporter.add_constant_data(data)

        # For convenience, add the porosity and permeability to the iteration exporter
        # if it exists.
        if hasattr(self, "iteration_exporter"):
            self.iteration_exporter.add_constant_data(data)  # type: ignore


# The various protocols define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SPE11Mixin(
    CapillaryPressureSPE11,
    EquationsSPE11,
    ModifiedBoundarySPE11,
    SolutionStrategySPE11,
    ModelGeometrySPE11,
): ...  # type: ignore
