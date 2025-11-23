"""Analyze convergence etc. with the default Corey relative permeability model.

The function is analyzed on the 2D test case in section 3.1. of "Unconditionally
convergent nonlinear solver for hyperbolic conservation laws with S-shaped flux
functions" [Jenny et. al, 2009].

"""

import logging
import pathlib
import random
import shutil
import warnings

import numpy as np
import porepy as pp
from numba import config
from tpf.models.flow_and_transport import (
    BoundaryConditionsTPF,
    EquationsTPF,
    TwoPhaseFlow,
)
from tpf.models.phase import Phase, PhaseConstants
from tpf.models.reconstruction import (
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
    SolutionStrategyReconstructions,
)

# Disable numba JIT for debugging.
config.DISABLE_JIT = True

# Catch numpy warnings.
np.seterr(all="raise")
warnings.filterwarnings("default")

# Fix seed for reproducability.
random.seed(0)
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ModifiedGeometry(pp.ModelGeometry):
    def set_domain(self) -> None:
        r"""Set domain of the problem.

        The domain is :math:`\Omega=[0,120]\times[0,60]`.

        """
        bounding_box: dict[str, pp.number] = {
            "xmin": 0,
            "xmax": 20,
            "ymin": 0,
            "ymax": 20,
            # "xmin": 0,
            # "xmax": 2,
            # "ymin": 0,
            # "ymax": 2,
        }
        self._domain = pp.Domain(bounding_box)


class ModifiedEquations(EquationsTPF):
    r"""Modifications to the two-phase flow model:

    - Source term in the subdomain :math:`\Omega_{in} = [0,1] \times [0,1]`; sink term
      in the subdomain :math:`\Omega_{out} = [119,120] \times [59,60]`.
    - Varying permeability across the domain.

    """

    # def _permeability(self, g: pp.Grid) -> np.ndarray:
    #     # Function for base-10 log. of permeability along x-axis.
    #     def function_x(x: float) -> float:
    #         return 3 * x * (x - 0.7) * (x - 2.3)

    #     # Function for base-10 log. of permeability along y-axis.
    #     def function_y(x: float) -> float:
    #         return 5 * (x - 0.2) * (x - 0.8) * (x + 2)

    #     log_permeability = np.zeros([60, 120], dtype=float)
    #     # log_permeability = np.zeros([2, 2], dtype=float)
    #     for i, row in enumerate(log_permeability):
    #         for j, _ in enumerate(row):
    #             log_permeability[i, j] = function_x(i / 60.0) * function_y(j / 120.0)
    #             # log_permeability[i, j] = function_x(i / 2.0) * function_y(j / 2.0)

    #     # Add noise.
    #     log_permeability += np.random.normal(0, 0.2, [60, 120])
    #     # log_permeability += np.random.normal(0, 0.2, [2, 2])
    #     return (10**log_permeability).flatten()

    def phase_fluid_source(self, g: pp.Grid, phase: Phase) -> np.ndarray:
        """Volumetric phase source term. Given as volumetric flux. This
        unmodified base function assumes a zero phase source.

        NOTE: This is the average value per grid cell, i.e., it gets scaled with the
        cell volume in the equation.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        # TODO: Does this has to be a vector instead?
        if phase.name == "wetting":
            array = super().phase_fluid_source(g, phase)
            array[0] = 0.5
            array[119] = -0.5
            return array
        elif phase.name == "nonwetting":
            array = super().phase_fluid_source(g, phase)
            # array[19] = 3
            return array


class ModifiedBoundaryConditions(BoundaryConditionsTPF):
    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann)."""
        # Dirichlet conditions for both phases.
        north_faces = self.domain_boundary_sides(g).north
        return pp.BoundaryCondition(g, north_faces, "dir")


# class ModifiedTwoPhaseFlow(ModifiedEquations, ModifiedGeometry, TwoPhaseFlow): ...  # type: ignore
class ModifiedTwoPhaseFlow(ModifiedEquations, ModifiedGeometry, TwoPhaseFlow): ...  # type: ignore


# Set up folder and files for logging/plots/saved time steps.
foldername: pathlib.Path = (
    pathlib.Path(__file__).parent.resolve()
    / "results"
    / "brooks-corey_no_pc_simplex_test"
)

try:
    shutil.rmtree(foldername)
    foldername.mkdir(parents=True)
except Exception:
    pass

solid_constants: pp.SolidConstants = pp.SolidConstants(
    {
        "porosity": 1,
        "permeability": 1.0,
    }
)
wetting_constants: PhaseConstants = PhaseConstants(
    {
        "density": 600.0,
        "viscosity": 2.0,
        "residual_saturation": 0.1,
    }
)
nonwetting_constants: PhaseConstants = PhaseConstants(
    {
        "density": 800.0,
        "viscosity": 1.0,
        "residual_saturation": 0.1,
    }
)

params = {
    # Base folder and file name. These will get changed by
    # ``ConvergenceAnalysisExtended``.
    "folder_name": foldername,
    "file_name": "setup",
    "max_iterations": 60,
    "nl_convergence_tol": 1e-5,
    "progressbars": True,
    # grid and time
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 1.0},
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 20]),
        dt_init=0.02,
        constant_dt=True,
    ),
    "material_constants": {
        "solid": solid_constants,
        "wetting": wetting_constants,
        "nonwetting": nonwetting_constants,
    },
    # Brooks-Corey-Burdine
    "rel_perm_constants": {
        "model": "Brooks-Corey",
        "limit": False,
        "n1": 2,
        "n2": 1 + 2 / 2,  # 1 + 2 / n_b
        "n3": 1,
    },
    "cap_press_constants": {"model": "Brooks-Corey", "entry_pressure": 1e-2, "n_b": 2},
}

logger.info("start")
model = ModifiedTwoPhaseFlow(params)
pp.run_time_dependent_model(model=model, params=params)
