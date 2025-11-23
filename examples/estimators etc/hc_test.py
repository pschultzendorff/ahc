"""Analyze convergence etc. with the default Corey relative permeability model.

The function is analyzed on the 2D test case in section 3.1. of "Unconditionally
convergent nonlinear solver for hyperbolic conservation laws with S-shaped flux
functions" [Jenny et. al, 2009].

"""

import itertools
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
from tpf.models.homotopy_continuation import (
    HCSolver,
    RelativePermeabilityHC,
    SolutionStrategyHC,
)
from tpf.models.phase import Phase, PhaseConstants

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

        The domain is :math:`\Omega=[0,40]\times[0,40]`.

        """
        bounding_box: dict[str, pp.number] = {
            "xmin": 0,
            "xmax": 40,
            "ymin": 0,
            "ymax": 40,
        }
        self._domain = pp.Domain(bounding_box)


class ModifiedEquations(EquationsTPF):
    r"""Modifications to the two-phase flow model:

    - Source term in the subdomain :math:`\Omega_{in} = [0,1] \times [0,1]`; sink term
      in the subdomain :math:`\Omega_{out} = [119,120] \times [59,60]`.

    """

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
            return array
        elif phase.name == "nonwetting":
            array = super().phase_fluid_source(g, phase)
            array[39] = 0.1
            return array


class ModifiedBoundaryConditions(BoundaryConditionsTPF):
    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann)."""
        # Dirichlet conditions for both phases.
        north_faces = self.domain_boundary_sides(g).north
        return pp.BoundaryCondition(g, north_faces, "dir")


class HCTwoPhaseFlow(
    RelativePermeabilityHC,
    ModifiedEquations,
    ModifiedGeometry,
    ModifiedBoundaryConditions,
    SolutionStrategyHC,
    TwoPhaseFlow,
): ...  # type: ignore


solid_constants: pp.SolidConstants = pp.SolidConstants(
    {
        "porosity": 0.1,
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
    "file_name": "setup",
    "progressbars": True,
    # HC params:
    "nonlinear_solver": HCSolver,
    "hc_max_iterations": 60,
    "hc_decay": 0.8,
    # Nonlinear params:
    "max_iterations": 60,
    "nl_convergence_tol": 1e-10,
    # Grid and time discretization:
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 1.0},
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 20]),
        dt_init=0.2,
        constant_dt=True,
    ),
    # Model:
    "material_constants": {
        "solid": solid_constants,
        "wetting": wetting_constants,
        "nonwetting": nonwetting_constants,
    },
    "rel_perm_constants": {},
    "cap_press_constants": {"model": None, "entry_pressure": 1e-2, "n_b": 2},
}

rel_perm_constants_list = [
    {
        "model": "linear",
        "limit": False,
        "linear_param_w": 1,
        "linear_param_n": 1,
    },
    {
        "model": "Corey",
        "limit": False,
        "power": 3,
        "linear_param_w": 1,
        "linear_param_n": 1,
    },
    # {
    #     "model": "van Genuchten-Mualem",
    #     "limit": False,
    #     "kappa_g": 1,
    #     "n_g": 2,
    # },
    # {
    #     "model": "van Genuchten-Burdine",
    #     "limit": False,
    #     "kappa_g": 1,
    #     "n_g": 2,
    # },
    # {
    #     "model": "Brooks-Corey",
    #     "limit": False,
    #     "n1": 2,
    #     "n2": 1 + 2 / 2,
    #     "n3": 1,
    # },
]


for model_1, model_2 in itertools.product(
    rel_perm_constants_list, rel_perm_constants_list
):
    if model_1["model"] == model_2["model"]:
        continue
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "results"
        / "HC"
        / f"{model_1['model']} to {model_2['model']}"
    )
    try:
        shutil.rmtree(foldername)
        foldername.mkdir(parents=True)
    except Exception:
        pass

    params["folder_name"] = foldername
    params["rel_perm_constants"]["model_1"] = model_1
    params["rel_perm_constants"]["model_2"] = model_2
    model = HCTwoPhaseFlow(params)
    pp.run_time_dependent_model(model=model, params=params)
