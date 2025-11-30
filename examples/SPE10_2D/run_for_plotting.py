r"""Run the simulation with small uniform time steps for fancy plotting.

We loosely follow the setup of Wang and Tchelepi (2013). The considered model is similar
to the heterogeneous 3D models in the article (section 4.6.4), but on a 2D domain.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 1200x2200 ft domain
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 10 days
- Solid properties:
    - Porosity: Layers 10 and 80 of SPE10, case 2A.
    - Permeability: Layers 10 and 80 of SPE10, case 2A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Saturation: 0.3.
- Rel. perm. model:
    - Brooks-Corey-Mualem
- Capillary pressure model:
    - Brooks-Corey

"""

import logging
import pathlib
import shutil
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import porepy as pp
from tpf.derived_models.spe10 import INITIAL_PRESSURE, SPE10Mixin
from tpf.models.flow_and_transport import TwoPhaseFlow
from tpf.models.protocol import TPFProtocol
from tpf.utils.constants_and_typing import FEET
from tpf.viz.solver_statistics import SolverStatisticsTPF

# region SETUP


# Setup logging and warnigns
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("default")

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

# endregion


# region MODEL
class InitialConditionsMixin(TPFProtocol):
    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        initial_pressure = np.full(self.g.num_cells, INITIAL_PRESSURE)
        initial_saturation = np.full(
            self.g.num_cells, self.params["spe10_initial_saturation"]
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


class SPE10Newton(
    InitialConditionsMixin,
    SPE10Mixin,
    TwoPhaseFlow,
):  # type: ignore
    ...


# endregion

# region UTILS

default_params: dict[str, Any] = {
    "progressbars": True,
    # Model:
    "material_constants": {},
    "rel_perm_constants": {
        "model": "Brooks-Corey-Mualem",
        "n_b": 4.0,
        "eta": 2.0,
        "limit": True,
    },
    "cap_press_constants": {
        "model": "Brooks-Corey",
        "n_b": 4.0,
        "entry_pressure": 50 * pp.PASCAL,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
    "grid_type": "simplex",
    "meshing_arguments": {"cell_size": 600 * FEET / 30},
    "spe10_quarter_domain": False,
    "spe10_isotropic_perm": True,
    "spe10_initial_saturation": 0.3,
    # NewtonAppleyard solver params:
    "nonlinear_solver_statistics": SolverStatisticsTPF,
    "nonlinear_solver": pp.NewtonSolver,
    "nl_adaptive": False,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e30,
    "max_iterations": 50,
    "nl_appleyard_chopping": True,
    "nl_enforce_physical_saturation": True,
}

time_manager_params: dict[str, Any] = {
    "schedule": np.array([0.0, 30.0 * pp.DAY]),
    "dt_init": 0.05 * pp.DAY,
    "constant_dt": True,
}


@dataclass(unsafe_hash=True)
class SimulationConfig:
    """Class to store all the simulation parameters that can vary."""

    folder_name: pathlib.Path
    file_name: str
    spe10_layer: int


def run_simulation(config: SimulationConfig) -> None:
    """Run simulation for a single configuration."""

    # Build params dictionary
    params = default_params.copy()

    params.update(
        {
            "spe10_layer": config.spe10_layer,
            "folder_name": config.folder_name,
            "file_name": config.file_name,
            "solver_statistics_file_name": config.folder_name
            / "solver_statistics.json",
            "time_manager": pp.TimeManager(**time_manager_params),
        }
    )

    try:
        shutil.rmtree(config.folder_name)
        config.folder_name.mkdir(parents=True)
    except Exception:
        pass

    model = SPE10Newton(params)
    pp.run_time_dependent_model(model=model, params=params)


# endregion

# region RUN

if __name__ == "__main__":
    for spe10_layer in [10, 55]:
        config = SimulationConfig(
            file_name=f"plotting_layer_{spe10_layer}",
            folder_name=dirname / f"plotting_layer_{spe10_layer}",
            spe10_layer=spe10_layer,
        )
        run_simulation(config)

# endregion
