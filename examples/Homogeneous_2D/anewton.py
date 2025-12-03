r"""Homogenous five-spot example. Study convergence of Newton on different grid sizes.

The following solvers are employed:
- Adaptive Newton
- Newton with Appleyard chopping

Model description:
- 600x1100 ft domain (we just take a quarter of the original SPE10 domain)
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 10 days
- Solid properties:
    - Porosity: Homogeneous 0.3.
    - Permeability: Homogenous; 1e-15 m^2.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 6000 psi
    - Saturation: residual water saturation (0.2) or 0.3.
- Rel. perm. models:
    - linear
    - Corey with power 2.
- Capillary pressure model:
    - Brooks-Corey

"""

import itertools
import logging
import os
import pathlib
import shutil
import warnings
from typing import Any

import numpy as np
import porepy as pp
from tpf.derived_models.spe10 import INITIAL_PRESSURE, SPE10Mixin
from tpf.models.adaptive_newton import TwoPhaseFlowANewton
from tpf.models.flow_and_transport import EquationsTPF
from tpf.utils.constants_and_typing import FEET, PSI
from tpf.viz.iteration_exporting import IterationExportingMixin
from tpf.viz.solver_statistics import SolverStatisticsEst

# region SETUP


# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow, which may occur when calculating estimators.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion


# region MODEL
class HomogeneousSPE10Newton(
    SPE10Mixin,
    TwoPhaseFlowANewton,
):  # type: ignore
    """Override the heterogeneous geometry of the SPE10 model by using methods of
    ``EquationsTPF`` instead.

    """

    def permeability(self, g: pp.Grid) -> dict[str, np.ndarray]:
        """Homogeneous solid permeability. Units are set by
        :attr:`self.solid`."""
        return EquationsTPF.permeability(self, g)

    def porosity(self, g: pp.Grid) -> np.ndarray:
        """Homogeneous solid porosity. Chosen layer of the SPE10 model."""
        return EquationsTPF.porosity(self, g)

    def load_spe10_model(self, g: pp.Grid) -> None:
        pass

    def add_constant_spe10_data(self) -> None:
        pass

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


# endregion

# region RUN
params: dict[str, Any] = {
    "progressbars": True,
    # Model:
    "material_constants": {
        "solid": pp.SolidConstants({"porosity": 0.3, "permeability": 1e-15}),
    },
    "rel_perm_constants": {
        "model": "Brooks-Corey",
        "limit": False,
        "n1": 2,
        "n2": 2,  # 1 + 2/n_b
        "n3": 1,
    },
    "cap_press_constants": {"model": None},
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    # Nonlinear params:
    "nonlinear_solver_statistics": SolverStatisticsEst,
    "nonlinear_solver": pp.NewtonSolver,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e15,
}

adaptive_error_ratios: list[float] = [0.1, 0.01]
cell_sizes: list[float] = [
    # 600 * FEET / 7.5,
    # 600 * FEET / 15,
    # 600 * FEET / 30,
    600 * FEET / 60,
]
appleyard_chopping_list: list[bool] = [False, True]
initial_saturation_list: np.ndarray = np.linspace(0.2, 0.3, 5)

# region VARYING_CELL_SIZES
for i, (
    adaptive_error_ratio,
    appleyard_chopping,
    initial_saturation,
    cell_size,
) in enumerate(
    itertools.product(
        adaptive_error_ratios,
        appleyard_chopping_list,
        initial_saturation_list[[0, -1]],
        cell_sizes,
    )
):
    if appleyard_chopping:
        max_iterations: int = 50
        solver_name: str = f"Newton_Appleyard_{adaptive_error_ratio}"
    else:
        max_iterations = 20
        solver_name = f"Newton_{adaptive_error_ratio}"

    logger.info(f"Varying cell sizes. Run {i + 1} of {len(cell_sizes)}.")
    logger.info(f"Cell size: {cell_size}, initial saturation: {initial_saturation}")

    filename: str = f"cellsz_{int(cell_size)}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent.resolve()
        / solver_name
        / "varying_cell_sizes"
        / f"init_s_{initial_saturation}"
        / filename
    )

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    params.update(
        {
            "folder_name": foldername,
            "file_name": filename,
            "solver_statistics_file_name": foldername / "solver_statistics.json",
            # Reinitialize the time manager for each run.
            "time_manager": pp.TimeManager(
                schedule=np.array([0.0, 10.0 * pp.DAY]),
                dt_init=10.0 * pp.DAY,
                constant_dt=False,
                dt_min_max=(1e-3 * pp.DAY, 10.0 * pp.DAY),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "spe10_initial_saturation": initial_saturation,
            "meshing_arguments": {"cell_size": cell_size},
            "nl_adaptive": True,
            "nl_adaptive_convergence_tol": 1e3,
            "nl_error_ratio": adaptive_error_ratio,
            "nl_appleyard_chopping": appleyard_chopping,
            "nl_sat_increment_norm_scaling": 1.0,
            "nl_press_increment_norm_scaling": INITIAL_PRESSURE,
            "max_iterations": max_iterations,
        }
    )
    model = HomogeneousSPE10Newton(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion
# endregion
