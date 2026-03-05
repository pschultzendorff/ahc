r"""Homogenous five-spot example. Study convergence of adaptive homotopy continuation
 (AHC) on different grid sizes.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton.


Model description:
- 1200x2200 ft domain
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 30 days
- Solid properties:
    - Porosity: Homogeneous 0.3.
    - Permeability: Homogenous; 1e-15 m^2.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
- Initial values:
    - Pressure: 6000 psi (initial guess for Newton, no influence on the solution)
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
from tpf.models.flow_and_transport import EquationsTPF
from tpf.models.homotopy_continuation import TwoPhaseFlowHC
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.utils.constants_and_typing import FEET
from tpf.viz.solver_statistics import SolverStatisticsHC

# region SETUP


# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow. The latter can appear during estimator calculation.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion


# region MODEL
class HomogeneousSPE10HC(SPE10Mixin, TwoPhaseFlowHC):  # type: ignore
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
        "model_1": {"model": "linear"},
        "model_2": {
            "model": "Brooks-Corey",
            "limit": False,
            "n1": 2,
            "n2": 2,  # 1 + 2/n_b
            "n3": 1,
        },
    },
    "cap_press_constants": {"model": None},
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    # HC params:
    "nonlinear_solver_statistics": SolverStatisticsHC,
    "nonlinear_solver": HCSolver,
    "hc_max_iterations": 20,
    "hc_adaptive": True,
    # HC decay parameters.
    "hc_constant_decay": False,
    "hc_lambda_decay": 0.9,
    "hc_decay_min_max": (0.1, 0.95),
    "nl_iter_optimal_range": (4, 7),
    "nl_iter_relax_factors": (0.7, 1.3),
    "hc_decay_recomp_max": 5,
    # Adaptivity parameters.
    "nl_error_ratio": 0.1,
    "hc_nl_convergence_tol": 1e3,
    # Nonlinear params:
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e15,
}

adaptive_error_ratios: list[float] = [0.1, 0.01]
cell_sizes: list[float] = [600 * FEET / 60]
initial_saturation_list: np.ndarray = np.linspace(0.2, 0.3, 5)


def run_simulation(
    adaptive_error_ratio: float, initial_saturation: float, cell_size: float
) -> None:
    logger.info(f"Cell size: {cell_size}, initial saturation: {initial_saturation}")
    filename = f"cellsz_{int(cell_size)}"
    folder = (
        pathlib.Path(__file__).parent
        / f"ahc_{adaptive_error_ratio}"
        / "varying_cell_sizes"
        / f"init_s_{initial_saturation}"
        / filename
    )
    try:
        shutil.rmtree(folder)
    except Exception:
        pass
    folder.mkdir(parents=True)

    local_params = params.copy()
    local_params.update(
        {
            "folder_name": folder,
            "file_name": filename,
            "solver_statistics_file_name": folder / "solver_statistics.json",
            "time_manager": pp.TimeManager(
                schedule=np.array([0.0, 10.0 * pp.DAY]),
                dt_init=10.0 * pp.DAY,
                constant_dt=False,
                dt_min_max=(1e-3 * pp.DAY, 10.0 * pp.DAY),
                iter_optimal_range=(9, 12),
                iter_relax_factors=(0.7, 1.3),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "spe10_initial_saturation": initial_saturation,
            "meshing_arguments": {"cell_size": cell_size},
            "hc_error_ratio": adaptive_error_ratio,
        }
    )

    model = HomogeneousSPE10HC(local_params)
    try:
        pp.run_time_dependent_model(model=model, params=local_params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")
        raise error


# VARYING_CELL_SIZES
combos = itertools.product(
    adaptive_error_ratios, initial_saturation_list[[0, -1]], cell_sizes
)
for i, (adapt_err, init_sat, cell_sz) in enumerate(combos):
    logger.info(f"Varying cell sizes. Run {i + 1} of {len(cell_sizes)}")
    run_simulation(adapt_err, init_sat, cell_sz)
# endregion
