r"""Study convergence of adaptive homotopy continuation on different grid sizes and with
 different rel. perm./cap. pressure models.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton.

We loosely follow the setup of ...


Model description:
- Constant CO2 injection in the center.
- No flow boundary condition on the sides and bottom. Homogeneous Dirichlet on top.
- Simulation time: 10 days
- Solid properties:
    - Porosity: SPE11, case A.
    - Permeability: SPE11, case A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.1.
    - CO2: From the NIST database, taken at 20°C and atmospheric pressure. Residual
      saturation is 0.1.
- Initial values:
    - Pressure: Atmospheric pressure.
    - Saturation: Varying between 0.8 and 0.9.
- Rel. perm. models:
    - linear
    - Brooks-Corey
- Capillary pressure model:
    - None

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
from tpf.derived_models.spe11 import INITIAL_PRESSURE, SPE11Mixin
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.utils.constants_and_typing import PSI
from tpf.viz.solver_statistics import SolverStatisticsHC

# region SETUP

# Disable numba JIT for debugging.
# config.DISABLE_JIT = False

# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow. The latter can appear during estimator
# calculation.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion


# region MODEL
class SPE11HC(
    SPE11Mixin,
    TwoPhaseFlowAHC,
):  # type: ignore

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        initial_pressure = np.full(self.g.num_cells, INITIAL_PRESSURE)
        initial_saturation = np.full(
            self.g.num_cells, self.params["spe11_initial_saturation"]
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
    "formulation": "fractional_flow",
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
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
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e30,
    "max_iterations": 20,
}

adaptive_error_ratios: list[float] = [0.1, 0.01]
refinement_factors: list[float] = [20.0, 10.0, 5.0]
refinement_factors: list[float] = [20, 10, 5]
rel_perm_constants_list_1: list[dict[str, Any]] = [{"model": "linear", "limit": False}]
rel_perm_constants_list_2: list[dict[str, Any]] = [
    {
        "model": "Brooks-Corey",
        "limit": False,
        "n1": 2,
        "n2": 2,  # 1 + 2/n_b
        "n3": 1,
    },
    {"model": "Corey", "limit": False, "power": 2},
    {"model": "Corey", "limit": False, "power": 3},
    # {"model": "van Genuchten-Burdine", "limit": False, "n_g": -1.0},
]
cap_press_constants_list: list[dict[str, Any]] = [
    {"model": None},
    {"model": "linear", "linear_param": 30 * PSI},
]
# NOTE Appleyard chopping with AHC is not fixed yet.
appleyard_chopping_list: list[bool] = [False, True]
initial_saturation_list: np.ndarray = np.linspace(0.8, 0.9, 5)

# region VARYING_CELL_SIZES
for i, (
    adaptive_error_ratio,
    initial_saturation,
    refinement_factor,
    rp_model_1,
    rp_model_2,
    cp_model,
) in enumerate(
    itertools.product(
        adaptive_error_ratios,
        initial_saturation_list[[0, -1]],
        refinement_factors,
        rel_perm_constants_list_1,
        rel_perm_constants_list_2[0:1],
        cap_press_constants_list[0:1],
    )
):
    logger.info(f"Varying cell sizes. Run {i + 1} of {len(refinement_factors)}")
    logger.info(
        f"Refinement factor: {refinement_factor}, initial saturation: {initial_saturation}"
        + f"RP model 1: {rp_model_1['model']}, RP model 2: {rp_model_2['model']}, CP model: {cp_model['model']}."
    )
    filename: str = f"ref_fac_{int(refinement_factor)}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / f"ahc_{adaptive_error_ratio}"
        / "varying_cell_sizes"
        / f"rp1_{rp_model_1['model']}_rp2_{rp_model_2['model']}_cp._{cp_model['model']}_init_s_{initial_saturation}"
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
                iter_optimal_range=(9, 12),
                iter_relax_factors=(0.7, 1.3),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "spe11_initial_saturation": initial_saturation,
            "meshing_arguments": {"spe11_refinement_factor": refinement_factor},
            "rel_perm_constants": {"model_1": rp_model_1, "model_2": rp_model_2},
            "cap_press_constants": cp_model,
            "hc_error_ratio": adaptive_error_ratio,
        }
    )
    model = SPE11HC(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")
        raise error

# endregion
# endregion
