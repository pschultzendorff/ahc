r"""Study how the estimators evolve during an adaptive homotopy continuation loop. The
 simulation is run for 3 time steps to study the behavior during each time step.
 
We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
continuation. The considered model is similar to the heterogeneous 3D models in the
article (section 4.6.4), but on a 2D domain for now.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 600x1100 ft domain (we just take a quarter of the original SPE10 domain)
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 10 days
- Solid properties:
    - Porosity: Uppermost layer of the SPE10, case 2A.
    - Permeability: Uppermost layer of the SPE10, case 2A.
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 6000 psi
    - Saturation: 0.3.
- Rel. perm. models:
    - linear
    - Corey with power 2.
- Capillary pressure model:
    - Brooks-Corey
- Time step size is kept constant s.t. the discretization error varies only with grid
  size.

"""

import itertools
import json
import logging
import pathlib
import shutil
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from tpf.derived_models.spe10 import SPE10Mixin
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.utils.constants_and_typing import FEET, PSI
from tpf.viz.iteration_exporting import IterationExportingMixin
from tpf.viz.solver_statistics import SolverStatisticsHC

# region SETUP

# Disable numba JIT for debugging.
# config.DISABLE_JIT = False

# Catch numpy warnings.
np.seterr(all="raise")
warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion

# region MODEL


class SPE10HC(
    IterationExportingMixin,
    SPE10Mixin,
    TwoPhaseFlowAHC,
): ...  # type: ignore


# endregion

# region RUN
spe10_layer: int = 80

params = {
    "progressbars": True,
    # Model:
    "formulation": "fractional_flow",
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    "spe10_layer": spe10_layer,
    "spe10_isotropic_perm": True,
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
    "hc_error_ratio": 0.1,
    "nl_error_ratio": 0.1,
    # Nonlinear params:
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e30,
}

cell_sizes: list[float] = [600 * FEET / 30, 600 * FEET / 60]
rel_perm_constants_list_1: list[dict[str, Any]] = [
    {
        "model": "linear",
        "limit": False,
    }
]
rel_perm_constants_list_2: list[dict[str, Any]] = [
    {
        "model": "Brooks-Corey",
        "limit": False,
        "n1": 2,
        "n2": 2,  # 1 + 2/n_b
        "n3": 1,
    },
]

cap_press_constants_list: list[dict[str, Any]] = [
    {"model": None},
    {"model": "linear", "linear_param": 5 * PSI},
]


for i, (cell_size, rp_model_1, rp_model_2, cp_model) in enumerate(
    itertools.product(
        cell_sizes,
        rel_perm_constants_list_1,
        rel_perm_constants_list_2,
        cap_press_constants_list,
    )
):
    logger.info(
        f"Run {i + 1} of {len(cell_sizes) *len(rel_perm_constants_list_1) *  len(rel_perm_constants_list_2)* len(cap_press_constants_list)}"
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, RP model 1: {rp_model_1['model']}, RP model 2: {rp_model_2['model']}, CP model: {cp_model['model']}"
    )

    # We have the file name both in the folder name and the filename to make
    # distinguishing different runs in ParaView easier.
    filename: str = (
        f"cellsz_{int(cell_size)}_rp1_{rp_model_1['model']} rp2_{rp_model_2['model']}_cp_{cp_model['model']}"
    )
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent / "ahc_3_ts" / f"lay_{spe10_layer}" / filename
    )

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    params.update(
        {
            # Reinitialize the time manager for each run.
            "time_manager": pp.TimeManager(
                schedule=np.array([0, 3.0 * pp.DAY]),
                dt_init=1.0 * pp.DAY,
                constant_dt=True,
            ),
            "folder_name": foldername,
            "file_name": filename,
            "solver_statistics_file_name": foldername / "solver_statistics.json",
            "meshing_arguments": {"cell_size": cell_size},
            "rel_perm_constants": {"model_1": rp_model_1, "model_2": rp_model_2},
            "cap_press_constants": cp_model,
        }
    )

    model = SPE10HC(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion

# region PLOTTING
for i, (rp_model_1, rp_model_2, cp_model) in enumerate(
    itertools.product(
        rel_perm_constants_list_1, rel_perm_constants_list_2, cap_press_constants_list
    )
):
    continue
    filename: str = (
        f"rp1_{rp_model_1['model']} rp2_{rp_model_2['model']}_cp_{cp_model['model']}"
    )
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "adaptive homotopy continuation_3_ts"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}"
        / filename
    )
    solver_statistics_file: pathlib.Path = foldername / "solver_statistics.json"
    with open(solver_statistics_file) as f:
        history: dict[str, Any] = json.load(f)
        history_list: list[Any] = list(history.values())[1:]

    fig, ax = plt.subplots()

    discretization_est: list[float] = []
    hc_est: list[float] = []
    linearization_est: list[float] = []
    for n, time_step in enumerate(history_list):
        for nl_step in list(time_step.values()):
            if isinstance(nl_step, int) or isinstance(nl_step, float):
                continue
            discretization_est.extend(nl_step["discretization_error_estimates"])
            hc_est.extend(nl_step["hc_error_estimates"])
            linearization_est.extend(nl_step["linearization_error_estimates"])

        # Draw a vertical line when the time increase in the next time step, i.e., the
        # nonlinear iterations at the current time step converged.
        time: float = time_step["current time"]
        try:
            next_time: float = history_list[n + 1]["current time"]
            if time < next_time:
                ax.axvline(x=len(discretization_est), color="gray", linestyle="--")
        except IndexError:
            pass

    ax.semilogy(discretization_est[:100], label="Discretization estimator")
    ax.semilogy(hc_est[:100], label="HC estimator")
    ax.semilogy(linearization_est[:100], label="Linearization estimator")
    ax.set_xlabel("Nonlinear iteration")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Estimator values")
    ax.legend()
    plt.show()
    fig.savefig(foldername / "solver_convergence.png")

    # fig, ax = plt.subplots()
    # ax.semilogy(discretization_est[:100], label="Discretization estimator")
    # ax.semilogy(hc_est[:100], label="HC estimator")
    # ax.semilogy(linearization_est[:100], label="Linearization estimator")
    # # ax.set_ylim([5e-5, 1e5])
    # ax.set_xlabel("Nonlinear iteration")
    # ax.set_ylabel("Estimator")
    # ax.set_title(f"Estimator values")
    # ax.legend()
    # plt.show()
    # fig.savefig(foldername / "solver_convergence_zoomed.png")

# endregion
