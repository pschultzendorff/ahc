r"""Study how the estimators evolve during a Newton loop. The simulation is run for 3
 time steps to study the behavior during each time step.


We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
continuation. The considered model is similar to the heterogeneous 3D models in the
article (section 4.6.4), but on a 2D domain for now.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 600x1100 ft domain (we just take a quarter of the original SPE 10th CSP domain)
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 10 days
- Solid properties:
    - Porosity: Uppermost layer of the SPE 10th CSP (model 2).
    - Permeability: Uppermost layer of the SPE 10th CSP (model 2).
- Fluid properties:
    - Water: pp.fluid_values.water. Residual saturation is 0.2.
    - Oil: PVT table from the SPE 10th CSP (model 2). We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 6000 psi
    - Saturation: residual water saturation (0.2).
- Rel. perm. models:
    - linear
    - Corey with power 2.
- Capillary pressure model:
    - Brooks-Corey

"""

import itertools
import json
import logging
import os
import pathlib
import random
import shutil
import warnings
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from numba import config
from tpf.models.error_estimate import TwoPhaseFlowErrorEstimate
from tpf.spe10.model import SPE10
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    FEET,
    GLOBAL_PRESSURE,
    PSI,
)
from tpf.viz.solver_statistics import SolverStatisticsEst

# region SETUP

# Disable numba JIT for debugging.
config.DISABLE_JIT = False

# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch numpy warnings.
np.seterr(all="raise")
warnings.filterwarnings("default")

# Fix seed for reproducability.
random.seed(0)
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion


# region MODEL


class SPE10Newton(
    # SPE10:
    SPE10,
    # Two phase flow with estimators:
    TwoPhaseFlowErrorEstimate,
): ...  # type: ignore


# endregion

# region RUN
spe10_layer: int = 80

params = {
    # Base folder and file name. These will get changed by
    # ``ConvergenceAnalysisExtended``.
    "file_name": "setup",
    "progressbars": True,
    # Model:
    "formulation": "fractional_flow",
    "material_constants": {
        "solid": pp.SolidConstants({"porosity": 0.3, "permeability": 1e-15}),
    },
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    "spe10_layer": spe10_layer - 1,
    "spe10_isotropic_perm": True,
    # Nonlinear params:
    "nonlinear_solver_statistics": SolverStatisticsEst,
    "nonlinear_solver": pp.NewtonSolver,
    # Nonlinear params:
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e15,
    # "nl_convergence_tol": 1e-10
    # * 10000
    # * PSI,  # Scale the nonlinear tolerance by pressure values
}

cell_sizes: list[float] = [600 * FEET / 30]
rel_perm_constants_list: list[dict[str, Any]] = [
    {"model": "linear", "limit": False},
    {
        "model": "Brooks-Corey",
        "limit": True,
        "n1": 2,
        "n2": 2,  # 1 + 2/n_b
        "n3": 1,
    },
]
cap_press_constants_list: list[dict[str, Any]] = [
    {"model": "linear", "entry_pressure": 5 * PSI},
]


for i, (cell_size, rp_model, cp_model) in enumerate(
    itertools.product(cell_sizes, rel_perm_constants_list, cap_press_constants_list)
):
    logger.info(
        f"Run {i + 1} of {len(cell_sizes) * len(rel_perm_constants_list) * len(cap_press_constants_list)}"
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, RP model: {rp_model['model']}, CP model: {cp_model['model']}"
    )

    # We have the file name both in the folder name and the filename to make
    # distinguishing different runs in ParaView easier.
    filename: str = f"rp_{rp_model['model']}_cp._{cp_model['model']}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "newton_adaptive_ts"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}"
        / filename
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
                schedule=np.array([0, 10 * pp.DAY]),  # 5 days
                dt_init=0.5 * pp.DAY,  # Initial time step size in days.
                constant_dt=False,
                dt_min_max=(1e-3 * pp.DAY, 1.5 * pp.DAY),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "folder_name": foldername,
            "file_name": filename,
            "solver_statistics_file_name": foldername / "solver_statistics.json",
            "meshing_arguments": {"cell_size": cell_size},
            "rel_perm_constants": rp_model,
            "cap_press_constants": cp_model,
        }
    )
    model = SPE10Newton(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion

# region PLOTTING

for i, (rp_model, cp_model) in enumerate(
    itertools.product(rel_perm_constants_list, cap_press_constants_list)
):
    filename: str = f"rp_{rp_model['model']}_cp._{cp_model['model']}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "newton_adaptive_ts"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}"
        / filename
    )
    solver_statistics_file: pathlib.Path = foldername / "solver_statistics.json"
    with open(solver_statistics_file) as f:
        history = json.load(f)
        history_list = list(history.values())[1:]

    residual_and_flux_est: list[float] = []
    glob_nonconformity_est: list[float] = []
    compl_nonconformity_est: list[float] = []
    fig, ax = plt.subplots()

    for n, time_step in enumerate(history_list):
        residual_and_flux_est.extend(time_step["residual_and_flux_est"])
        # Transform list of dicts to dict containing two lists.
        converted_nonconformity_est = defaultdict(list)
        for nonconformity in time_step["nonconformity_est"]:
            for key, value in nonconformity.items():
                converted_nonconformity_est[key].append(value)
        glob_nonconformity_est.extend(converted_nonconformity_est[GLOBAL_PRESSURE])
        compl_nonconformity_est.extend(
            converted_nonconformity_est[COMPLIMENTARY_PRESSURE]
        )

        # Draw a vertical line when the time increase in the next time step, i.e., the
        # nonlinear iterations at the current time step converged.
        time: float = time_step["current time"]
        try:
            next_time: float = history_list[n + 1]["current time"]
            if time < next_time:
                ax.axvline(x=len(residual_and_flux_est), color="gray", linestyle="--")
        except IndexError:
            pass

    ax.semilogy(residual_and_flux_est, label="Residual and flux estimator")
    ax.semilogy(glob_nonconformity_est, label="Global pressure nonconformity estimator")
    ax.semilogy(
        compl_nonconformity_est, label="Complimentary pressure nonconformity estimator"
    )
    ax.set_xlabel("Nonlinear iteration (over multiple time steps)")
    ax.set_ylabel("Estimator")
    ax.set_title(f"Discretization estimator")
    # ax.set_ylim([5e-4, 1e3])
    ax.legend()
    plt.show()
    fig.savefig(foldername / "solver_convergence.png")

# endregion
