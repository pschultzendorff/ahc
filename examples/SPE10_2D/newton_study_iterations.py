r"""Study how the estimators evolve during a Newton loop. The simulation is run for 10
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
import os
import pathlib
import shutil
import warnings
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import porepy as pp
from tpf.models.error_estimate import TwoPhaseFlowErrorEstimate
from tpf.spe10.model import SPE10Mixin
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    FEET,
    GLOBAL_PRESSURE,
    PSI,
)
from tpf.viz.iteration_exporting import IterationExportingMixin
from tpf.viz.solver_statistics import SolverStatisticsEst

# region SETUP

# Disable numba JIT for debugging.
# config.DISABLE_JIT = False
dirname: pathlib.Path = pathlib.Path(__file__).parent

# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch numpy warnings.
np.seterr(all="raise")
warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion


# region MODEL


class SPE10Newton(
    IterationExportingMixin,
    SPE10Mixin,
    TwoPhaseFlowErrorEstimate,
): ...  # type: ignore


# endregion

# region RUN
spe10_layer: int = 80

params: dict[str, Any] = {
    "progressbars": True,
    # Model:
    "formulation": "fractional_flow",
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    "spe10_layer": spe10_layer - 1,
    "spe10_isotropic_perm": True,
    # Nonlinear params:
    "nonlinear_solver_statistics": SolverStatisticsEst,
    "nonlinear_solver": pp.NewtonSolver,
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e30,
}

cell_sizes: list[float] = [600 * FEET / 30, 600 * FEET / 60]
rel_perm_constants_list: list[dict[str, Any]] = [
    {"model": "linear", "limit": False},
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


for i, (cell_size, rp_model, cp_model) in enumerate(
    itertools.product(cell_sizes, rel_perm_constants_list, cap_press_constants_list)
):
    logger.info(
        f"Run {i + 1} of {len(cell_sizes) * len(rel_perm_constants_list) * len(cap_press_constants_list)}"
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, RP model: {rp_model['model']}, CP model: {cp_model['model']}"
    )

    filename: str = (
        f"cellsz_{int(cell_size)}_rp_{rp_model['model']}_cp._{cp_model['model']}"
    )
    foldername: pathlib.Path = dirname / "newton_3_ts" / f"lay_{spe10_layer}" / filename

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    params.update(
        {
            # Reinitialize the time manager for each run.
            "time_manager": pp.TimeManager(
                schedule=np.array([0, 5.0 * pp.DAY]),
                dt_init=0.5 * pp.DAY,
                constant_dt=True,
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
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for i, (cell_size, rp_model, cp_model) in enumerate(
    itertools.product(cell_sizes, rel_perm_constants_list, cap_press_constants_list)
):

    filename = f"cellsz_{int(cell_size)}_rp_{rp_model['model']}_cp._{cp_model['model']}"
    foldername = dirname / "newton_3_ts" / f"lay_{spe10_layer}" / filename
    solver_statistics_file: pathlib.Path = foldername / "solver_statistics.json"

    with open(solver_statistics_file) as f:
        data = json.load(f)
        data_list = list(data.values())[1:]

    residual_and_flux_est: list[float] = []
    glob_nonconformity_est: list[float] = []
    compl_nonconformity_est: list[float] = []
    total_est: list[float] = []
    global_energy_norm: list[float] = []

    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    for n, time_step in enumerate(data_list):
        residual_and_flux_est.extend(time_step["residual_and_flux_est"])
        global_energy_norm.extend(time_step["global_energy_norm"])

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
            next_time: float = data_list[n + 1]["current time"]
            if time < next_time:
                ax3.axvline(x=len(residual_and_flux_est), color="gray", linestyle="--")
        except IndexError:
            pass

    ax1.semilogy(
        np.array(residual_and_flux_est)
        + np.array(glob_nonconformity_est)
        + np.array(compl_nonconformity_est),
        label=filename,
    )
    ax2.semilogy(
        (
            np.array(residual_and_flux_est)
            + np.array(glob_nonconformity_est)
            + np.array(compl_nonconformity_est)
        )
        / np.array(global_energy_norm),
        label=filename,
    )

    ax3.semilogy(
        residual_and_flux_est, label="Residual and flux estimator", color="blue"
    )
    ax3.semilogy(
        glob_nonconformity_est,
        label="Global pressure nonconformity estimator",
        color="orange",
    )
    ax3.semilogy(
        compl_nonconformity_est,
        label="Complimentary pressure nonconformity estimator",
        color="green",
    )
    ax3.set_xlabel("Nonlinear iteration (over multiple time steps)")
    ax3.set_ylabel("Estimator value")
    ax3.xaxis.set_major_locator(tck.MultipleLocator(base=5))
    ax3.set_title(f"Discretization estimators")
    ax3.legend()
    fig3.savefig(foldername / "estimators.png")

    ax4.semilogy(
        np.array(glob_nonconformity_est) / np.array(global_energy_norm),
        label="Relative global pressure nonconformity estimator",
        color="blue",
    )
    ax4.semilogy(
        np.array(residual_and_flux_est) / np.array(global_energy_norm),
        label="Relative residual and flux estimator",
        color="orange",
    )
    ax4.semilogy(
        np.array(compl_nonconformity_est) / np.array(global_energy_norm),
        label="Relative complimentary pressure nonconformity estimator",
        color="green",
    )
    ax4.set_xlabel("Nonlinear iteration (over multiple time steps)")
    ax4.set_ylabel("Relative estimator value")
    ax4.xaxis.set_major_locator(tck.MultipleLocator(base=5))
    ax4.set_title(f"Relative discretization estimators")
    ax4.legend()
    fig4.savefig(foldername / "relative_estimators.png")

ax1.set_xlabel("Nonlinear iteration")
ax1.set_ylabel("Estimator value")
ax1.set_ylim(top=1e3)
ax1.xaxis.set_major_locator(tck.MultipleLocator(base=5))
ax1.set_title("Total estimator")
ax1.legend()
fig1.savefig(dirname / "newton_3_ts" / f"lay_{spe10_layer}" / "total_estimator.png")

ax2.set_xlabel("Nonlinear iteration")
ax2.set_ylabel("Relative estimator value")
ax2.set_ylim(top=1e3)
ax2.xaxis.set_major_locator(tck.MultipleLocator(base=5))
ax2.set_title("Relative total estimator")
ax2.legend()
fig2.savefig(
    dirname / "newton_3_ts" / f"lay_{spe10_layer}" / "relative_total_estimator.png"
)
# endregion
