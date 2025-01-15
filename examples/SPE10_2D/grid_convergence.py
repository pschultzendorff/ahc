r"""We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
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
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from numba import config
from tpf.models.error_estimate import (
    DataSavingEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
)
from tpf.models.flow_and_transport import EquationsTPF, TwoPhaseFlow
from tpf.models.phase import FluidPhase
from tpf.models.reconstruction import (
    DataSavingReconstruction,
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
)
from tpf.spe10.fluid_values import oil, water
from tpf.spe10.model import SPE10
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    FEET,
    GLOBAL_PRESSURE,
    PSI,
)
from tpf.viz.solver_statistics import SolverStatisticsEst, SolverStatisticsTPF

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


class ConvergenceAnalysisEstimatesSPE(
    # SPE10:
    SPE10,
    # Modified model:
    # Estimator mixins:
    ErrorEstimateMixin,
    SolutionStrategyEst,
    DataSavingEst,
    # Reconstruction mixins:
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    DataSavingReconstruction,
    # Base data saving:
    TwoPhaseFlow,
): ...  # type: ignore


# endregion

# region RUN
spe10_layer: int = 80

params = {
    # Base folder and file name. These will get changed by
    # ``ConvergenceAnalysisExtended``.
    "file_name": "setup",
    "progressbars": True,
    "nonlinear_solver_statistics": SolverStatisticsEst,
    "spe10_layer": spe10_layer - 1,
    "spe10_isotropic_perm": True,
    # HC params:
    "nonlinear_solver": pp.NewtonSolver,
    # Nonlinear params:
    "max_iterations": 20,
    # "nl_convergence_tol": 1e-10
    # * 10000
    # * PSI,  # Scale the nonlinear tolerance by pressure values.
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e15,
    # Grid and time discretization:
    "grid_type": "simplex",
    # Model:
    "formulation": "fractional_flow",
    "rel_perm_constants": {},
    "cap_press_constants": {},
}

cell_sizes: list[float] = [
    600 * FEET / 7.5,
    # 600 * FEET / 15,
    # 600 * FEET / 30,
    # 600 * FEET / 60,
    # 600 * FEET / 120,
]
rel_perm_constants_list: list[dict[str, Any]] = [
    {
        "model": "Brooks-Corey",
        "limit": False,
        "n1": 2,
        "n2": 2,  # 1 + 2/n_b
        "n3": 1,
    },
]
cap_press_constants_list: list[dict[str, Any]] = [
    {
        "model": "Brooks-Corey",
        "entry_pressure": 5 * PSI,
        "n_b": 2,
    },
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
        / "grid_convergence"
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
            # Reinitialize the time manager for each run
            "time_manager": pp.TimeManager(
                schedule=np.array([0, 1.0 * pp.DAY]),  # 5 days
                dt_init=1.0 * pp.DAY,  # time step size in days
                dt_min_max=(1e-6 * pp.DAY, 1.0 * pp.DAY),
                constant_dt=False,
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
    model = ConvergenceAnalysisEstimatesSPE(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")
        raise error

# endregion

# region PLOTTING
fig, ax = plt.subplots()

for i, (cell_size, rp_model, cp_model) in enumerate(
    itertools.product(cell_sizes, rel_perm_constants_list, cap_press_constants_list)
):
    filename: str = f"rp_{rp_model['model']}_cp._{cp_model['model']}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "grid_convergence"
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
    times: list[float] = []
    time_deltas: list[float] = []

    for j, time_step in enumerate(history_list):
        # Find out whether the time step converged. If not, we do not plot anything.
        # If yes, we plot the final value of the estimators.
        time: float = time_step["current time"]
        time_delta: float = time_step["time step size"]
        time_step_converged: bool = True
        try:
            next_time: float = history_list[j + 1]["current time"]
            if time >= next_time:
                time_step_converged = False
        except IndexError:
            # At the last time step, we assume convergence.
            time_step_converged = True

        if not time_step_converged:
            continue

        times.append(time)
        time_deltas.append(time_delta)
        residual_and_flux_est.append(
            time_step["residual_and_flux_est"][-1] / time_delta
        )
        glob_nonconformity_est.append(
            time_step["nonconformity_est"][-1][GLOBAL_PRESSURE] / time_delta
        )
        compl_nonconformity_est.append(
            time_step["nonconformity_est"][-1][COMPLIMENTARY_PRESSURE] / time_delta
        )

    # ax.semilogy(
    #     time_steps,
    #     np.array(residual_and_flux_est),
    #     label=f"{foldername.parents[0].stem} residual and flux error stimator",
    # )
    # ax.semilogy(
    #     time_steps,
    #     np.array(glob_nonconformity_est),
    #     label=f"{foldername.parents[0].stem} global pressure error estimator",
    # )
    # ax.semilogy(
    #     time_steps,
    #     np.array(compl_nonconformity_est),
    #     label=f"{foldername.parents[0].stem} complimentary pressure error estimator",
    # )
    # When there is only one time step, plot a constant value over the entire time
    # interval.
    if j == 0:
        times.insert(0, 0.0)
        residual_and_flux_est.append(residual_and_flux_est[-1])
        glob_nonconformity_est.append(glob_nonconformity_est[-1])
        compl_nonconformity_est.append(compl_nonconformity_est[-1])

    ax.semilogy(
        times,
        np.array(residual_and_flux_est)
        + np.array(glob_nonconformity_est)
        + np.array(compl_nonconformity_est),
        label=f"{foldername.parents[0].stem} total error estimator",
    )

ax.set_xlabel("Time (s)")
ax.set_ylabel("Estimator")
ax.set_title(f"Total error estimator")
# ax.set_ylim([5e-2, 1e3])
ax.legend()
plt.show()
fig.savefig(pathlib.Path(__file__).parent / "grid_convergence" / "convergence_plot.png")

# endregion
