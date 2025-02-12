r"""We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
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
    - Saturation: residual water saturation (0.2).
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
import random
import shutil
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from numba import config
from tpf.derived_models.fluid_values import oil, water
from tpf.derived_models.spe10 import SPE10
from tpf.models.error_estimate import (
    DataSavingEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
)
from tpf.models.flow_and_transport import TwoPhaseFlow
from tpf.models.reconstruction import (
    DataSavingRec,
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
)
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
    # IterationExporting,
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
    DataSavingRec,
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
    # Model:
    "formulation": "fractional_flow",
    "material_constants": {
        "solid": pp.SolidConstants({"porosity": 0.3, "permeability": 1e-15}),
    },
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    "spe10_layer": spe10_layer,
    "spe10_isotropic_perm": True,
    # Nonlinear params:
    "nonlinear_solver_statistics": SolverStatisticsEst,
    "nonlinear_solver": pp.NewtonSolver,
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e15,
    # "nl_convergence_tol": 1e-10
    # * 10000
    # * PSI,  # Scale the nonlinear tolerance by pressure values
}

cell_sizes: list[float] = [
    # 600 * FEET / 15,
    600 * FEET / 30,
    600 * FEET / 60,
]
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
    filename: str = f"cellsz_{int(cell_size)}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "grid_convergence"
        / f"lay_{spe10_layer}_rp_{rp_model['model']}_cp._{cp_model['model']}"
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
                schedule=np.array([0, 5.0 * pp.DAY]),  # 5 days
                dt_init=1.0 * pp.DAY,  # time step size in days
                # dt_min_max=(1e-6 * pp.DAY, 1.0 * pp.DAY),
                # constant_dt=False,
                # recomp_factor=0.1,
                # recomp_max=5,
                # Run with constant time step s.t. the discretization error varies only
                # with grid size.
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
    model = ConvergenceAnalysisEstimatesSPE(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")
        raise error

# endregion

# region PLOTTING
fig, (ax1, ax2) = plt.subplots(2, 1)

for i, (cell_size, rp_model, cp_model) in enumerate(
    itertools.product(cell_sizes, rel_perm_constants_list, cap_press_constants_list)
):
    filename: str = f"cellsz_{int(cell_size)}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / "grid_convergence"
        / f"lay_{spe10_layer}_rp_{rp_model['model']}_cp._{cp_model['model']}"
        / filename
    )
    solver_statistics_file: pathlib.Path = foldername / "solver_statistics.json"
    with open(solver_statistics_file) as f:
        history = json.load(f)
        history_list = list(history.values())[1:]

    flow_equation_mismatch: list[float] = []
    transport_equation_mismatch: list[float] = []

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
        flow_equation_mismatch.extend(
            [iteration["flow"] for iteration in time_step["equilibrated_flux_mismatch"]]
        )
        transport_equation_mismatch.extend(
            iteration["transport"]
            for iteration in time_step["equilibrated_flux_mismatch"]
        )

        residual_and_flux_est.append(
            time_step["residual_and_flux_est"][-1]  # / time_delta
        )
        glob_nonconformity_est.append(
            time_step["nonconformity_est"][-1][GLOBAL_PRESSURE]  # / time_delta
        )
        compl_nonconformity_est.append(
            time_step["nonconformity_est"][-1][COMPLIMENTARY_PRESSURE]  # / time_delta
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

    ax1.semilogy(
        times,
        np.array(residual_and_flux_est)
        + np.array(glob_nonconformity_est)
        + np.array(compl_nonconformity_est),
        label=f"{filename} total error estimator",
        marker="s",
    )
    ax2.semilogy(
        range(len(flow_equation_mismatch)),
        np.array(flow_equation_mismatch),
        label=f"{filename} flow equation mismatch",
        marker="s",
    )
    ax2.semilogy(
        range(len(transport_equation_mismatch)),
        np.array(transport_equation_mismatch),
        label=f"{filename} transport equation mismatch",
        marker="s",
    )

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Estimator")
ax1.set_title(f"Total error estimator")
ax1.legend()

ax2.set_xlabel("Nonlinear iteration")
ax2.set_ylabel("Mismatch")
ax2.set_title(f"Flux equilibrations mismatch")
ax2.legend()

plt.show()
fig.savefig(pathlib.Path(__file__).parent / "grid_convergence" / "convergence_plot.png")

# endregion
