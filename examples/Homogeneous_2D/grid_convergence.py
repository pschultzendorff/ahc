r"""Homogenous five-spot example. Study convergence of the error estimators with grid
 size.

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
from tpf.derived_models.spe10 import SPE10Mixin
from tpf.models.error_estimate import TwoPhaseFlowErrorEstimate
from tpf.models.flow_and_transport import EquationsTPF
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
class HomogeneousGeometryMixin:
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


class ConvergenceAnalysisEstimatesHomogeneous(
    IterationExportingMixin,
    HomogeneousGeometryMixin,
    SPE10Mixin,
    TwoPhaseFlowErrorEstimate,
): ...  # type: ignore


# endregion

# region RUN

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
    # Nonlinear params:
    "nonlinear_solver_statistics": SolverStatisticsEst,
    "nonlinear_solver": pp.NewtonSolver,
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e15,
}

cell_sizes: list[float] = [
    # 600 * FEET / 7.5,
    # 600 * FEET / 15,
    600 * FEET / 30,
]
rel_perm_constants_list: list[dict[str, Any]] = [
    {"model": "linear", "limit": True},
]
cap_press_constants_list: list[dict[str, Any]] = [
    {"model": "linear", "entry_pressure": 0 * PSI},
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
    continue
    # We have the file name both in the folder name and the filename to make
    # distinguishing different runs in ParaView easier.
    filename: str = f"cellsz_{int(cell_size)}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent.resolve()
        / "grid_convergence"
        / f"rp_{rp_model['model']}_cp._{cp_model['model']}"
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
                schedule=np.array([0, 50.0 * pp.DAY]),  # 5 days
                dt_init=1.0 * pp.DAY,  # Time step size in days
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
    model = ConvergenceAnalysisEstimatesHomogeneous(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")
        raise error

# endregion

# region PLOTTING
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

for i, (cell_size, rp_model, cp_model) in enumerate(
    itertools.product(cell_sizes, rel_perm_constants_list, cap_press_constants_list)
):
    filename: str = f"cellsz_{int(cell_size)}"
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent.resolve()
        / "grid_convergence"
        / f"rp_{rp_model['model']}_cp._{cp_model['model']}"
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
    global_energy_norm: list[float] = []
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

        residual_and_flux_est.append(time_step["residual_and_flux_est"][-1])
        glob_nonconformity_est.append(
            time_step["nonconformity_est"][-1][GLOBAL_PRESSURE]
        )
        compl_nonconformity_est.append(
            time_step["nonconformity_est"][-1][COMPLIMENTARY_PRESSURE]
        )
        global_energy_norm.append(time_step["global_energy_norm"][-1])

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
        global_energy_norm.append(global_energy_norm[-1])

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
    for name, est in zip(
        ["Residual and flux", "NC1", "NC2"],
        [residual_and_flux_est, glob_nonconformity_est, compl_nonconformity_est],
    ):
        ax3.semilogy(
            times,
            np.array(est) / np.array(global_energy_norm),
            label=f"{filename} relative {name} estimator",
            marker="s",
        )
    ax4.semilogy(times, np.array(global_energy_norm), marker="s", label=filename)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Estimator")
ax1.set_title(f"Total error estimator")
ax1.legend()

ax2.set_xlabel("Nonlinear iteration")
ax2.set_ylabel("Mismatch")
ax2.set_title(f"Flux equilibrations mismatch")
ax2.legend()

ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Relative error")
ax3.set_title(f"Relative error estimators")
ax3.legend()

ax4.set_ylim(top=1e0)
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Global energy norm")
ax4.set_title(f"Global energy")
ax4.legend()

fig1.savefig(
    pathlib.Path(__file__).parent.resolve() / "grid_convergence" / "total_estimator.png"
)
fig2.savefig(
    pathlib.Path(__file__).parent.resolve()
    / "grid_convergence"
    / "equilibration_mismatch.png"
)
fig3.savefig(
    pathlib.Path(__file__).parent.resolve()
    / "grid_convergence"
    / "relative_estimators.png"
)
fig4.savefig(
    pathlib.Path(__file__).parent.resolve() / "grid_convergence" / "global_energy.png"
)

# endregion
