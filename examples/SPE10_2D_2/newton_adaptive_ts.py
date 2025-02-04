r"""Study convergence of Newton on different grid sizes and with different rel.
 perm./cap. pressure models.

The following solvers are employed:
- Newton
- Newton with Appleyard chopping

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
    - Saturation: Varying between 0.2 and 0.3.
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
import shutil
import warnings
from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from tpf.models.error_estimate import TwoPhaseFlowErrorEstimate
from tpf.spe10.fluid_values import INITIAL_PRESSURE
from tpf.spe10.model import SPE10Mixin
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    FEET,
    GLOBAL_PRESSURE,
    PSI,
)
from tpf.viz.solver_statistics import SolverStatisticsEst

# region SETUP

# Disable numba JIT for debugging.
# config.DISABLE_JIT = False

# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch numpy warnings.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion


# region MODEL
class SPE10Newton(
    SPE10Mixin,
    TwoPhaseFlowErrorEstimate,
):  # type: ignore

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation.

        The corner cells get prescibed the right values immediately. Inside the
        reservoir, the initial pressure is higher. The initial saturation is set to the
        residual wetting saturation + 0.1 inside the reservoir.

        """
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
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e30,
}

cell_sizes: list[float] = [600 * FEET / 30, 600 * FEET / 60, 600 * FEET / 120]
rel_perm_constants_list: list[dict[str, Any]] = [
    {"model": "linear", "limit": False},
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
appleyard_chopping_list: list[bool] = [False, True]
initial_saturation_list: np.ndarray = np.linspace(0.2, 0.3, 5)

# region VAYRING_CELL_SIZES
for i, (
    appleyard_chopping,
    initial_saturation,
    cell_size,
    rp_model,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list,
        # initial_saturation_list[[0, -1]],
        initial_saturation_list[0:1],
        cell_sizes[:2],
        rel_perm_constants_list[1:2],
        cap_press_constants_list[0:1],
    )
):
    continue
    logger.info(f"Varying cell sizes. Run {i + 1} of {len(cell_sizes)}.")
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model: {rp_model['model']}, CP model: {cp_model['model']}."
    )

    filename: str = f"cellsz_{int(cell_size)}"
    appleyard_str: str = "_appleyard" if appleyard_chopping else ""
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / f"newton{appleyard_str}_adaptive_ts"
        / "varying_cell_sizes"
        / f"lay_{spe10_layer}_rp_{rp_model['model']}_cp._{cp_model['model']}_init_s_{initial_saturation}"
        / filename
    )

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    max_iterations: int = 50 if appleyard_chopping else 20
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
            "rel_perm_constants": rp_model,
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )
    model = SPE10Newton(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion

# region VARYING_CAP_PRESS_MODELS
for i, (
    appleyard_chopping,
    initial_saturation,
    cell_size,
    rp_model,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list,
        # initial_saturation_list[[0, -1]],
        initial_saturation_list[0:1],
        cell_sizes[0:1],
        rel_perm_constants_list[1:2],
        cap_press_constants_list,
    )
):
    continue
    logger.info(f"Varying capillary pressure models. Run {i + 1} of {len(cell_sizes)}.")
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model: {rp_model['model']}, CP model: {cp_model['model']}."
    )

    filename = f"cp._{cp_model['model']}"
    appleyard_str = "_appleyard" if appleyard_chopping else ""
    foldername = (
        pathlib.Path(__file__).parent
        / f"newton{appleyard_str}_adaptive_ts"
        / "varying_cp_models"
        / f"cellsz{int(cell_size)}_lay_{spe10_layer}_rp_{rp_model['model']}_init_s_{initial_saturation}"
        / filename
    )

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    max_iterations if appleyard_chopping else 20
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
            "rel_perm_constants": rp_model,
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )
    model = SPE10Newton(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")
# endregion

# region VARYING_REL_PERM_MODELS
for i, (
    appleyard_chopping,
    initial_saturation,
    cell_size,
    rp_model,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list[1:2],
        initial_saturation_list[0:1],
        # initial_saturation_list[[0, -1]],
        cell_sizes[0:1],
        rel_perm_constants_list,
        cap_press_constants_list[0:1],
    )
):
    logger.info(
        f"Varying rel. perm. models. Run {i + 1} of {len(rel_perm_constants_list)}."
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model: {rp_model['model']}, CP model: {cp_model['model']}."
    )
    if rp_model["model"] == "Corey":
        filename = f"rp_{rp_model['model']}_power_{rp_model['power']}"
    else:
        filename = f"rp_{rp_model['model']}"
    appleyard_str = "_appleyard" if appleyard_chopping else ""
    foldername = (
        pathlib.Path(__file__).parent
        / f"newton{appleyard_str}_adaptive_ts"
        / "varying_rel_perm_models"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}_cp._{cp_model['model']}_init_s_{initial_saturation}"
        / filename
    )

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    max_iterations = 50 if appleyard_chopping else 20
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
            "rel_perm_constants": rp_model,
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )
    model = SPE10Newton(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")
        raise error

# endregion

# region VARYING_INITIAL_SATURATION
for i, (
    appleyard_chopping,
    initial_saturation,
    cell_size,
    rp_model,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list,
        initial_saturation_list[:-1],
        cell_sizes[0:1],
        rel_perm_constants_list[1:2],
        cap_press_constants_list[0:1],
    )
):
    logger.info(
        f"Varying initial saturations. Run {i + 1} of {len(initial_saturation_list)}."
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model: {rp_model['model']}, CP model: {cp_model['model']}."
    )

    filename = f"init_s_{initial_saturation}"
    appleyard_str = "_appleyard" if appleyard_chopping else ""
    foldername = (
        pathlib.Path(__file__).parent
        / f"newton{appleyard_str}_adaptive_ts"
        / "varying_saturations"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}_rp_{rp_model['model']}_cp._{cp_model['model']}"
        / filename
    )

    try:
        shutil.rmtree(foldername)
    except Exception:
        pass
    foldername.mkdir(parents=True)

    max_iterations = 50 if appleyard_chopping else 20
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
            "rel_perm_constants": rp_model,
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )
    model = SPE10Newton(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")


# endregion

# endregion

# region PLOTTING
for i, (rp_model, cp_model) in enumerate(
    itertools.product(rel_perm_constants_list, cap_press_constants_list)
):
    continue
    filename = f"rp_{rp_model['model']}_cp._{cp_model['model']}"
    foldername = (
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
    global_energy_norm: list[float] = []

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for n, time_step in enumerate(history_list):
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
            next_time: float = history_list[n + 1]["current time"]
            if time < next_time:
                ax1.axvline(x=len(residual_and_flux_est), color="gray", linestyle="--")
        except IndexError:
            pass

    ax1.semilogy(residual_and_flux_est, label="Residual and flux estimator")
    ax1.semilogy(
        glob_nonconformity_est, label="Global pressure nonconformity estimator"
    )
    ax1.semilogy(
        compl_nonconformity_est, label="Complimentary pressure nonconformity estimator"
    )
    ax1.set_xlabel("Nonlinear iteration (over multiple time steps)")
    ax1.set_ylabel("Estimator value")
    ax1.set_title(f"Discretization estimators")
    ax1.legend()
    fig1.savefig(foldername / "estimators.png")

    ax2.semilogy(
        np.array(glob_nonconformity_est) / np.array(global_energy_norm),
        label="Relative global pressure nonconformity estimator",
    )
    ax2.semilogy(
        np.array(residual_and_flux_est) / np.array(global_energy_norm),
        label="Relative residual and flux estimator",
    )
    ax2.semilogy(
        np.array(compl_nonconformity_est) / np.array(global_energy_norm),
        label="Relative complimentary pressure nonconformity estimator",
    )
    ax2.set_xlabel("Nonlinear iteration (over multiple time steps)")
    ax2.set_ylabel("Relative estimator value")
    ax2.set_title(f"Relative discretization estimators")
    ax2.legend()
    fig2.savefig(foldername / "relative_estimators.png")

# endregion
