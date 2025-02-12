r"""Study convergence of adaptive homotopy continuation on different grid sizes and with
 different rel. perm./cap. pressure models.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton.
- Adaptive homotopy continuation (AHC) with Newton and Appleyard chopping.

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
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from tpf.derived_models.fluid_values import INITIAL_PRESSURE
from tpf.derived_models.spe10 import SPE10Mixin
from tpf.models.homotopy_continuation import TwoPhaseFlowAHC
from tpf.numerics.nonlinear.hc_solver import HCSolver
from tpf.utils.constants_and_typing import FEET, PSI
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

# Catch numpy warnings.
np.seterr(all="raise")
warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# endregion


# region MODEL
class SPE10HC(
    SPE10Mixin,
    TwoPhaseFlowAHC,
):  # type: ignore

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
    "hc_nl_convergence_tol": 1e3,
    # Nonlinear params:
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e30,
}

cell_sizes: list[float] = [600 * FEET / 30, 600 * FEET / 60, 600 * FEET / 120]
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
    {"model": "van Genuchten-Burdine", "limit": False, "n_g": -1.0},
]
cap_press_constants_list: list[dict[str, Any]] = [
    {"model": None},
    {"model": "linear", "linear_param": 30 * PSI},
]
# NOTE Appleyard chopping with AHC is not fixed yet.
appleyard_chopping_list: list[bool] = [False]
initial_saturation_list: np.ndarray = np.linspace(0.2, 0.3, 5)

# region VARYING_CELL_SIZES
for i, (
    appleyard_chopping,
    initial_saturation,
    cell_size,
    rp_model_1,
    rp_model_2,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list,
        initial_saturation_list[[0, -1]],
        cell_sizes[:2],
        rel_perm_constants_list_1,
        rel_perm_constants_list_2[0:1],
        cap_press_constants_list[0:1],
    )
):
    continue
    logger.info(f"Varying cell sizes. Run {i + 1} of {len(cell_sizes)}")
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model 1: {rp_model_1['model']}, RP model 2: {rp_model_2['model']}, CP model: {cp_model['model']}."
    )
    filename: str = f"cellsz_{int(cell_size)}"
    appleyard_str: str = "_appleyard" if appleyard_chopping else ""
    foldername: pathlib.Path = (
        pathlib.Path(__file__).parent
        / f"ahc{appleyard_str}"
        / "varying_cell_sizes"
        / f"lay_{spe10_layer}_rp1_{rp_model_1['model']}_rp2_{rp_model_2['model']}_cp._{cp_model['model']}_init_s_{initial_saturation}"
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
                iter_optimal_range=(9, 12),
                iter_relax_factors=(0.7, 1.3),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "spe10_initial_saturation": initial_saturation,
            "meshing_arguments": {"cell_size": cell_size},
            "rel_perm_constants": {"model_1": rp_model_1, "model_2": rp_model_2},
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )
    model = SPE10HC(params)
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
    rp_model_1,
    rp_model_2,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list,
        initial_saturation_list[[0, -1]],
        cell_sizes[0:1],
        rel_perm_constants_list_1,
        rel_perm_constants_list_2[0:1],
        cap_press_constants_list,
    )
):
    logger.info(f"Varying capillary pressure models. Run {i + 1} of {len(cell_sizes)}.")
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model 1: {rp_model_1['model']}, RP model 2: {rp_model_2['model']}, CP model: {cp_model['model']}."
    )

    filename = f"cp._{cp_model['model']}"
    appleyard_str = "_appleyard" if appleyard_chopping else ""
    foldername = (
        pathlib.Path(__file__).parent
        / f"ahc{appleyard_str}"
        / "varying_cp_models"
        / f"cellsz{int(cell_size)}_lay_{spe10_layer}_rp1_{rp_model_1['model']}_rp2_{rp_model_2['model']}_init_s_{initial_saturation}"
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
                iter_optimal_range=(9, 12),
                iter_relax_factors=(0.7, 1.3),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "spe10_initial_saturation": initial_saturation,
            "meshing_arguments": {"cell_size": cell_size},
            "rel_perm_constants": {"model_1": rp_model_1, "model_2": rp_model_2},
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )

    model = SPE10HC(params)
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
    rp_model_1,
    rp_model_2,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list,
        initial_saturation_list[[0, -1]],
        cell_sizes[0:1],
        rel_perm_constants_list_1,
        rel_perm_constants_list_2,
        cap_press_constants_list[0:1],
    )
):
    continue
    logger.info(
        f"Varying rel. perm. models. Run {i + 1} of {len(rel_perm_constants_list_2)}."
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model 1: {rp_model_1['model']}, RP model 2: {rp_model_2['model']}, CP model: {cp_model['model']}."
    )

    if rp_model_2["model"] == "Corey":
        filename = f"rp1_{rp_model_1['model']}_rp2_{rp_model_2['model']}_power_{rp_model_2['power']}"
    else:
        filename = f"rp1_{rp_model_1['model']}_rp2_{rp_model_2['model']}"
    appleyard_str = "_appleyard" if appleyard_chopping else ""
    foldername = (
        pathlib.Path(__file__).parent
        / f"ahc{appleyard_str}"
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
                iter_optimal_range=(9, 12),
                iter_relax_factors=(0.7, 1.3),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "spe10_initial_saturation": initial_saturation,
            "meshing_arguments": {"cell_size": cell_size},
            "rel_perm_constants": {"model_1": rp_model_1, "model_2": rp_model_2},
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )

    model = SPE10HC(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion

# region VARYING_INITIAL_SATURATION
for i, (
    appleyard_chopping,
    initial_saturation,
    cell_size,
    rp_model_1,
    rp_model_2,
    cp_model,
) in enumerate(
    itertools.product(
        appleyard_chopping_list,
        initial_saturation_list,
        cell_sizes[0:1],
        rel_perm_constants_list_1,
        rel_perm_constants_list_2[0:1],
        cap_press_constants_list[0:1],
    )
):
    continue
    logger.info(
        f"Varying initial saturations. Run {i + 1} of {len(initial_saturation_list)}."
    )
    logger.info(
        f"Cell size: {cell_size:.2f}, initial saturation: {initial_saturation}"
        + f"RP model 1: {rp_model_1['model']}, RP model 2: {rp_model_2['model']}, CP model: {cp_model['model']}."
    )

    filename = f"init_s_{initial_saturation}"
    appleyard_str = "_appleyard" if appleyard_chopping else ""
    foldername = (
        pathlib.Path(__file__).parent
        / f"ahc{appleyard_str}"
        / "varying_saturations"
        / f"lay_{spe10_layer}_cellsz_{int(cell_size)}_rp1_{rp_model_1['model']}_rp2_{rp_model_2['model']}_cp._{cp_model['model']}"
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
                iter_optimal_range=(9, 12),
                iter_relax_factors=(0.7, 1.3),
                recomp_factor=0.1,
                recomp_max=5,
            ),
            "spe10_initial_saturation": initial_saturation,
            "meshing_arguments": {"cell_size": cell_size},
            "rel_perm_constants": {"model_1": rp_model_1, "model_2": rp_model_2},
            "cap_press_constants": cp_model,
            "nl_appleyard_chopping": appleyard_chopping,
            "max_iterations": max_iterations,
        }
    )

    model = SPE10HC(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

# endregion

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
        / "adaptive homotopy continuation"
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
