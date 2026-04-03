r"""Study convergence of solvers with different rel. perm./cap. pressure models and
 different initial saturations.

The following solvers are employed:
- Adaptive homotopy continuation (AHC) with Newton
- Homotopy continuation (HC) with Newton
- Adaptive Newton
- Adaptive Newton with Appleyard chopping


We loosely follow the setup of Wang and Tchelepi (2013) to test the homotopy
continuation. The considered model is similar to the heterogeneous 3D models in the
article (section 4.6.4), but on a 2D domain for now.

X. Wang and H. A. Tchelepi, “Trust-region based solver for nonlinear transport in
   heterogeneous porous media,” Journal of Computational Physics, vol. 253, pp.
   114–137, Nov. 2013, doi: 10.1016/j.jcp.2013.06.041.

Model description:
- 1200x2200 ft domain, layer 55 of the SPE10, case 2A (default; all 85 layers are
  tested in ``run_all_layers.py``).
- Constant water injection in the center: 87.5 m^3/day
- Oil production at the four corners: 4000 psi bhp
    - This is simulated by prescribing the bottom hole pressure and saturation (residual
      oil saturation) in the corner cells. We do NOT use a well model.
- Simulation time: 30 days
- Solid properties:
    - Porosity: SPE10 case 2A, layer 55
    - Permeability: SPE10 case 2A, layer 55
- Fluid properties:
    - Water: ``pp.fluid_values.water``. Residual saturation is 0.2.
    - Oil: PVT table from the SPE10, case 2A. We use the values at 8000 psi.
      Residual saturation is 0.2.
- Initial values:
    - Pressure: 4000 psi (= BHP; initial guess for Newton, no influence on the solution)
    - Saturation: Varying between 0.2 and 0.3.
- Rel. perm. models (model_2; model_1 is always linear for HC/AHC):
    - Corey with power 2.
    - Corey with power 3.
    - Brooks-Corey-Mualem with n_b=2, eta=2.
    - Brooks-Corey-Mualem with n_b=4, eta=2.
- Capillary pressure models (model_2; model_1 is always None for HC/AHC):
    - None.
    - Linear, entry pressure 50 Pa.
    - Brooks-Corey with n_b=2, entry pressure 100 Pa.
    - Brooks-Corey with n_b=4, entry pressure varying between 100 Pa and 300 Pa.

"""

import copy
import logging
import os
import pathlib
import shutil
import sys
import warnings
from typing import Type

import numpy as np
import porepy as pp
from ahc.derived_models.spe10 import INITIAL_PRESSURE, SPE10Mixin
from ahc.models.adaptive_newton import TwoPhaseFlowANewton
from ahc.models.homotopy_continuation import TwoPhaseFlowHC
from ahc.models.protocol import TPFProtocol
from ahc.viz.iteration_exporting import IterationExportingMixin

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from ..utils import SimulationConfig, clean_up_after_simulation, setup_params

# region SETUP


# Limit number of threads for NREC.
N_THREADS = "4"
os.environ["MKL_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = N_THREADS
os.environ["OMP_NUM_THREADS"] = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS

# Catch all numpy errors except underflow, which may occur when calculating estimators.
np.seterr(all="raise")
np.seterr(under="ignore")

warnings.filterwarnings("default")

# Setup logging.
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()

# endregion


# region MODEL
class InitialConditionsMixin(TPFProtocol):
    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation."""
        initial_pressure = np.full(self.g.num_cells, INITIAL_PRESSURE)
        initial_saturation = self.bound_saturation(
            np.full(self.g.num_cells, self.params["spe10_initial_saturation"])
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


class SPE10HC(
    InitialConditionsMixin,
    SPE10Mixin,
    TwoPhaseFlowHC,
):  # type: ignore
    ...


class SPE10Newton(
    InitialConditionsMixin,
    SPE10Mixin,
    TwoPhaseFlowANewton,
):  # type: ignore
    ...


# endregion

# region UTILS
spe10_layer: int = 55

default_solver_params = {
    "progressbars": True,
    # Model:
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    "spe10_quarter_domain": False,
    "spe10_isotropic_perm": True,
    # Nonlinear solver:
    "nl_enforce_physical_saturation": True,
    # Error estimator:
    "disable_spatial_est": True,
}

default_time_manager_params = {
    "schedule": np.array([0.0, 30.0 * pp.DAY]),
    "dt_init": 30.0 * pp.DAY,
    "constant_dt": False,
    "dt_min_max": (1e-3 * pp.DAY, 30.0 * pp.DAY),
    "iter_optimal_range": (9, 12),
    "iter_relax_factors": (0.7, 1.3),
    "recomp_factor": 0.1,
    "recomp_max": 5,
}


def setup_model(
    solver: str, iteration_exporting: bool = False
) -> Type[SPE10HC] | Type[SPE10Newton]:
    """Return a model class based on the solver name.

    Parameters:
        solver: The name of the solver ("AHC", "HC", "Newton", or "NewtonAppleyard").

    Returns:
        The model class with the correct adaptive solver.

    """
    if solver in ["HC", "AHC"]:
        model_class = SPE10HC
    elif solver in ["Newton", "NewtonAppleyard"]:
        model_class = SPE10Newton
    else:
        raise ValueError(f"Unknown solver: {solver}")
    if iteration_exporting:
        model_class = type(
            f"{model_class.__name__}WithIterationExporting",
            (IterationExportingMixin, model_class),
            {},
        )
    return model_class


def run_simulation(
    config: SimulationConfig,
    solver_params: dict | None = None,
    time_manager_params: dict | None = None,
    **kwargs,
) -> None:
    """Run simulation for a single configuration.

    Parameters:
        config: The simulation configuration.
        solver_params: Optional dictionary of solver parameters. If None, default
        parameters
            are used and updated with the parameters from the config.
        time_manager_params: Optional dictionary of time manager parameters. If None,
            default parameters are used and updated with the parameters from the config.
        **kwargs: Additional keyword arguments to pass to the model setup and parameter
            setup functions.

    """
    logger.info(
        f"solver: {config.solver_name}, "
        f"adaptive error ratio: {config.adaptive_error_ratio:.2f}, "
        f"cell size: {config.cell_size:.2f}, \n"
        f"initial saturation: {config.init_s}, "
        f"RP model 1: {config.rp_model_1}, "
        f"RP model 2: {config.rp_model_2}, \n"
        f"CP model: {config.cp_model_2}."
    )
    model_class = setup_model(config.solver_name, **kwargs)
    updated_solver_params, updated_time_manager_params = setup_params(
        config.solver_name, config.adaptive_error_ratio, **kwargs
    )

    # Build params dictionaries.
    if solver_params is None:
        solver_params = copy.deepcopy(default_solver_params) | updated_solver_params
    if time_manager_params is None:
        time_manager_params = (
            copy.deepcopy(default_time_manager_params) | updated_time_manager_params
        )

    # Newton and Appleyard Newton require only one of each constitutive law.
    if config.solver_name.startswith("Newton"):
        rel_perm_constants = config.rp_model_2
        cap_press_constants = config.cp_model_2
    else:
        rel_perm_constants = {
            "model_1": config.rp_model_1,
            "model_2": config.rp_model_2,
        }
        cap_press_constants = {
            "model_1": config.cp_model_1,
            "model_2": config.cp_model_2,
        }
    solver_params.update(
        {
            "meshing_arguments": {"cell_size": config.cell_size},
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": cap_press_constants,
            "spe10_initial_saturation": config.init_s,
            "spe10_layer": config.spe10_layer,
            "folder_name": config.folder_name,
            "file_name": config.file_name,
            "solver_statistics_file_name": config.folder_name
            / "solver_statistics.json",
            "time_manager": pp.TimeManager(**time_manager_params),
        }
    )

    try:
        shutil.rmtree(config.folder_name)
        config.folder_name.mkdir(parents=True)
    except Exception:
        pass

    try:
        model = model_class(solver_params)
        pp.run_time_dependent_model(model=model, params=solver_params)
    except Exception as e:
        logger.error(f"Run failed with error: {e}.")

    # Save number of grid cells to a file.
    with (config.folder_name / "num_grid_cells.txt").open("w") as f:
        f.write(str(model.g.num_cells))


# endregion

# region RUN
solvers_and_ratios: list[tuple[str, float]] = [
    ("AHC", 0.1),
    ("AHC", 0.01),
    ("HC", 0.1),
    ("Newton", 0.1),
    ("NewtonAppleyard", 0.1),
]


rp_models = {
    "linear": {"model": "linear", "limit": True},
    "Brooks-Corey_nb_4": {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 4.0,
        "eta": 2.0,
    },  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1
    "Brooks-Corey_nb_2": {
        "model": "Brooks-Corey-Mualem",
        "limit": True,
        "n_b": 2.0,
        "eta": 2.0,
    },  #  n_1 = eta = 2, n_2 = 1 + 1/n_b = 2, n_3 = 1
    "Corey_power_2": {"model": "Corey", "power": 2, "limit": True},
    "Corey_power_3": {"model": "Corey", "power": 3, "limit": True},
}

cp_models = {
    "None": {
        "model": None,
    },
    "linear": {
        "model": "linear",
        "entry_pressure": 50 * pp.PASCAL,
        "linear_param": 5.0,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
    "Brooks-Corey_nb_4": {
        "model": "Brooks-Corey",
        "n_b": 4.0,
        "entry_pressure": 100 * pp.PASCAL,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
    "Brooks-Corey_nb_2": {
        "model": "Brooks-Corey",
        "n_b": 2.0,
        "entry_pressure": 100 * pp.PASCAL,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
}


def generate_configs() -> list[SimulationConfig]:
    """Generate all simulation configurations."""
    results_dir = dirname / "results"
    results_dir.mkdir(exist_ok=True)

    configs = []

    # region VISCOUS

    if True:
        # Varying rel. perm. models at init_s = 0.2 and init_s = 0.3 with linear capillary
        # pressure.
        for init_s in [0.2, 0.3]:
            for rp_model_name, rp_model in rp_models.items():
                if rp_model_name == "linear":
                    continue
                for solver_name, adaptive_error_ratio in solvers_and_ratios:
                    folder_name = (
                        results_dir
                        / f"{solver_name}_{adaptive_error_ratio:.3f}"
                        / "viscous"
                        / "varying_rp"
                        / f"init_s_{init_s}"
                        / rp_model_name
                    )
                    configs.append(
                        SimulationConfig(
                            file_name=rp_model_name,
                            folder_name=folder_name,
                            solver_name=solver_name,
                            adaptive_error_ratio=adaptive_error_ratio,
                            init_s=init_s,
                            rp_model_1=rp_models["linear"],
                            rp_model_2=rp_model,
                            cp_model_1=cp_models["None"],
                            cp_model_2=cp_models["linear"],
                            spe10_layer=spe10_layer,
                        )
                    )

    if True:
        # Varying init_s for the more challenging Brooks-Corey rel. perm. model.
        for init_s in list(np.linspace(0.2, 0.3, 5)[1:-1]):
            for solver_name, adaptive_error_ratio in solvers_and_ratios:
                file_name = f"init_s_{init_s:.2f}"
                folder_name = (
                    results_dir
                    / f"{solver_name}_{adaptive_error_ratio:.3f}"
                    / "viscous"
                    / "varying_init_s"
                    / file_name
                )
                configs.append(
                    SimulationConfig(
                        file_name=file_name,
                        folder_name=folder_name,
                        solver_name=solver_name,
                        adaptive_error_ratio=adaptive_error_ratio,
                        init_s=init_s,
                        rp_model_1=rp_models["linear"],
                        rp_model_2=rp_models["Brooks-Corey_nb_2"],
                        cp_model_1=cp_models["None"],
                        cp_model_2=cp_models["linear"],
                        spe10_layer=spe10_layer,
                    )
                )

    # endregion

    # region VISCOUS_AND_CAPILLARY
    # NOTE HC starts with a linear rel. perm. model and zero capillary pressure.

    if True:
        # Varying rel. perm. and cap. press. models at init_s = 0.3 with Brooks-Corey
        # capillary pressure.
        for rp_model_name, rp_model in rp_models.items():
            if rp_model_name == "linear":
                continue
            for solver_name, adaptive_error_ratio in solvers_and_ratios:
                folder_name = (
                    results_dir
                    / f"{solver_name}_{adaptive_error_ratio:.3f}"
                    / "viscous_and_capillary"
                    / "varying_rp"
                    / f"init_s_{0.3}"
                    / rp_model_name
                )
                cp_model_2 = (
                    cp_models["Brooks-Corey_nb_2"]
                    if rp_model_name == "Brooks-Corey_nb_2"
                    else cp_models["Brooks-Corey_nb_4"]
                )
                configs.append(
                    SimulationConfig(
                        file_name=rp_model_name,
                        folder_name=folder_name,
                        solver_name=solver_name,
                        adaptive_error_ratio=adaptive_error_ratio,
                        init_s=0.3,
                        rp_model_1=rp_models["linear"],
                        rp_model_2=rp_model,
                        cp_model_1=cp_models["None"],
                        cp_model_2=cp_model_2,
                        spe10_layer=spe10_layer,
                    )
                )

    if True:
        # Varying init_s for the less challenging Brooks-Corey model.
        for init_s in list(np.linspace(0.2, 0.3, 5)[1:-1]):
            for solver_name, adaptive_error_ratio in solvers_and_ratios:
                file_name = f"init_s_{init_s:.2f}"
                folder_name = (
                    results_dir
                    / f"{solver_name}_{adaptive_error_ratio:.3f}"
                    / "viscous_and_capillary"
                    / "varying_init_s"
                    / file_name
                )
                configs.append(
                    SimulationConfig(
                        file_name=file_name,
                        folder_name=folder_name,
                        solver_name=solver_name,
                        adaptive_error_ratio=adaptive_error_ratio,
                        init_s=init_s,
                        rp_model_1=rp_models["linear"],
                        rp_model_2=rp_models["Brooks-Corey_nb_4"],
                        cp_model_1=cp_models["None"],
                        cp_model_2=cp_models["Brooks-Corey_nb_4"],
                        spe10_layer=spe10_layer,
                    )
                )

    if True:
        # Less challenging Brooks-Corey cap. pressure with different entry pressures.
        for entry_pressure in [100, 200, 300]:
            for solver_name, adaptive_error_ratio in solvers_and_ratios:
                file_name = f"entry_pressure_{entry_pressure}_hc_from_none"
                folder_name = (
                    results_dir
                    / f"{solver_name}_{adaptive_error_ratio:.3f}"
                    / "viscous_and_capillary"
                    / "varying_entry_pressure"
                    / file_name
                )
                cp_model_2 = cp_models["Brooks-Corey_nb_4"].copy()
                cp_model_2["entry_pressure"] = entry_pressure * pp.PASCAL
                configs.append(
                    SimulationConfig(
                        file_name=file_name,
                        folder_name=folder_name,
                        solver_name=solver_name,
                        adaptive_error_ratio=adaptive_error_ratio,
                        init_s=0.3,
                        rp_model_1=rp_models["linear"],
                        rp_model_2=rp_models["Brooks-Corey_nb_4"],
                        cp_model_1=cp_models["None"],
                        cp_model_2=cp_model_2,
                        spe10_layer=spe10_layer,
                    )
                )
    # endregion

    return configs


if __name__ == "__main__":
    configs = generate_configs()
    for config in configs:
        run_simulation(config)
        clean_up_after_simulation(config)

# endregion
