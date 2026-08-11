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

import numpy as np
import porepy as pp
from ahc.derived_models.spe10 import (
    HEIGHT,
    INITIAL_PRESSURE,
    WIDTH,
    SPE10Mixin,
    oil,
    water,
)
from ahc.models.adaptive_newton import TwoPhaseFlowANewton
from ahc.models.homotopy_continuation import TwoPhaseFlowHC
from ahc.models.phase import FluidPhase
from ahc.models.protocol import TPFProtocol
from ahc.utils.constants_and_typing import NONWETTING, WETTING
from ahc.viz.iteration_exporting import IterationExportingMixin

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils import SimulationConfig, clean_up_after_simulation, setup_porepy_params

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

# Directories for results.
dirname: pathlib.Path = pathlib.Path(__file__).parent.resolve()
results_dir = dirname / "results"

# endregion


# region MODEL
class EditableSPE10ParametersMixin(TPFProtocol):
    def set_phases(self) -> None:
        """Change oil density"""
        self.phases: dict[str, FluidPhase] = {}
        for phase_name, constants in zip([WETTING, NONWETTING], [water, oil]):
            if phase_name == WETTING:
                constants.update(
                    {"density": self.params.get("spe10_water_density")}
                )  # kg/m^3

            phase = FluidPhase(constants)
            phase.set_units(self.units)
            setattr(self, phase_name, phase)
            self.phases[phase_name] = phase

    def initial_condition(self) -> None:
        """Change initial values for pressure and saturation.

        - Gravity separation: The upper half of the domain is fully saturated with the
          more dense phase (water), lower half is fully saturated with the less dense
          phase (oil).
        - Five-spot setup: The saturation in the full domain is set to
          ``self.params["spe10_initial_saturation"]``.

        """

        if self.params["spe10_case"] == "gravity_separation":
            initial_pressure = np.full(self.g.num_cells, 0.0)
            height: float = (
                (HEIGHT / 2) if self.params["spe10_quarter_domain"] else HEIGHT
            )
            width: float = WIDTH / 2 if self.params["spe10_quarter_domain"] else WIDTH
            # Choose initial saturation depending on whether cell is above or below
            # slanted line
            initial_saturation = self.bound_saturation(
                # self.g.cell_centers has shape=(ambient_dimension, num_cells)
                np.array(
                    [
                        0.7
                        if cell[1] >= height / 2 + 10 * (1 - 2 * cell[0] / width)
                        else 0.3
                        for cell in np.swapaxes(self.g.cell_centers, 0, 1)
                    ]
                )
            )
        elif self.params["spe10_case"] == "five_spot":
            initial_pressure = np.full(self.g.num_cells, INITIAL_PRESSURE)
            initial_saturation = self.bound_saturation(
                np.full(self.g.num_cells, self.params["spe10_initial_saturation"])
            )
        else:
            raise ValueError(
                f"Unknown SPE10 case '{self.params['spe10_case']}'."
                + " Supported cases are 'gravity_separation' and 'five_spot'."
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
    EditableSPE10ParametersMixin,
    SPE10Mixin,
    TwoPhaseFlowHC,
):  # type: ignore
    ...


class SPE10Newton(
    EditableSPE10ParametersMixin,
    SPE10Mixin,
    TwoPhaseFlowANewton,
):  # type: ignore
    ...


# endregion

# region UTILS
SPE10_LAYER: int = 55

default_solver_params = {
    "progressbars": True,
    # Model:
    "material_constants": {},
    "rel_perm_constants": {},
    "cap_press_constants": {},
    "grid_type": "simplex",
    "spe10_quarter_domain": False,
    "spe10_isotropic_perm": True,
    "spe10_case": "five_spot",
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


def setup_porepy_model(
    config: SimulationConfig, **kwargs
) -> type[SPE10HC] | type[SPE10Newton]:
    """Return a model class based on the solver name.

    Parameters:
        config: The simulation configuration specifying the solver type.
        **kwargs: Additional keyword arguments. Currently, only ``iteration_exporting``
            is supported, which adds the ``IterationExportingMixin`` to the model class.

    Returns:
        The model class with the correct adaptive solver.

    """
    if config.solver_name in ["HC", "AHC"]:
        model_class = SPE10HC
    elif config.solver_name in ["Newton", "NewtonAppleyard"]:
        model_class = SPE10Newton
    else:
        raise ValueError(f"Unknown solver: {config.solver_name}")

    if kwargs.get("iteration_exporting", False):
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
            parameters are used and updated with the parameters from the config.
        time_manager_params: Optional dictionary of time manager parameters. If None,
            default parameters are used and updated with the parameters from the config.
        **kwargs: Additional keyword arguments to pass to the model setup and parameter
            setup functions.

    """
    logger.info(
        f"solver: {config.solver_name}, "
        f"HC tolerance: {config.hc_tol:.2f}, "
        f"NL tolerance: {config.nl_tol:.2f}, "
        f"cell size: {config.cell_size:.2f}, \n"
        f"initial saturation: {config.init_s}, "
        f"RP model 1: {config.rp_model_1}, "
        f"RP model 2: {config.rp_model_2}, \n"
        f"CP model: {config.cp_model_2}."
        f"Buoyancy: {config.buoyancy_constants_2}"
    )

    model_class = setup_porepy_model(config, **kwargs)
    updated_solver_params, updated_time_manager_params = setup_porepy_params(
        config, **kwargs
    )

    # Build porepy params dictionaries.
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
        buoyancy_constants = config.buoyancy_constants_2
    else:
        rel_perm_constants = {
            "model_1": config.rp_model_1,
            "model_2": config.rp_model_2,
        }
        cap_press_constants = {
            "model_1": config.cp_model_1,
            "model_2": config.cp_model_2,
        }
        buoyancy_constants = {
            "model_1": config.buoyancy_constants_1,
            "model_2": config.buoyancy_constants_2,
        }
    folder_name = config.folder_name()
    solver_params.update(
        {
            "meshing_arguments": {"cell_size": config.cell_size},
            "rel_perm_constants": rel_perm_constants,
            "cap_press_constants": cap_press_constants,
            "buoyancy_constants": buoyancy_constants,
            "spe10_initial_saturation": config.init_s,
            "spe10_layer": config.spe10_layer,
            "spe10_case": config.spe10_case,
            "spe10_water_density": config.spe10_water_density,
            "folder_name": folder_name,
            "file_name": config.case.name
            if isinstance(config.case, pathlib.Path)
            else config.case,
            "solver_statistics_file_name": folder_name / "solver_statistics.json",
            "time_manager": pp.TimeManager(**time_manager_params),
        }
    )

    # Remove previous runs.
    shutil.rmtree(folder_name, ignore_errors=True)
    folder_name.mkdir(parents=True)

    try:
        model = model_class(solver_params)
        pp.run_time_dependent_model(model=model, params=solver_params)
    except Exception as e:
        logger.error(f"Run failed with error: {e}.")
        raise e

    # Save number of grid cells to a file.
    with (folder_name / "num_grid_cells.txt").open("w") as f:
        f.write(str(model.g.num_cells))


# endregion

# region SIMULATIONS
solvers_and_tols: list[tuple[str, float, float]] = [
    ("AHC", 0.1, 0.1),
    ("AHC", 0.1, 0.01),
    ("AHC", 0.01, 0.01),
    ("HC", 0.05, 1e-3),
    ("HC", 0.01, 1e-3),
    ("HC", 0.01, 1e-5),
    ("Newton", 0.0, 0.1),
    ("NewtonAppleyard", 0.0, 0.1),
]


LINEAR_RP_MODEL = {"model": "linear", "limit": True}
rp_models = {
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

ZERO_CP_MODEL = {"model": None}
cp_models = {
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
        "entry_pressure": 200 * pp.PASCAL,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
    "Brooks-Corey_nb_2": {
        "model": "Brooks-Corey",
        "n_b": 2.0,
        "entry_pressure": 200 * pp.PASCAL,
        "limit": True,
        "max": 1e6 * pp.PASCAL,
    },
}

ZERO_BUOYANCY_MODEL = {"gravity_acceleration": 0.0}
buoyancy_models = {"gravity_on": {"gravity_acceleration": pp.GRAVITY_ACCELERATION}}


def generate_viscous_varying_rp_cases(init_s: float) -> list[SimulationConfig]:
    """Generate simulation configurations for viscous-dominated flow with varying
    relative permeability models, linear capillary pressure, and prescribed initial
    saturation.

    """
    cases = []
    for rp_model_name, rp_model in rp_models.items():
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous",
                    study="varying_rp",
                    case=pathlib.Path(f"init_s_{init_s}") / rp_model_name,
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=init_s,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_model,
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_models["linear"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                    spe10_layer=SPE10_LAYER,
                )
            )

    return cases


def generate_viscous_varying_init_s_cases() -> list[SimulationConfig]:
    """Generate simulation configurations for viscous-dominated flow with initial
    saturation varying between 0.225 and 0.275, the more challenging Brooks-Corey rel. perm.
    model, and linear capillary pressure.

    """
    cases = []
    for init_s in list(np.linspace(0.2, 0.3, 5)[1:-1]):
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous",
                    study="varying_init_s",
                    case=f"init_s_{init_s:.2f}",
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=init_s,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_models["Brooks-Corey_nb_2"],
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_models["linear"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                    spe10_layer=SPE10_LAYER,
                )
            )

    return cases


def generate_gravity_separation_cases() -> list[SimulationConfig]:
    """Generate simulation configurations for pure gravity-driven flow with varying
    water density, the less challenging Brooks-Corey rel. perm. model, and Brooks-Corey
    capillary pressure.

    """
    cases = []
    for water_density in [10000.0, 5000.0, water["density"], 200.0]:
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="gravity_separation",
                    study="varying_water_density",
                    case=f"water_density_{water_density:.2f}",
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=0.0,  # NOTE This is overwritten for spe10_case="gravity_separation".
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_models["Brooks-Corey_nb_4"],
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_models["Brooks-Corey_nb_4"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=buoyancy_models["gravity_on"],
                    spe10_layer=SPE10_LAYER,
                    spe10_case="gravity_separation",
                    spe10_water_density=water_density,  # kg/m^3
                )
            )

    return cases


def generate_capillary_varying_rp() -> list[SimulationConfig]:
    """Generate simulations for coupled viscous-capillary flow with varying rel. perm.
    and cap. press. models and an initial saturation of :math:`0.3`.

    """
    cases = []
    for rp_model_name, rp_model in rp_models.items():
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            # Make sure that the target capillary pressure model is compatible with the
            # relative permeability model.
            cp_model_2 = (
                cp_models["Brooks-Corey_nb_2"]
                if rp_model_name == "Brooks-Corey_nb_2"
                else cp_models["Brooks-Corey_nb_4"]
            )
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous_and_capillary",
                    study="varying_rp",
                    case=pathlib.Path(f"init_s_{0.3}") / rp_model_name,
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=0.3,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_model,
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_model_2,
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                    spe10_layer=SPE10_LAYER,
                )
            )

    return cases


def generate_capillary_varying_init_s() -> list[SimulationConfig]:
    """Generate simulations for coupled viscous-capillary flow with varying initial
    saturation between 0.225 and 0.275 and the less challenging Brooks-Corey rel. perm.
    and cap. press. model.

    """
    cases = []
    for init_s in list(np.linspace(0.2, 0.3, 5)[1:-1]):
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous_and_capillary",
                    study="varying_init_s",
                    case=f"init_s_{init_s:.2f}",
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=init_s,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_models["Brooks-Corey_nb_4"],
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_models["Brooks-Corey_nb_4"],
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                    spe10_layer=SPE10_LAYER,
                )
            )

    return cases


def generate_capillary_varying_entry_pressure() -> list[SimulationConfig]:
    """Generate simulations for coupled viscous-capillary flow with varying entry
    pressure, the less challenging Brooks-Corey rel. perm. and cap. press. model, and an
    initial saturation of :math:`0.3`.

    """
    cases = []
    for entry_pressure in [200, 500, 1000]:
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cp_model_2 = cp_models["Brooks-Corey_nb_4"].copy()
            cp_model_2["entry_pressure"] = entry_pressure * pp.PASCAL
            cases.append(
                SimulationConfig(
                    results_dir=results_dir,
                    regime="viscous_and_capillary",
                    study="varying_entry_pressure",
                    case=f"entry_pressure_{entry_pressure}_hc_from_none",
                    solver_name=solver_name,
                    hc_tol=hc_tol,
                    nl_tol=nl_tol,
                    init_s=0.3,
                    rp_model_1=LINEAR_RP_MODEL,
                    rp_model_2=rp_models["Brooks-Corey_nb_4"],
                    cp_model_1=ZERO_CP_MODEL,
                    cp_model_2=cp_model_2,
                    buoyancy_constants_1=ZERO_BUOYANCY_MODEL,
                    buoyancy_constants_2=ZERO_BUOYANCY_MODEL,
                    spe10_layer=SPE10_LAYER,
                )
            )

    return cases


def generate_buoyancy_varying_rp() -> list[SimulationConfig]:
    """Generate simulations for coupled viscous-capillary-gravity flow with varying rel.
    perm. models, Brooks-Corey capillary pressure, and an initial saturation of
    :math:`0.3`.

    Note: The HC solvers are run both from gravity on and gravity off as an auxiliary
        problem.

    """
    cases = []
    for rp_model_name, rp_model in rp_models.items():
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            cp_model_2 = (
                cp_models["Brooks-Corey_nb_2"]
                if rp_model_name == "Brooks-Corey_nb_2"
                else cp_models["Brooks-Corey_nb_4"]
            )

            if solver_name.endswith("HC"):
                hc_gravity_cases = [
                    ("", ZERO_BUOYANCY_MODEL),
                    ("from_gravity_on", buoyancy_models["gravity_on"]),
                ]
            else:
                hc_gravity_cases = [("", ZERO_BUOYANCY_MODEL)]

            for solver_name_postfix, buoyancy_constants_1 in hc_gravity_cases:
                cases.append(
                    SimulationConfig(
                        results_dir=results_dir,
                        regime="viscous_and_capillary_and_gravity",
                        study="varying_rp",
                        case=pathlib.Path(f"init_s_{0.3}") / rp_model_name,
                        solver_name=solver_name,
                        solver_name_postfix=solver_name_postfix,
                        hc_tol=hc_tol,
                        nl_tol=nl_tol,
                        init_s=0.3,
                        rp_model_1=LINEAR_RP_MODEL,
                        rp_model_2=rp_model,
                        cp_model_1=ZERO_CP_MODEL,
                        cp_model_2=cp_model_2,
                        buoyancy_constants_1=buoyancy_constants_1,
                        buoyancy_constants_2=buoyancy_models["gravity_on"],
                        spe10_layer=SPE10_LAYER,
                    )
                )

    return cases


def generate_buoyancy_varying_density() -> list[SimulationConfig]:
    """Generate simulations for coupled viscous-capillary-gravity flow with varying
    density contrast, the less challenging Brooks-Corey rel. perm. and capillary
    pressure models, and an initial saturation of :math:`0.3`.

    Note: The HC solvers are run both from gravity on and gravity off as an auxiliary
        problem.

    """
    cases = []
    for water_density in [10000.0, 5000.0, water["density"], 200.0]:
        for solver_name, hc_tol, nl_tol in solvers_and_tols:
            if solver_name.endswith("HC"):
                hc_gravity_cases = [
                    ("", ZERO_BUOYANCY_MODEL),
                    ("from_gravity_on", buoyancy_models["gravity_on"]),
                ]
            else:
                hc_gravity_cases = [("", ZERO_BUOYANCY_MODEL)]

            for solver_name_postfix, buoyancy_constants_1 in hc_gravity_cases:
                cases.append(
                    SimulationConfig(
                        results_dir=results_dir,
                        regime="viscous_and_capillary_and_gravity",
                        study="varying_water_density",
                        case=f"water_density_{water_density:.2f}",
                        solver_name=solver_name,
                        solver_name_postfix=solver_name_postfix,
                        hc_tol=hc_tol,
                        nl_tol=nl_tol,
                        init_s=0.3,
                        rp_model_1=LINEAR_RP_MODEL,
                        rp_model_2=rp_models["Brooks-Corey_nb_4"],
                        cp_model_1=ZERO_CP_MODEL,
                        cp_model_2=cp_models["Brooks-Corey_nb_4"],
                        buoyancy_constants_1=buoyancy_constants_1,
                        buoyancy_constants_2=buoyancy_models["gravity_on"],
                        spe10_layer=SPE10_LAYER,
                        spe10_water_density=water_density,  # kg/m^3
                    )
                )

    return cases


studies: dict[str, list[SimulationConfig]] = {
    "viscous_varying_rp_init_s_02": generate_viscous_varying_rp_cases(init_s=0.2),
    "viscous_varying_rp_init_s_03": generate_viscous_varying_rp_cases(init_s=0.3),
    "viscous_varying_init_s": generate_viscous_varying_init_s_cases(),
    "gravity_separation": generate_gravity_separation_cases(),
    "capillary_varying_rp": generate_capillary_varying_rp(),
    "capillary_varying_init_s": generate_capillary_varying_init_s(),
    "capillary_varying_entry_pressure": generate_capillary_varying_entry_pressure(),
    "buoyancy_varying_rp": generate_buoyancy_varying_rp(),
    "buoyancy_varying_density": generate_buoyancy_varying_density(),
}

# endregion

if __name__ == "__main__":
    results_dir.mkdir(exist_ok=True)
    for study in studies.values():
        for config in study:
            run_simulation(config)
            clean_up_after_simulation(config)
