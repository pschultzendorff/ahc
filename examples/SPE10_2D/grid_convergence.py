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
from collections import defaultdict
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from numba import config
from tpf.models.error_estimate import (
    DataSavingMixinEst,
    ErrorEstimateMixin,
    SolutionStrategyEst,
    SolverStatisticsEst,
)
from tpf.models.flow_and_transport import (
    BoundaryConditionsTPF,
    ConstitutiveLawsTPF,
    DataSavingTPF,
    EquationsTPF,
    SolutionStrategyTPF,
    VariablesTPF,
)
from tpf.models.phase import FluidPhase
from tpf.models.reconstruction import (
    DataSavingMixinRec,
    EquilibratedFluxMixin,
    GlobalPressureMixin,
    PressureReconstructionMixin,
)
from tpf.spe10.mixin import SPE10Mixin
from tpf.utils.constants_and_typing import (
    COMPLIMENTARY_PRESSURE,
    FEET,
    GLOBAL_PRESSURE,
    PSI,
)

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
class ModifiedGeometry(pp.ModelGeometry):

    def set_domain(self) -> None:
        r"""Single layer of the SPE10 problem 2 model. Extend of the full domain is
        :math:`\qty{1200 x 2200 x 170}{\feet}`. A single layer is
        :math:`\qty{1200 x 2200}{\feet}`.

        """
        bounding_box: dict[str, pp.number] = {
            "xmin": 0,
            "xmax": 600 * FEET,
            "ymin": 0,
            "ymax": 1100 * FEET,
        }
        self._domain = pp.Domain(bounding_box)


class ModifiedEquations(EquationsTPF):

    def phase_fluid_source(self, g: pp.Grid, phase: FluidPhase) -> np.ndarray:
        r"""Volumetric phase source term. Given as volumetric flux.

        Five-spot setup. Water (wetting) injection in the center, oil (nonwetting)
        production in the four corners.

        NOTE: This is the average value per grid cell, i.e., it gets scaled with the
        cell volume in the equation.

        SI Units: m^d/(m^(d-1)*s) -> Depends on the units of the other parameters.

        """
        if phase.name == self.wetting.name:
            array: np.ndarray = super().phase_fluid_source(g, phase)
            if np.isclose(
                self.params["meshing_arguments"]["cell_size"], 600 * FEET / 7.5
            ):
                array[128] = 87.5 / pp.DAY  # 87.5 m^3/day in [m^3/s]
            elif np.isclose(
                self.params["meshing_arguments"]["cell_size"], 600 * FEET / 15
            ):
                array[602] = 87.5 / pp.DAY  # 87.5 m^3/day in [m^3/s]
            elif np.isclose(
                self.params["meshing_arguments"]["cell_size"], 600 * FEET / 30
            ):
                array[1358] = 87.5 / pp.DAY  # 87.5 m^3/day in [m^3/s]
            elif np.isclose(
                self.params["meshing_arguments"]["cell_size"], 600 * FEET / 60
            ):
                array[5173] = 87.5 / pp.DAY  # 87.5 m^3/day in [m^3/s]
            elif np.isclose(
                self.params["meshing_arguments"]["cell_size"], 600 * FEET / 120
            ):
                array[34294] = 87.5 / pp.DAY  # 87.5 m^3/day in [m^3/s]
            return array
        elif phase.name == self.nonwetting.name:
            return super().phase_fluid_source(g, phase)

    def corner_cell_ids(self, g: pp.Grid) -> list[int]:
        """Get the corner cell ids."""
        if np.isclose(self.params["meshing_arguments"]["cell_size"], 600 * FEET / 7.5):
            return [212, 213, 219, 220]
        elif np.isclose(self.params["meshing_arguments"]["cell_size"], 600 * FEET / 15):
            return [857, 904, 905, 906]
        elif np.isclose(self.params["meshing_arguments"]["cell_size"], 600 * FEET / 30):
            return [3616, 3618, 3622, 3624]
        elif np.isclose(self.params["meshing_arguments"]["cell_size"], 600 * FEET / 60):
            return [14957, 14958, 14960, 14993]
        elif np.isclose(
            self.params["meshing_arguments"]["cell_size"], 600 * FEET / 120
        ):
            return [60532, 60485, 60487, 60490]

    def corner_masks(self, g: pp.Grid) -> tuple[pp.ad.DenseArray, pp.ad.DenseArray]:
        """Create masks that hide and single out the corner cells."""
        corner_cell_ids: list[int] = self.corner_cell_ids(g)
        corner_mask_ndarray: np.ndarray = np.zeros((g.num_cells))
        corner_mask_ndarray[corner_cell_ids] = 1
        corner_mask = pp.ad.DenseArray(corner_mask_ndarray)
        corner_mask_inverse = pp.ad.DenseArray(1 - corner_mask_ndarray)
        corner_mask.set_name("Corner mask")
        corner_mask_inverse.set_name("Corner mask inverse")
        return corner_mask, corner_mask_inverse

    def set_equations(self, equation_names: Optional[dict[str, str]] = None) -> None:
        """Modify the equations s.t. the corner cells get prescibed a pressure and
        saturation explicitly. This simulates production wells.

        """
        super().set_equations(equation_names)

        # Prescibre the corner cell values directly. This resembles Dirichlet
        # conditions.
        g: pp.Grid = self.mdg.subdomains()[0]
        flow_equation: pp.ad.Operator = self.equation_system.equations["Flow equation"]
        transport_equation: pp.ad.Operator = self.equation_system.equations[
            "Transport equation"
        ]

        corner_mask, corner_mask_inverse = self.corner_masks(g)

        # Subdivide new equations in 3 parts for easier debugging.
        old_flow_equation_masked: pp.ad.Operator = corner_mask_inverse * flow_equation
        old_transport_equation_masked: pp.ad.Operator = (
            corner_mask_inverse * transport_equation
        )
        old_flow_equation_masked.set_name("Old flow equation masked")
        old_transport_equation_masked.set_name("Old transport equation masked")

        explicit_pressure: pp.ad.Operator = corner_mask * (
            self.nonwetting.p - pp.ad.Scalar(4000 * PSI)
        )
        explicit_saturation: pp.ad.Operator = corner_mask * (
            self.wetting.s - pp.ad.Scalar(1 - self.wetting.residual_saturation)
        )
        explicit_pressure.set_name("Explicit pressure")
        explicit_saturation.set_name("Explicit saturation")

        # Attach the pressure values to the flow equation and the saturation values to
        # the transport equation.
        new_flow_equation: pp.ad.Operator = old_flow_equation_masked + explicit_pressure
        new_transport_equation: pp.ad.Operator = (
            old_transport_equation_masked + explicit_saturation
        )
        new_flow_equation.set_name("Flow equation")
        new_transport_equation.set_name("Transport equation")

        self.equation_system.equations["Flow equation"] = new_flow_equation
        self.equation_system.equations["Transport equation"] = new_transport_equation

    def divergence_mismatch(self) -> None:
        r"""Modify the check for divergence mismatch of the equilibrated fluxes
        s.t. the mismatch is zero in the corners.

        """
        g, g_data = self.mdg.subdomains(return_data=True)[0]

        corner_mask_inverse: pp.ad.DenseArray = self.corner_masks(g)[1]

        # Spatial discretization operators.
        div = pp.ad.Divergence([g])
        flux_mpfa = pp.ad.MpfaAd(self.flux_key, [g])

        # Time derivatives.
        dt = pp.ad.Scalar(self.time_manager.dt)
        dt_s: pp.ad.Operator = pp.ad.time_derivatives.dt(self.wetting.s, dt)

        # Ad source.
        source_ad_w = pp.ad.DenseArray(self.phase_fluid_source(g, self.wetting))
        source_ad_t = pp.ad.DenseArray(self.total_fluid_source(g))

        # Ad parameters.
        porosity_ad = pp.ad.DenseArray(self.porosity(g))

        # Compute cap pressure and relative permeabilities.
        p_cap = self.cap_press(self.wetting.s)
        # p_cap_bc = pp.ad.DenseArray(self._bc_values_cap_press(g))

        mobility_w = self.phase_mobility(g, self.wetting)
        mobility_n = self.phase_mobility(g, self.nonwetting)
        mobility_t = self.total_mobility(g)

        # Ad equations
        if self.formulation == "fractional_flow":
            # Note, that for ``flux_t``, the total mobility is already included.
            flux_t = pp.ad.DenseArray(
                pp.get_solution_values(
                    "total_flux_equilibrated", g_data, iterate_index=0
                )
            )
            flux_w = pp.ad.DenseArray(
                pp.get_solution_values(
                    f"{self.wetting.name}_flux_equilibrated", g_data, iterate_index=0
                )
            )
            fractional_flow_w = mobility_w / mobility_t
            vector_source_w = pp.ad.DenseArray(self.vector_source(g, self.wetting))
            vector_source_n = pp.ad.DenseArray(self.vector_source(g, self.nonwetting))

            flow_equation = (
                div @ flux_t - self.volume_integral(source_ad_t, [g], 1)
            ) * corner_mask_inverse
            transport_equation_ff = (
                porosity_ad * (self.volume_integral(dt_s, [g], 1))
                + div
                @ (
                    fractional_flow_w * flux_t
                    + fractional_flow_w
                    * mobility_n
                    * (
                        flux_mpfa.flux() @ p_cap
                        # TODO: Plus boundary values here or are they included in the total flux?
                        + flux_mpfa.vector_source() @ vector_source_w
                        - flux_mpfa.vector_source() @ vector_source_n
                    )
                )
                - self.volume_integral(source_ad_w, [g], 1)
            ) * corner_mask_inverse
            transport_equation_wf = (
                porosity_ad * (self.volume_integral(dt_s, [g], 1))
                + div @ flux_w
                - self.volume_integral(source_ad_w, [g], 1)
            ) * corner_mask_inverse

        flow_equation.set_name("Flow reconstruction mismatch")
        transport_equation_ff.set_name(
            "Transport reconstruction fractional flow mismatch"
        )
        transport_equation_wf.set_name("Transport reconstruction wetting flow mismatch")
        logger.info(
            f"Flow equation mismatch {np.sum(np.abs(flow_equation.value(self.equation_system)))}"
        )
        logger.info(
            f"Transport equation mismatch fractional flow {np.sum(np.abs(transport_equation_ff.value(self.equation_system)))}"
        )
        logger.info(
            f"Transport equation mismatch wetting flow {np.sum(np.abs(transport_equation_wf.value(self.equation_system)))}"
        )


class ModifiedBoundaryConditions(BoundaryConditionsTPF):

    def bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """BC type (Dirichlet or Neumann).

        We assign Neumann conditions for all faces. The four corner cells get prescribed
        a pressure explicitely, which acts as a Dirichlet condition.

        """
        return pp.BoundaryCondition(g)


class ModifiedSolutionStrategyMixin:

    mdg: pp.MixedDimensionalGrid
    wetting: FluidPhase
    nonwetting: FluidPhase
    corner_cell_ids: Callable[[pp.Grid], list[int]]

    def initial_condition(self) -> None:
        """Set initial values for pressure and saturation.

        The corner cells get prescibed the right values immediately. Inside the
        reservoir, the initial pressure is higher. The initial saturation is set to the
        residual wetting saturation + 0.1 inside the reservoir.

        """
        g: pp.Grid = self.mdg.subdomains()[0]
        corner_cell_ids: list[int] = self.corner_cell_ids(g)

        initial_pressure = np.full(g.num_cells, 6000 * PSI)
        initial_pressure[corner_cell_ids] = 4000 * PSI
        initial_saturation = np.full(g.num_cells, 0.4)
        initial_saturation[corner_cell_ids] = 1 - self.wetting.residual_saturation
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


class ModifiedTPF(
    # SPE10:
    SPE10Mixin,
    # Modified model:
    ModifiedEquations,
    VariablesTPF,
    ConstitutiveLawsTPF,
    ModifiedGeometry,
    ModifiedBoundaryConditions,
    ModifiedSolutionStrategyMixin,
    # Estimator mixins:
    ErrorEstimateMixin,
    SolutionStrategyEst,
    DataSavingMixinEst,
    # Reconstruction mixins:
    GlobalPressureMixin,
    PressureReconstructionMixin,
    EquilibratedFluxMixin,
    DataSavingMixinRec,
    # Base data saving:
    SolutionStrategyTPF,
    DataSavingTPF,
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
    600 * FEET / 15,
    600 * FEET / 30,
    600 * FEET / 60,
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
    continue
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
    model = ModifiedTPF(params)
    try:
        pp.run_time_dependent_model(model=model, params=params)
    except Exception as error:
        logger.error(f"Model {model} failed with error: {error}")

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
fig.savefig(
    pathlib.Path(__file__).parent / "grid_convergence" / "solver_convergence.png"
)

# endregion
