r"""Benchmark for error estimates. Run a single time step.

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
    - Saturation: residual water saturation (0.2).
- Rel. perm. models:
    - linear
    - Corey with power 2.
- Capillary pressure model:
    - Brooks-Corey
- Time step size is kept constant s.t. the discretization error varies only with grid
  size.

"""

import logging
import os
import pathlib
import shutil
import subprocess
import warnings

import numpy as np
import porepy as pp
from numba import config
from tpf.derived_models.spe10 import SPE10Mixin
from tpf.models.error_estimate import TwoPhaseFlowErrorEstimate
from tpf.models.flow_and_transport import TwoPhaseFlow
from tpf.utils.constants_and_typing import FEET, PSI
from tpf.viz.iteration_exporting import IterationExportingMixin
from tpf.viz.solver_statistics import SolverStatisticsEst
from viztracer import VizTracer  # type: ignore[import]

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
# NOTE Logging creates a small overhead. To gain some speedup, set the level to WARNING.
logger.setLevel(logging.WARNING)

# endregion


# region MODEL


class Benchmark(
    IterationExportingMixin,
    SPE10Mixin,
    TwoPhaseFlowErrorEstimate,
): ...  # type: ignore


# endregion

# region RUN
foldername: pathlib.Path = pathlib.Path(__file__).parent / "benchmark"

params = {
    "folder_name": foldername,
    "file_name": "setup",
    "solver_statistics_file_name": foldername / "solver_statistics.json",
    "meshing_arguments": {"cell_size": 600 * FEET / 30},
    "progressbars": True,
    # Model:
    "formulation": "fractional_flow",
    "material_constants": {},
    "rel_perm_constants": {"model": "linear", "limit": False},
    "cap_press_constants": {"model": "linear", "linear_param": 0.1 * PSI},
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 1.0 * pp.DAY]),
        dt_init=1.0 * pp.DAY,
        constant_dt=True,
    ),
    "grid_type": "simplex",
    "spe10_quarter_domain": True,
    "spe10_layer": 80 - 1,
    "spe10_isotropic_perm": True,
    # Nonlinear params:
    "nonlinear_solver_statistics": SolverStatisticsEst,
    "nonlinear_solver": pp.NewtonSolver,
    "max_iterations": 20,
    "nl_convergence_tol": 1e-5,
    "nl_divergence_tol": 1e180,
}


try:
    shutil.rmtree(foldername)
except Exception:
    pass
foldername.mkdir(parents=True)

tracer = VizTracer(
    min_duration=1e3,  # μs
    ignore_c_function=True,
    ignore_frozen=True,
)
tracer.start()

model = Benchmark(params)
pp.run_time_dependent_model(model=model, params=params)

tracer.stop()

# Save the results and open them in a browser with vizviewer.
results_path: pathlib.Path = foldername / "viztracer.json"
tracer.save(str(results_path))
subprocess.run(["vizviewer", "--port", "9002", results_path])

# endregion
