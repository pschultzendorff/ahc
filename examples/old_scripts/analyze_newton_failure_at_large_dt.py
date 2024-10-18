import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import porepy as pp
from tpf_lab.models.buckley_leverett import (  # type: ignore
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettDataSaving,
    BuckleyLeverettDefaultGeometry,
    BuckleyLeverettEquations,
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettSolutionStrategy,
    DiagnosticsMixinExtended,
    VariablesTPF,
    VerificationUtils,
)

# Fix seed for reproducability.
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BuckleyLeverettModifiedSolutionStrategy(BuckleyLeverettSolutionStrategy):
    def assemble_linear_system(self) -> None:
        super().assemble_linear_system()
        # logger.info(f"A {self.linear_system[0][:]}")
        # logger.info(f"b {self.linear_system[1]}")

    def before_nonlinear_iteration(self) -> None:
        upwind_w = pp.ad.UpwindAd(self.w_flux_key, self.mdg.subdomains())
        # logger.info(f"upwind: {upwind_w.upwind.evaluate(self.equation_system)}")
        return super().before_nonlinear_iteration()

    def after_nonlinear_iteration(self, solution: np.ndarray) -> None:
        # Show total flux
        # flux_t = self._flux_t(self.mdg.subdomains())
        # logger.info(f"flux_t: {flux_t.evaluate(self.equation_system).val}")
        # rel_perm_w = self._rel_perm_w()
        # logger.info(f"rel_perm_w: {rel_perm_w.evaluate(self.equation_system).val}")
        # rel_perm_n = self._rel_perm_n()
        # logger.info(f"rel_perm_n: {rel_perm_n.evaluate(self.equation_system).val}")
        # s_normalized = self._s_normalized()
        # logger.info(
        #     f"normalized_saturation: {s_normalized.evaluate(self.equation_system).val}"
        # )

        return super().after_nonlinear_iteration(solution)


class BuckleyLeverettSetup(  # type: ignore
    BuckleyLeverettEquations,
    VariablesTPF,
    BuckleyLeverettBoundaryConditions,
    BuckleyLeverettModifiedSolutionStrategy,
    #
    BuckleyLeverettDefaultGeometry,
    #
    BuckleyLeverettSemiAnalyticalSolution,
    BuckleyLeverettDataSaving,
    VerificationUtils,
    DiagnosticsMixinExtended,
): ...


####################
# Default parameters
####################
MAX_NEWTON_ITERATIONS = 30

DEFAULT_NUM_GRID_CELLS = 200
DEFAULT_PHYS_SIZE = 20
# Default grid boundaries for the BuckleyLeverett class
XMIN = -10
XMAX = 10

POROSITY = 1.0
VISCOSITY_W = 1.0
VISCOSITY_N = 1.0
DENSITY_W = 1.0
DENSITY_N = 1.0

RESIDUAL_SATURATION_W = 0.3
RESIDUAL_SATURATION_N = 0.3

REL_PERM_MODEL = "linear"
REL_PERM_LINEAR_PARAM_W = 1.0
REL_PERM_LINEAR_PARAM_N = 1.0
# Do not limit the rel. perm. to avoid the solver crashing!
LIMIT_REL_PERM = False
INFLUX = 1.0
ANGLE = math.pi / 4

DT = 1.0

# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "buckley_leverett",
    "analyze_Newton_failure",
    f"linear_rel_perm_dt_{DT}_rel_perm_limited_{str(LIMIT_REL_PERM)}",
    f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
)
try:
    os.makedirs(foldername)
except Exception:
    pass


####################
# Linear rel. perms.
####################
params = {
    "progressbars": True,
    "formulation": "n_pressure_w_saturation",
    # grid
    "meshing_arguments": {
        "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
    },
    "phys size": DEFAULT_PHYS_SIZE,
    "time_manager": pp.TimeManager(
        schedule=np.array([0, 1.0]),
        dt_init=DT,
        constant_dt=True,
    ),
    # fluid and solid params
    "porosity": POROSITY,
    "viscosity_w": VISCOSITY_W,
    "viscosity_n": VISCOSITY_N,
    "density_w": DENSITY_W,
    "density_n": DENSITY_N,
    "S_M": 1 - RESIDUAL_SATURATION_W,
    "S_m": RESIDUAL_SATURATION_N,
    "residual_saturation_w": RESIDUAL_SATURATION_W,
    "residual_saturation_n": RESIDUAL_SATURATION_N,
    # rel. perm model
    "rel_perm_model": REL_PERM_MODEL,
    "rel_perm_linear_param_w": REL_PERM_LINEAR_PARAM_W,
    "rel_perm_linear_param_n": REL_PERM_LINEAR_PARAM_N,
    "limit_rel_perm": LIMIT_REL_PERM,
    # Buckley-Leverett params
    "angle": ANGLE,
    "influx": INFLUX,
    # Linear flow function. Necessary parameter for the analytical solver.
    "linear_flow": True,
}

params.update(
    {
        "folder_name": foldername,
        "export_each_iteration": True,
    }
)
model = BuckleyLeverettSetup(params)

try:
    pp.models.run_models.run_time_dependent_model(
        model, {"max_iterations": MAX_NEWTON_ITERATIONS}
    )
except Exception as e:
    print(e)


####################
# Power rel. perms.
#####################
# foldername = os.path.join(
#     "results",
#     "buckley_leverett",
#     "analyze_Newton_failure",
#     f"power_rel_perm_dt_{DT}_rel_perm_limited_{str(LIMIT_REL_PERM)}",
#     f"max_newton_iterations_{MAX_NEWTON_ITERATIONS}",
# )
# try:
#     os.makedirs(foldername)
# except Exception:
#     pass

# REL_PERM_MODEL = "power"
# params.update(
#     {
#         "rel_perm_model": REL_PERM_MODEL,
#         "folder_name": foldername,
#         "linear_flow": False,
#     }
# )
# model = BuckleyLeverettSetup(params)

# try:
#     pp.models.run_models.run_time_dependent_model(
#         model, {"max_iterations": MAX_NEWTON_ITERATIONS}
#     )
# except Exception as e:
#     print(e)
