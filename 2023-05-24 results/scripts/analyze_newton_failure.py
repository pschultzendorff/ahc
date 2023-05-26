import logging
import math
import os

import numpy as np
import porepy as pp
from tpf_lab.models.buckley_leverett import BuckleyLeverettSetup


# Fix seed for reproducability.
np.random.seed(0)

# Setup logging.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
LIMIT_REL_PERM = True

INFLUX = 1.0
ANGLE = math.pi / 4

# This will be changed based on the CFL condition
DEFAULT_MAX_TIME_STEP: float = 0.1


# Set up folder and files for logging/plots/saved time steps.
foldername: str = os.path.join(
    "results",
    "buckley_leverett",
    "analyze_Newton_failure",
    "linear rel perm",
    f"NEWTON_ITERATIONS_{MAX_NEWTON_ITERATIONS}",
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
    "limit_rel_perm": True,
    # Buckley-Leverett params
    "angle": ANGLE,
    "influx": INFLUX,
    # Linear flow function. Necessary parameter for the analytical solver.
    "linear_flow": True,
}

params.update(
    {
        "folder_name": foldername,
        "meshing_arguments": {
            "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
        },
        "time_manager": pp.TimeManager(
            schedule=np.array([0, 1]),
            dt_init=0.5,
            constant_dt=True,
        ),
        "limit_rel_perm": True,
        "export_each_iteration": True,
    }
)
# model = BuckleyLeverettSetup(params)
# try:
#     pp.models.run_models.run_time_dependent_model(
#         model, {"max_iterations": MAX_NEWTON_ITERATIONS}
#     )
# except Exception as e:
#     print(e)

####################
# Shorter time step
###################
foldername = os.path.join(
    "results",
    "buckley_leverett",
    "analyze_Newton_failure",
    "linear_rel_perm_dt_0.1",
    f"NEWTON_ITERATIONS_{MAX_NEWTON_ITERATIONS}",
)

params.update(
    {
        "folder_name": foldername,
        "meshing_arguments": {
            "cell_size": DEFAULT_PHYS_SIZE / float(DEFAULT_NUM_GRID_CELLS)
        },
        "time_manager": pp.TimeManager(
            schedule=np.array([0, 0.1]),
            dt_init=0.1,
            constant_dt=True,
        ),
        "limit_rel_perm": True,
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


###################
# Power rel. perms.
###################
# Set up folder and files for logging/plots/saved time steps.
foldername = os.path.join(
    "results",
    "buckley_leverett",
    "analyze_Newton_failure",
    "Corey_rel_perm",
    f"NEWTON_ITERATIONS_{MAX_NEWTON_ITERATIONS}",
)
try:
    os.makedirs(foldername)
except Exception:
    pass

REL_PERM_MODEL = "power"
params.update(
    {
        "rel_perm_model": REL_PERM_MODEL,
        "linear_flow": False,
        "folder_name": foldername,
    }
)

# model = BuckleyLeverettSetup(params)
# try:
#     pp.models.run_models.run_time_dependent_model(
#         model, {"max_iterations": MAX_NEWTON_ITERATIONS}
#     )
# except Exception:
#     pass
