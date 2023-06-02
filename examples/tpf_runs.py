"""Some test runs with the ``two_phase_flow_model.TwoPhaseFlow`` class. If not noted
otherwise, the runs were performed on a 2D grid with 20x20 cells.
"""
import os
from typing import Optional, Union

import numpy as np

import porepy as pp
from porepy.models.run_models import run_time_dependent_model
from src.tpf_lab.models.two_phase_flow import TwoPhaseFlowSetup


# Simple test run
# Neumann bc on three sides, Dirichlet bc on one side.
model = TwoPhaseFlowSetup(
    {
        "file_name": "simple",
        "folder_name": os.path.join("two_phase_flow_setup_run", "simple"),
    }
)

model._schedule: np.ndarray = np.array([0, 0.2])
# run_time_dependent_model(model, {})


class TwoPhaseFlow_Dirichlet(TwoPhaseFlowEquations):
    def _bc_type(self, g: pp.Grid) -> pp.BoundaryCondition:
        """Homogeneous Dirichlet conditions on all external boundaries."""
        all_bf, *_ = self._domain_boundary_sides(g)
        return pp.BoundaryCondition(g, all_bf, "dir")


# Simple test run
model = TwoPhaseFlow_Dirichlet(
    {
        "file_name": "dirichlet",
        "folder_name": os.path.join("two_phase_flow_runs", "dirichlet"),
    }
)
# run_time_dependent_model(model, {})


# Test run with wetting source
class TwoPhaseFlow_WSource(TwoPhaseFlowEquations):
    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[209] = 0.2
        return array


model = TwoPhaseFlow_WSource(
    {
        "file_name": "w_source_neu",
        "folder_name": os.path.join("two_phase_flow_runs", "w_source_neu"),
    }
)
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-5,
#         "nl_divergence_tol": 1e5,
#     },
# )


class TwoPhaseFlow_WSource_Dir(TwoPhaseFlow_Dirichlet):
    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[209] = 0.2
        return array


model = TwoPhaseFlow_WSource_Dir(
    {
        "file_name": "w_source_dir",
        "folder_name": os.path.join("two_phase_flow_runs", "w_source_dir"),
    }
)
run_time_dependent_model(
    model,
    {
        "max_iterations": 30,
        "nl_convergence_tol": 1e-5,
        "nl_divergence_tol": 1e5,
    },
)


# Test run with non-wetting source
class TwoPhaseFlow_NWSource(TwoPhaseFlowEquations):
    def _n_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_n(g)
        array[209] = 0.2
        return array


model = TwoPhaseFlow_NWSource(
    {
        "file_name": "nw_source_neu",
        "folder_name": os.path.join("two_phase_flow_runs", "nw_source_neu"),
    }
)
# run_time_dependent_model(model, {})


model = TwoPhaseFlow_NWSource(
    {
        "file_name": "nw_source_neu_long_inj",
        "folder_name": os.path.join("two_phase_flow_runs", "nw_source_neu_long_inj"),
    }
)
model._schedule: np.ndarray = np.array([0, 100.0])
# run_time_dependent_model(model, {})


class TwoPhaseFlow_NWSource_Dir(TwoPhaseFlow_Dirichlet):
    def _n_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[209] = 0.2
        return array


model = TwoPhaseFlow_NWSource_Dir(
    {
        "file_name": "nw_source_dir",
        "folder_name": os.path.join("two_phase_flow_runs", "nw_source_dir"),
    }
)
# run_time_dependent_model(model, {})


# Longer injection
class TwoPhaseFlow_LongInj(TwoPhaseFlow_WSource):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # Let the model run for a longer time
        self._time_step: float = 0.2
        self._schedule: np.ndarray = np.array([0, 100.0])


model = TwoPhaseFlow_LongInj(
    {
        "file_name": "long_inj_neu",
        "folder_name": os.path.join("two_phase_flow_runs", "long_inj_neu"),
    }
)
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-4,
#         "nl_divergence_tol": 1e5,
#     },
# )


class TwoPhaseFlow_LongInj_Dir(TwoPhaseFlow_WSource_Dir):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # Let the model run for a longer time
        self._time_step: float = 0.2
        self._schedule: np.ndarray = np.array([0, 150.0])


model = TwoPhaseFlow_LongInj_Dir(
    {
        "file_name": "long_inj_dir_sat_growth_limit",
        "folder_name": os.path.join(
            "two_phase_flow_runs", "long_inj_dir_sat_growth_limit"
        ),
    }
)
model._limit_saturation_change = True
# run_time_dependent_model(model, {})


# Test run with extraction


# Neumann conditions do not converge
class TwoPhaseFlow_Ext(TwoPhaseFlowEquations):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # Let the model run for a longer time
        self._time_step: float = 0.2
        self._schedule: np.ndarray = np.array([0, 10.0])

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[309] = -0.2
        return array


model = TwoPhaseFlow_Ext(
    {
        "file_name": "ext_neu",
        "folder_name": os.path.join("two_phase_flow_runs", "ext_neu"),
    }
)
# run_time_dependent_model(model, {})


class TwoPhaseFlow_Ext_Dir(TwoPhaseFlow_Dirichlet):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # Let the model run for a longer time
        self._time_step: float = 0.2
        self._schedule: np.ndarray = np.array([0, 10.0])

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[309] = -0.2
        return array


model = TwoPhaseFlow_Ext_Dir(
    {
        "file_name": "ext_dir",
        "folder_name": os.path.join("two_phase_flow_runs", "ext_dir"),
    }
)
# run_time_dependent_model(model, {})


# Test run with injection and extraction at the same time
class TwoPhaseFlow_InjExt(TwoPhaseFlowEquations):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # Let the model run for a longer time
        self._time_step: float = 0.1
        self._schedule: np.ndarray = np.array([0, 10.0])

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[109] = 0.2
        array[309] = -0.2
        return array


model = TwoPhaseFlow_InjExt(
    {
        "file_name": "inj_ext_neu",
        "folder_name": os.path.join("two_phase_flow_runs", "inj_ext_neu"),
    }
)
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-4,
#         "nl_divergence_tol": 1e5,
#     },
# )


class TwoPhaseFlow_InjExt_Dir(TwoPhaseFlow_Dirichlet):
    def __init__(self, params: Optional[dict] = None) -> None:
        super().__init__(params)
        # Let the model run for a longer time
        self._time_step: float = 0.1
        self._schedule: np.ndarray = np.array([0, 10.0])

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[109] = 0.2
        array[309] = -0.2
        return array


model = TwoPhaseFlow_InjExt_Dir(
    {
        "file_name": "inj_ext_dir",
        "folder_name": os.path.join("two_phase_flow_runs", "inj_ext_dir"),
    }
)
# run_time_dependent_model(model, {})


# Test run with more complicated geometry
class TwoPhaseFlow_Geom(TwoPhaseFlow_WSource):
    def _permeability(self, g: pp.Grid) -> np.ndarray:
        """Low permeability regions in the upper and lower half, high permeability
        region in the left half.
        """
        array: np.ndarray = super()._permeability(g)
        array[265:275] = 1.0
        array[125:135] = 20
        return array


model = TwoPhaseFlow_Geom(
    {
        "file_name": "complex_geom",
        "folder_name": os.path.join("two_phase_flow_runs", "complex_geom"),
    }
)
model._schedule = [0, 100.0]
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-4,
#         "nl_divergence_tol": 1e5,
#     },
# )


# Test run on triangle grid
class TwoPhaseFlow_TriangleGrid(TwoPhaseFlow_WSource):
    def create_grid(self) -> None:
        GRID_SIZE: int = 20
        PHYS_SIZE: int = 2
        cell_dims: np.ndarray = np.array(
            [
                GRID_SIZE,
                GRID_SIZE,
            ]
        )
        phys_dims: np.ndarray = np.array(
            [
                PHYS_SIZE,
                PHYS_SIZE,
            ]
        )
        g_cart: pp.CartGrid = pp.CartGrid(cell_dims, phys_dims)
        g_tetra: pp.TriangleGrid = pp.TriangleGrid(g_cart.nodes[:2])
        g_tetra.compute_geometry()
        self.mdg: pp.MixedDimensionalGrid = pp.meshing.subdomains_to_mdg([[g_tetra]])
        self.box: dict = pp.bounding_box.from_points(
            np.array(
                [
                    [
                        0,
                        0,
                    ],
                    [
                        GRID_SIZE,
                        GRID_SIZE,
                    ],
                ]
            ).T
        )


model = TwoPhaseFlow_TriangleGrid(
    {
        "file_name": "triangle_grid_w_source",
        "folder_name": os.path.join("two_phase_flow_runs", "triangle_grid_w_source"),
    }
)
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-5,
#         "nl_divergence_tol": 1e5,
#     },
# )
