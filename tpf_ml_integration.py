"""Some test runs with the ``two_phase_flow_model.TwoPhaseFlow`` class. If not noted
otherwise, the runs were performed on a 2D grid with 20x20 cells.
"""
import os
from typing import Optional, Union
import torch
import numpy as np

import porepy as pp
from src.tpf_lab.models.run_models import run_time_dependent_model
from src.tpf_lab.models.two_phase_flow_model import TwoPhaseFlow
from src.tpf_lab.ml.pp_nn import BaseNN
from src.tpf_lab.ml.ml_ad import nn_wrapper


class TwoPhaseFlow_DataRelPerm(TwoPhaseFlow):
    def _w_rel_perm(self) -> pp.ad.Operator:
        """Wetting phase relative permeability pressure computed with a nn."""
        s = self._ad.saturation
        model = BaseNN({"depth": 1, "act": "linear"})
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "saved_models",
                    "BaseNN_RelPermW_BrooksCorey_1HiddenLayers.pt",
                )
            )
        )
        nn_func = pp.ad.Function(nn_wrapper(model), "w_nn")
        return nn_func(s)

    def _n_rel_perm(self) -> pp.ad.Operator:
        s = self._ad.saturation
        model = BaseNN({"depth": 1, "act": "linear"})
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "saved_models", "BaseNN_RelPermN_BrooksCorey_1HiddenLayers.pt"
                )
            )
        )
        nn_func = pp.ad.Function(nn_wrapper(model), "nw_nn")
        return nn_func(s)

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[209] = 0.3
        return array


# Simple test run
model = TwoPhaseFlow_DataRelPerm(
    {
        "file_name": "RelPerm_BrooksCorey_Data",
        "folder_name": os.path.join("two_phase_flow_runs", "RelPerm_BrooksCorey_Data"),
    }
)
model._schedule = [0, 100.0]
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-8,
#         "nl_divergence_tol": 1e5,
#     },
# )


class TwoPhaseFlow_DataRelPerm(TwoPhaseFlow):
    def _n_rel_perm(self) -> pp.ad.Operator:
        s = self._ad.saturation
        model = BaseNN({"depth": 1, "act": "linear"})
        model.load_state_dict(
            torch.load(
                os.path.join(
                    "saved_models", "BaseNN_RelPermN_BrooksCorey_1HiddenLayers.pt"
                )
            )
        )
        nn_func = pp.ad.Function(nn_wrapper(model), "nw_nn")
        return nn_func(s)

    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[209] = 0.3
        return array


# Simple test run
model = TwoPhaseFlow_DataRelPerm(
    {
        "file_name": "RelPerm_BrooksCorey_DataN_FormulaW",
        "folder_name": os.path.join(
            "two_phase_flow_runs", "RelPerm_BrooksCorey_DataN_FormulaW"
        ),
    }
)
run_time_dependent_model(
    model,
    {
        "max_iterations": 30,
        "nl_convergence_tol": 1e-8,
        "nl_divergence_tol": 1e5,
    },
)


class TwoPhaseFlow_WSource(TwoPhaseFlow):
    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._source_w(g)
        array[209] = 0.3
        return array


model = TwoPhaseFlow_WSource(
    {
        "file_name": "RelPerm_BrooksCorey_Formula",
        "folder_name": os.path.join(
            "two_phase_flow_runs", "RelPerm_BrooksCorey_Formula"
        ),
    }
)
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-10,
#         "nl_divergence_tol": 1e5,
#     },
# )
