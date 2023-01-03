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
        model = BaseNN()
        model.load_state_dict(
            torch.load(os.path.join("saved_models", "BaseNN_RelPerm_Wetting.pt"))
        )
        nn_func = pp.ad.Function(nn_wrapper(model), "w_nn")
        return nn_func(s)

    def _nw_rel_perm(self) -> pp.ad.Operator:
        s = self._ad.saturation
        model = BaseNN()
        model.load_state_dict(
            torch.load(os.path.join("saved_models", "BaseNN_RelPerm_NonWetting.pt"))
        )
        nn_func = pp.ad.Function(nn_wrapper(model), "nw_nn")
        return nn_func(s)


# Simple test run
model = TwoPhaseFlow_DataRelPerm(
    {
        "file_name": "DataRelPerm",
        "folder_name": os.path.join("two_phase_flow_runs", "DataRelPerm"),
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


class TwoPhaseFlow_WSource_Dir(TwoPhaseFlow_DataRelPerm):
    def _w_source(self, g: pp.Grid) -> np.ndarray:
        array: np.ndarray = super()._w_source(g)
        array[209] = 1
        return array


model = TwoPhaseFlow_WSource_Dir(
    {
        "file_name": "DataRelPerm_w_source_dir",
        "folder_name": os.path.join("two_phase_flow_runs", "DataRelPerm_w_source_dir"),
    }
)
# run_time_dependent_model(
#     model,
#     {
#         "max_iterations": 30,
#         "nl_convergence_tol": 1e-8,
#         "nl_divergence_tol": 1e5,
#     },
# )
