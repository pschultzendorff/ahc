"""Some test runs with the ``two_phase_flow_model.TwoPhaseFlow`` class. If not noted
otherwise, the runs were performed on a 2D grid with 20x20 cells.
"""
import os
from typing import Optional, Union
import torch
import numpy as np

import porepy as pp
from porepy.models.run_models import run_time_dependent_model
from src.porepy_adaptions.models.two_phase_flow_model import TwoPhaseFlow
from src.porepy_adaptions.ml.pp_nn import BaseNN
from src.porepy_adaptions.ml.ml_ad import nn_wrapper


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

    # def _nw_rel_perm(self) -> pp.ad.Operator:
    #     s = self._ad.saturation
    #     model = BaseNN()
    #     model.load_state_dict(
    #         torch.load(os.path.join("saved_models", "BaseNN_RelPerm_NonWetting.pt"))
    #     )
    #     nn_func = pp.ad.Function(nn_wrapper(model), "nw_nn")
    #     return nn_func(s)


# Simple test run
model = TwoPhaseFlow_DataRelPerm(
    {
        "file_name": "DataRelPerm",
        "folder_name": os.path.join("two_phase_flow_runs", "DataRelPerm"),
    }
)
run_time_dependent_model(model, {})
