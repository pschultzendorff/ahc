import unittest
from functools import partial

import numpy as np
import porepy as pp
import scipy.sparse as sps
from porepy.models.run_models import run_time_dependent_model
from porepy.numerics.ad.forward_mode import Ad_array

from src.tpf_lab.models.two_phase_flow import TwoPhaseFlow
from src.tpf_lab.numerics.ad import functions as af
from src.tpf_lab.numerics.ad.functions import pow

# This was calculated by hand.
# p_c = 0.1\hat{S}_w = 0.1 \frac{S_w - 0.3}{0.7} = S_w / 7 -0.03
# p'_c = 1 / 7
A_true = np.zeros([4, 8])
jac_value = -1 / 7
A_true[:, 4:] = np.diag([jac_value] * 4)
value = 0.5 / 7 - 0.03  # == S_w / 7 -0.03
b_true = np.full([4, 1], value)


class TwoPhaseFlowTest(TwoPhaseFlow):
    def _assign_equations(self) -> None:
        super()._assign_equations()
        # s = self._ad.saturation
        # normalized_s = (s - self._residual_w_saturation) / (
        #     1 - self._residual_n_saturation - self._residual_w_saturation
        # )
        p_cap = self._cap_pressure()
        self._eq_manager.equations.update({"p_cap_eq": p_cap})
        # invert_func = pp.ad.Function(partial(pow, exponent=-1), "invert")
        # s_invert = invert_func(s)
        # self._eq_manager.equations.update({"norm_s": normalized_s})
        # self._eq_manager.equations.update({"1/s": s_invert})
        # self._eq_manager.equations.update({"s": s})


def test_cap_pressure_function() -> None:
    model = TwoPhaseFlowTest()
    model._cap_pressure_model == "Brooks-Corey"
    model._grid_size = 2
    model._phys_size = 2
    model.prepare_simulation()
    model.before_newton_loop()
    model.before_newton_iteration()
    # Initial saturation and pressure are 0.5 and 0.5.
    A, b = model._eq_manager.assemble_subsystem(eq_names=["p_cap_eq"])
    assert np.allclose(A.todense(), A_true)
    assert np.allclose(b, b_true)


test_cap_pressure_function()
