"""
Test the two-phase flow model in the fractional flow formulation.

The following parts of the equations are tested separately

    - source term

    - normalized saturation
    - total mobility
    - wetting/nonwetting mobility
    - cap. pressure

    - tpfa and divergence

    - wetting/nonwetting flux
    - flux induced by capillary pressure

    - flow/pressure equation
    - transport/saturation equation

All of the tests are without gravity. The last two tests are repeated with gravity on.

Everything is tested on a 2x2 grid. The boundary conditions are homogeneous Neumann bc
at the top, right and left and homogeneous Dirichlet bc (i.e. :math:`p=0,\lambda_t=0.5`)
at the bottom. As initial conditions, the pressure is set to :math:`p=0` and the
saturation to :math:`S=0.5` on the entire grid.

See the ``model_integration_test.md`` file for more details.

NOTE: As of now, everything is tested without gravity.

NOTE: In these tests we use ``A`` to denote the Jacobian of an equation system and ``b``
to denote its residual. ``PorePy`` provides them in the form :math:`A=J` and
:math:`b=-r`, s.t. the problem can be immediately solved by :math:`A\Delta x=b`.
Thus, we need to always multiply the residual by -1.

"""

from typing import Any

import numpy as np
import porepy as pp
import pytest
from tpf.models.flow_and_transport import TwoPhaseFlow


@pytest.mark.parametrize("params", [{}])
def test_TwoPhaseFlow(params: dict[str, Any]) -> None:
    model = TwoPhaseFlow(params)
    pp.run_time_dependent_model(model, params=params)
