from typing import Any

import numpy as np
import porepy as pp
import pytest
from tpf_lab.constants_and_typing import NONWETTING, WETTING
from tpf_lab.models.flow_and_transport import TwoPhaseFlow
from tpf_lab.models.phase import Phase, PhaseConstants

# TODO Change the residual saturation of one phase and adjust the tests.


def residual_saturation(phase_name: str) -> float:
    if phase_name == WETTING:
        return 0.1
    elif phase_name == NONWETTING:
        return 0.1


@pytest.fixture
def params() -> dict[str, Any]:
    return {
        "material_constants": {
            "wetting": PhaseConstants(
                {"residual_saturation": residual_saturation(WETTING)}
            ),
            "nonwetting": PhaseConstants(
                {"residual_saturation": residual_saturation(NONWETTING)}
            ),
        }
    }


@pytest.fixture
def phase(request) -> Phase:
    p = Phase(request.param)
    p.set_constants({"residual_saturation": residual_saturation(p.name)})
    return p


@pytest.fixture
def tpf_model(params: dict[str, Any]) -> TwoPhaseFlow:
    model = TwoPhaseFlow(params)
    model.prepare_simulation()
    return model


# TODO Add another test for when saturation is of type
# ``pp.ad.MixedDimensionalVariable``.
@pytest.mark.parametrize("phase", [WETTING, NONWETTING], indirect=True)
@pytest.mark.parametrize(
    "saturation,expected",
    [
        (
            pp.ad.DenseArray(np.array([0.5, 0.6, 0.7, 0.8])),
            np.array([0.5, 0.625, 0.75, 0.875]),
        ),
        (
            pp.ad.DenseArray(np.array([0, 0.1, 0.7, 1.0])),
            np.array([1e-5, 1e-5, 0.75, 1 - 1e-5]),
        ),
        (
            pp.ad.DenseArray(np.array([-1, 0.1, 1.6, 0.3])),
            np.array([1e-5, 1e-5, 1 - 1e-5, 0.25]),
        ),
        (pp.ad.Scalar(0.4), np.array([0.375])),
        (pp.ad.Scalar(0.1), np.array([1e-5])),
        (pp.ad.Scalar(0.9), np.array([1 - 1e-5])),
        (pp.ad.Scalar(0), np.array([1e-5])),
        (pp.ad.Scalar(1.5), np.array([1 - 1e-5])),
        (pp.ad.Scalar(-5), np.array([1e-5])),
    ],
)
def test_normalize_saturation_dense_array(
    tpf_model: TwoPhaseFlow,
    phase: Phase,
    saturation: pp.ad.Operator,
    expected: np.ndarray,
):
    assert tpf_model.wetting.constants.residual_saturation() == 0.1
    assert tpf_model.nonwetting.constants.residual_saturation() == 0.1

    normalized_saturation: pp.ad.Operator = tpf_model.normalize_saturation(
        saturation, phase=phase
    )
    assert np.allclose(
        normalized_saturation.value(tpf_model.equation_system),
        expected,
    )
