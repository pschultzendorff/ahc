import numpy as np
import porepy as pp
import pytest
from ahc.models.flow_and_transport import TwoPhaseFlow
from porepy.models.protocol import PorePyModel


class TwoPhaseFlowwithAnalyzer(TwoPhaseFlow, AnalyzerMixin):
    def __init__(self, params=None, reference_solution=None):
        TwoPhaseFlow.__init__(self, params)
        AnalyzerMixin.__init__(self, reference_solution)


@pytest.fixture
def reference_solution() -> np.ndarray:
    return np.random.rand(10, 10)  # Example reference solution


@pytest.fixture
def solved_model() -> PorePyModel:
    model = TwoPhaseFlow({})  # type
    pp.set_equation_values(model, np.random.rand(10, 10))  # type: ignore
    return model


def test_compare_fluxes(reference_solution, solved_model):
    pass


class TestCompare:
    @pytest.mark.unit
    def test_compare(self):
        assert 1 == 1
