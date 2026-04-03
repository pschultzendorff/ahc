import numpy as np
import porepy as pp
import pytest
from ahc.models.homotopy_continuation import TwoPhaseFlowHC

rng = np.random.default_rng()


@pytest.fixture
def model(request) -> TwoPhaseFlowHC:
    rp_model: str = request.param
    params = {
        "rel_perm_constants": {
            "model_1": {"model": "linear"},
            "model_2": {"model": rp_model},
        },
    }
    return TwoPhaseFlowHC(params=params)  # type: ignore


@pytest.fixture
def model_init_cond(model: TwoPhaseFlowHC, request):
    seed: int = request.param
    rng = np.random.default_rng(seed)
    init_sat: np.ndarray = rng.random(model.g.num_cells)
    init_press: np.ndarray = rng.random(model.g.num_cells)
    model.equation_system.set_variable_values(
        # Nonwetting saturation is 1 - wetting saturation.
        np.concat(
            [
                init_sat,
                1 - init_sat,
            ]
        ),
        [model.wetting.s, model.nonwetting.s],
        time_step_index=0,
        hc_index=0,
        iterate_index=0,
    )
    model.equation_system.set_variable_values(
        # Same values for wetting and nonwetting pressure.
        np.concat(
            [
                init_press,
                init_press,
            ]
        ),
        [model.wetting.p, model.nonwetting.p],
        time_step_index=0,
        hc_index=0,
        iterate_index=0,
    )
    return model


# FIXME Finish writing this test.
@pytest.mark.parametrize("model", ["Corey", "Brooks-Corey"], indirect=True)
@pytest.mark.parametrize("model_init_cond", [0, 1, 2, 3, 4], indirect=True)
def test_two_phase_flow_ahc_hc_estimator(model, model_init_cond: TwoPhaseFlowHC):
    # Set hc_lambda_ad to 0 and calculate the continuation estimators.
    model_init_cond.nonlinear_solver_statistics.hc_lambda_ad.set_value(0.0)
    model_init_cond.local_hc_est("total")

    # Check that the estimator values saved to g_data are almost zero
    estimator_values = pp.get_solution_values(
        "total_C_estimator", model_init_cond.g_data, iterate_index=0
    )
    assert np.allclose(estimator_values, 0.0, atol=1e-6)
