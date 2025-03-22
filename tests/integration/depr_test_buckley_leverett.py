r"""Deprecated until the Buckley-Leverett model is implemented in the new framework.

Test the Buckley-Leverett model.

NOTE: As of now, everything is tested without gravity.

NOTE: In these tests we use ``A`` to denote the Jacobian of an equation system and ``b``
to denote its residual. ``PorePy`` provides them in the form :math:`A = J` and
:math:`b = -r`, s.t. the problem can be immediately solved by :math:`A \Delta x = b`.
Thus, we need to always multiply the residual by -1.

"""

import numpy as np
import porepy as pp
import pytest
from tpf.models.buckley_leverett import BuckleyLeverettSetup


@pytest.fixture(scope="module")
def model_4_grid_cells() -> BuckleyLeverettSetup:
    phys_size: float = 4.0
    grid_cells: int = 4
    model = BuckleyLeverettSetup(
        {
            "formulation": "n_pressure_w_saturation",
            "meshing_arguments": {"cell_size": phys_size / grid_cells},
        }
    )
    model._time_step = 1
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    return model


@pytest.fixture(scope="module")
def model_200_grid_cells() -> BuckleyLeverettSetup:
    phys_size: float = 20.0
    grid_cells: int = 200
    model = BuckleyLeverettSetup(
        {
            "formulation": "n_pressure_w_saturation",
            "meshing_arguments": {"cell_size": phys_size / grid_cells},
        }
    )
    model.prepare_simulation()
    return model


@pytest.fixture
def saturation_w_init() -> np.ndarray:
    return np.array([0.7, 0.6, 0.4, 0.3])


@pytest.fixture
def normalized_saturation_w_init(
    saturation_w_init: np.ndarray, model_4_grid_cells: BuckleyLeverettSetup
) -> np.ndarray:
    return (saturation_w_init - model_4_grid_cells._residual_saturation_w) / (
        1
        - model_4_grid_cells._residual_saturation_w
        - model_4_grid_cells._residual_saturation_n
    )


@pytest.fixture
def normalized_saturation_w_init_upwind(
    saturation_w_init: np.ndarray, model_4_grid_cells: BuckleyLeverettSetup
) -> np.ndarray:
    return (
        np.full(model_4_grid_cells._grid_size + 1, saturation_w_init[0])
        - model_4_grid_cells._residual_saturation_w
    ) / (
        1
        - model_4_grid_cells._residual_saturation_w
        - model_4_grid_cells._residual_saturation_n
    )


@pytest.fixture
def pressure_w_init() -> np.ndarray:
    return np.array([0.0] * 4)


@pytest.fixture
def s_normalized_jac(
    saturation_w_init, model_4_grid_cells: BuckleyLeverettSetup
) -> np.ndarray:
    """Normalized saturation Jacobian w.r.t. saturation."""
    A = np.eye(saturation_w_init.shape[0]) / (
        1
        - model_4_grid_cells._residual_saturation_w
        - model_4_grid_cells._residual_saturation_n
    )
    return A


def test_normalized_s(
    model_4_grid_cells: BuckleyLeverettSetup, s_normalized_jac: np.ndarray
) -> None:
    """The model uses the normalized saturation for the rel. perm. and cap. pressure
    functions. We check that its Jacobian is calculated correctly."""
    s_normalized_system = model_4_grid_cells._s_normalized().evaluate(
        model_4_grid_cells.equation_system
    )
    assert np.allclose(s_normalized_system.jac.todense()[:, -4:], s_normalized_jac)


@pytest.fixture
def rel_perm_w(normalized_saturation_w_init_upwind: np.ndarray) -> np.ndarray:
    """Power rel. perm model and residual saturations at the boundaries"""
    rel_perm_w = np.minimum(normalized_saturation_w_init_upwind**3, 0.99)
    rel_perm_w[0] = 0.01
    rel_perm_w[-1] = 0.99
    return rel_perm_w


@pytest.fixture
def rel_perm_n(normalized_saturation_w_init_upwind: np.ndarray) -> np.ndarray:
    """Power rel. perm model and residual saturations at the boundaries"""
    rel_perm_n = np.maximum((1 - normalized_saturation_w_init_upwind) ** 3, 0.01)
    rel_perm_n[0] = 0.99
    rel_perm_n[-1] = 0.01
    return rel_perm_n


@pytest.fixture
def mobility_t(mobility_w: np.ndarray, mobility_n: np.ndarray) -> np.ndarray:
    """Total mobility for all domain interfaces at t=0. Calculated by hand."""
    mobility_t = mobility_w + mobility_n + 1e-7
    return mobility_t


def test_mobility_t(
    model_4_grid_cells: BuckleyLeverettSetup, mobility_t: np.ndarray
) -> None:
    """Test that the model's total mobility coincides with the one calculated by
    hand."""
    subdomains = model_4_grid_cells.mdg.subdomains()
    # Set ``atol`` high s.t. the epsilon in the total mobility does not influence the
    # comparison.
    assert np.allclose(
        model_4_grid_cells._mobility_t(subdomains)
        .evaluate(model_4_grid_cells.equation_system)
        .val,
        mobility_t,
    )


@pytest.fixture
def mobility_w(rel_perm_w: np.ndarray) -> np.ndarray:
    """Wetting mobility for all domain interfaces at t=0. Calculated by hand.

    Note that the viscosities equal 1, hence rel_perm==mobility.

    """
    return rel_perm_w


def test_mobility_w(
    model_4_grid_cells: BuckleyLeverettSetup, mobility_w: np.ndarray
) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model_4_grid_cells.mdg.subdomains()
    assert np.allclose(
        model_4_grid_cells._mobility_w(subdomains)
        .evaluate(model_4_grid_cells.equation_system)
        .val,
        mobility_w,
    )


@pytest.fixture
def mobility_n(
    model_4_grid_cells: BuckleyLeverettSetup, rel_perm_n: np.ndarray
) -> np.ndarray:
    """Nonwetting mobility for all domain interfaces at t=0. Calculated by hand.

    Note that the viscosities equal 1, hence rel_perm==mobility.

    """
    return rel_perm_n


def test_mobility_n(
    model_4_grid_cells: BuckleyLeverettSetup, mobility_n: np.ndarray
) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model_4_grid_cells.mdg.subdomains()
    assert np.allclose(
        model_4_grid_cells._mobility_n(subdomains)
        .evaluate(model_4_grid_cells.equation_system)
        .val,
        mobility_n,
    )


def test_porosity_n(model_4_grid_cells: BuckleyLeverettSetup):
    assert np.all(
        model_4_grid_cells._porosity(model_4_grid_cells.mdg.subdomains()[0]) == 1.0
    )


@pytest.fixture
def tpfa_array() -> np.ndarray:
    """Array of the divergence of the TPFA array."""
    return np.asarray([[3, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])


def test_tpfa(
    tpfa_array: np.ndarray,
    model_4_grid_cells: BuckleyLeverettSetup,
) -> None:
    """Test that TPFA works correctly."""
    subdomains = model_4_grid_cells.mdg.subdomains()
    tpfa = pp.ad.TpfaAd(model_4_grid_cells.w_flux_key, subdomains)
    div = pp.ad.Divergence(subdomains)
    tpfa_system = (div * tpfa.flux).evaluate(model_4_grid_cells.equation_system)
    assert np.allclose(tpfa_system.todense(), tpfa_array)


@pytest.fixture
def tpfa_array() -> np.ndarray:
    """Array of the divergence of the TPFA array."""
    return np.asarray([[3, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])


def test_tpfa(
    tpfa_array: np.ndarray,
    model_4_grid_cells: BuckleyLeverettSetup,
) -> None:
    """Test that MPFA works correctly."""
    subdomains = model_4_grid_cells.mdg.subdomains()
    tpfa = pp.ad.TpfaAd(model_4_grid_cells.w_flux_key, subdomains)
    div = pp.ad.Divergence(subdomains)
    tpfa_system = (div * tpfa.flux).evaluate(model_4_grid_cells.equation_system)
    assert np.allclose(tpfa_system.todense(), tpfa_array)


@pytest.fixture
def pressure_gradient_val(model_4_grid_cells: BuckleyLeverettSetup):
    """Pressure was initiated s.t. the pressure gradient across the domain is 1."""
    return np.eye(model_4_grid_cells._grid_size + 1)


@pytest.fixture
def flow_equation_val(
    model_4_grid_cells: BuckleyLeverettSetup,
) -> np.ndarray:
    """Residual of the flow equation at t=0. Calculated by hand.

    The residual equals the influx (Neumann bc).

    """
    b = np.zeros(model_4_grid_cells._grid_size)
    b[-1] = model_4_grid_cells._influx
    return b


@pytest.fixture
def flow_equation_jac_wrt_pressure(
    tpfa_array: np.ndarray, mobility_t: np.ndarray
) -> np.ndarray:
    """Jacobian of the flow equation w.r.t. to :math:`p_n` at t=0. Calculated by
    hand."""
    # Total flow induced inside the domain. The total mobility is takem from one of the
    # inner interfaces (they are equal for all).
    A_1 = mobility_t[1] * tpfa_array
    # Total flow induced by the Dirichlet bc for the left cell
    A_2 = (mobility_t[0] - mobility_t[1]) * np.asarray(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    return A_1 + A_2


@pytest.fixture
def flow_equation_jac_wrt_saturation(
    tpfa_array: np.ndarray,
) -> np.ndarray:
    """Jacobian of the flow equation w.r.t. to :math:`S_w` at t=0. Calculated by
    hand."""
    return np.zeros_like(tpfa_array)


def test_flow_equation(
    model_4_grid_cells: BuckleyLeverettSetup,
    flow_equation_jac_wrt_pressure: np.ndarray,
    flow_equation_jac_wrt_saturation: np.ndarray,
    flow_equation_val: np.ndarray,
) -> None:
    """Test the flow equation; both w.r.t. to :math:`p` and w.r.t. :math:`S`."""
    A, _ = model_4_grid_cells.equation_system.assemble_subsystem(
        equations=["Flow equation"], variables=[model_4_grid_cells.primary_pressure_var]
    )
    assert np.allclose(A.todense(), flow_equation_jac_wrt_pressure)
    A, b = model_4_grid_cells.equation_system.assemble_subsystem(
        equations=["Flow equation"], variables=[model_4_grid_cells.saturation_var]
    )
    assert np.allclose(A.todense(), flow_equation_jac_wrt_saturation)
    assert np.allclose(b, -flow_equation_val)


@pytest.fixture
def transport_equation_val(
    model_4_grid_cells: BuckleyLeverettSetup,
) -> np.ndarray:
    """Residual of the flow equation at t=0. Calculated by hand.

    The residual equals the influx (Neumann bc).

    """
    b = np.zeros(model_4_grid_cells._grid_size)
    b[-1] = model_4_grid_cells._influx
    return b


@pytest.fixture
def transport_equation_jac_wrt_pressure(
    tpfa_array: np.ndarray, mobility_w: np.ndarray
) -> np.ndarray:
    """Jacobian of the transport equation w.r.t. to :math:`p_n` at t=0. Calculated by
    hand."""
    # Total flow induced inside the domain. The fractional flow is taken from one of the
    # inner interfaces (they are equal for all).
    A_1 = mobility_w[1] * tpfa_array
    # Total flow induced by the Dirichlet bc for the southern two cells. The total
    # mobility is takem from one of the two southern boundaries (they are equal for
    # both). We substract the total mobility of an inner interface, to not count the
    # flow twice (it is already included in ``A_1``).
    A_2 = (mobility_w[0] - mobility_w[1]) * np.asarray(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    return A_1 + A_2


@pytest.fixture
def transport_equation_jac_wrt_saturation(
    tpfa_array: np.ndarray,
    mobility_w: np.ndarray,
    model_4_grid_cells: BuckleyLeverettSetup,
) -> np.ndarray:
    """Jacobian of the transport equation w.r.t. to :math:`S_w` at t=0. Calculated by
    hand."""
    # Pressure gradient is zero, hence no influence of .
    A_1 = np.zeros(model_4_grid_cells._grid_size)
    # Timestep
    A_2 = (
        model_4_grid_cells._porosity(model_4_grid_cells.mdg.subdomains()[0])[0]
        / model_4_grid_cells._time_step
        * np.eye(model_4_grid_cells._grid_size)
    )
    # The domain's porosity is homogeneous and equals 1.
    return A_1 + A_2


def test_transport_equation(
    model_4_grid_cells: BuckleyLeverettSetup,
    flow_equation_val: np.ndarray,
    transport_equation_jac_wrt_pressure: np.ndarray,
    transport_equation_jac_wrt_saturation: np.ndarray,
) -> None:
    """Test the transport equation; both w.r.t. to :math:`p` and w.r.t. :math:`S`."""
    A, b = model_4_grid_cells.equation_system.assemble_subsystem(
        equations=["Transport equation"],
        variables=[model_4_grid_cells.primary_pressure_var],
    )
    assert np.allclose(A.todense(), transport_equation_jac_wrt_pressure)
    A, b = model_4_grid_cells.equation_system.assemble_subsystem(
        equations=["Transport equation"], variables=[model_4_grid_cells.saturation_var]
    )
    # Add some tolerance, as the model divides by :math:`\lambda_t + 1e-7`, but
    # multiplies by :math:`\lambda_t`. We skip this step in the exact calculation.
    assert np.allclose(A.todense(), transport_equation_jac_wrt_saturation, atol=1e-5)
    assert np.allclose(b, -flow_equation_val)


@pytest.fixture(scope="module")
def model_nonmatching_phys_and_grid() -> BuckleyLeverettSetup:
    model = BuckleyLeverettSetup({"formulation": "n_pressure_w_saturation"})
    model._grid_size = 20
    model._phys_size = 1
    model.prepare_simulation()
    model.before_newton_loop()
    model.before_newton_iteration()
    return model


def test_fractional_flow(
    model_nonmatching_phys_and_grid: BuckleyLeverettSetup,
) -> None:
    subdomains = model_nonmatching_phys_and_grid.mdg.subdomains()
    mobility_w = model_nonmatching_phys_and_grid._mobility_w(subdomains=subdomains)
    mobility_t = model_nonmatching_phys_and_grid._mobility_t(subdomains=subdomains)
    fractional_flow = mobility_w / mobility_t
    fractional_flow_system = fractional_flow.evaluate(
        model_nonmatching_phys_and_grid.equation_system
    )
    assert np.all(fractional_flow_system.val == 1)


def test_initial_flux(model_200_grid_cells: BuckleyLeverettSetup) -> None:
    """Test that the initial values are set s.t. the total flux equals 1 on the entire
    domain."""
    flux_t: np.ndarray = (
        model_200_grid_cells._flux_t(model_200_grid_cells.subdomains()[0])
        .evaluate(model_200_grid_cells.equation_system)
        .val
    )
    assert np.allclose(flux_t, np.full_like(flux_t, 1.0))
