"""
Test the Buckley-Leverett model.

NOTE: As of now, everything is tested without gravity.

NOTE: In these tests we use ``A`` to denote the Jacobian of an equation system and ``b``
to denote its residual. ``PorePy`` provides them in the form :math:`A=J` and
:math:`b=-r`, s.t. the problem can be immediately solved by :math:`A\Delta x=b`.
Thus, we need to always multiply the residual by -1.

"""
import numpy as np
import porepy as pp
import pytest

from tpf_lab.models.buckley_leverett import BuckleyLeverettSetup


class BL_HomogeneousInitialSaturationBuckleyLeverett(BuckleyLeverettSetup):
    def initial_condition(self) -> None:
        """Residual nonwetting saturation in the left side of the domain. Residual
        wetting saturation in the right side of the domain. A transition zone in the
        middle."""
        sd = self.mdg.subdomains()[0]
        self.equation_system.set_variable_values(
            np.full(sd.num_cells, 0.0),
            [self._ad.pressure_w],
            time_step_index=self.time_manager.time_index,
        )
        # Initialize nonwetting pressure s.t. the gradient is 1 across the domain.
        self.equation_system.set_variable_values(
            np.linspace(self._phys_size, 0.0, self._grid_size),
            [self._ad.pressure_n],
            time_step_index=self.time_manager.time_index,
        )
        initial_saturation = np.full(self._grid_size, 1 - self._residual_saturation_n)
        self.equation_system.set_variable_values(
            initial_saturation,
            [self._ad.saturation],
            time_step_index=self.time_manager.time_index,
        )


@pytest.fixture(scope="module")
def model() -> BL_HomogeneousInitialSaturationBuckleyLeverett:
    model = BL_HomogeneousInitialSaturationBuckleyLeverett(
        {"formulation": "n_pressure_w_saturation"}
    )
    model._grid_size = 4
    model._phys_size = 4
    model._time_step = 1
    model.prepare_simulation()
    model.before_newton_loop()
    model.before_newton_iteration()
    return model


@pytest.fixture
def saturation_w_init() -> np.ndarray:
    return np.array([0.7, 0.6, 0.4, 0.3])


@pytest.fixture
def normalized_saturation_w_init(
    saturation_w_init: np.ndarray, model: BL_HomogeneousInitialSaturationBuckleyLeverett
) -> np.ndarray:
    return (saturation_w_init - model._residual_saturation_w) / (
        1 - model._residual_saturation_w - model._residual_saturation_n
    )


@pytest.fixture
def normalized_saturation_w_init_upwind(
    saturation_w_init: np.ndarray, model: BL_HomogeneousInitialSaturationBuckleyLeverett
) -> np.ndarray:
    return (
        np.full(model._grid_size + 1, saturation_w_init[0])
        - model._residual_saturation_w
    ) / (1 - model._residual_saturation_w - model._residual_saturation_n)


@pytest.fixture
def pressure_w_init() -> np.ndarray:
    return np.array([0.0] * 4)


@pytest.fixture
def s_normalized_jac(
    saturation_w_init, model: BL_HomogeneousInitialSaturationBuckleyLeverett
) -> np.ndarray:
    """Normalized saturation Jacobian w.r.t. saturation."""
    A = np.eye(saturation_w_init.shape[0]) / (
        1 - model._residual_saturation_w - model._residual_saturation_n
    )
    return A


def test_normalized_s(
    model: BL_HomogeneousInitialSaturationBuckleyLeverett, s_normalized_jac: np.ndarray
) -> None:
    """The model uses the normalized saturation for the rel. perm. and cap. pressure
    functions. We check that its Jacobian is calculated correctly."""
    s_normalized_system = model._s_normalized().evaluate(model.equation_system)
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
    model: BL_HomogeneousInitialSaturationBuckleyLeverett, mobility_t: np.ndarray
) -> None:
    """Test that the model's total mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    # Set ``atol`` high s.t. the epsilon in the total mobility does not influence the
    # comparison.
    assert np.allclose(
        model._mobility_t(subdomains).evaluate(model.equation_system).val,
        mobility_t,
    )


@pytest.fixture
def mobility_w(rel_perm_w: np.ndarray) -> np.ndarray:
    """Wetting mobility for all domain interfaces at t=0. Calculated by hand.

    Note that the viscosities equal 1, hence rel_perm==mobility.

    """
    return rel_perm_w


def test_mobility_w(
    model: BL_HomogeneousInitialSaturationBuckleyLeverett, mobility_w: np.ndarray
) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    assert np.allclose(
        model._mobility_w(subdomains).evaluate(model.equation_system).val,
        mobility_w,
    )


@pytest.fixture
def mobility_n(
    model: BL_HomogeneousInitialSaturationBuckleyLeverett, rel_perm_n: np.ndarray
) -> np.ndarray:
    """Nonwetting mobility for all domain interfaces at t=0. Calculated by hand.

    Note that the viscosities equal 1, hence rel_perm==mobility.

    """
    return rel_perm_n


def test_mobility_n(
    model: BL_HomogeneousInitialSaturationBuckleyLeverett, mobility_n: np.ndarray
) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    assert np.allclose(
        model._mobility_n(subdomains).evaluate(model.equation_system).val,
        mobility_n,
    )


def test_porosity_n(model: BL_HomogeneousInitialSaturationBuckleyLeverett):
    assert np.all(model._porosity(model.mdg.subdomains()[0]) == 1.0)


@pytest.fixture
def tpfa_array() -> np.ndarray:
    """Array of the divergence of the TPFA array."""
    return np.asarray([[3, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])


def test_tpfa(
    tpfa_array: np.ndarray,
    model: BL_HomogeneousInitialSaturationBuckleyLeverett,
) -> None:
    """Test that TPFA works correctly."""
    subdomains = model.mdg.subdomains()
    tpfa = pp.ad.TpfaAd(model.w_flux_key, subdomains)
    div = pp.ad.Divergence(subdomains)
    tpfa_system = (div * tpfa.flux).evaluate(model.equation_system)
    assert np.allclose(tpfa_system.todense(), tpfa_array)


@pytest.fixture
def mpfa_array() -> np.ndarray:
    """Array of the divergence of the TPFA array."""
    return np.asarray([[3, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])


def test_mpfa(
    mpfa_array: np.ndarray,
    model: BL_HomogeneousInitialSaturationBuckleyLeverett,
) -> None:
    """Test that MPFA works correctly."""
    subdomains = model.mdg.subdomains()
    tpfa = pp.ad.MpfaAd(model.w_flux_key, subdomains)
    div = pp.ad.Divergence(subdomains)
    tpfa_system = (div * tpfa.flux).evaluate(model.equation_system)
    assert np.allclose(tpfa_system.todense(), mpfa_array)


@pytest.fixture
def pressure_gradient_val(model: BL_HomogeneousInitialSaturationBuckleyLeverett):
    """Pressure was initiated s.t. the pressure gradient across the domain is 1."""
    return np.eye(model._grid_size + 1)


@pytest.fixture
def flow_equation_val(
    model: BL_HomogeneousInitialSaturationBuckleyLeverett,
) -> np.ndarray:
    """Residual of the flow equation at t=0. Calculated by hand.

    The residual equals the influx (Neumann bc).

    """
    b = np.zeros(model._grid_size)
    b[-1] = model._influx
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
    model: BL_HomogeneousInitialSaturationBuckleyLeverett,
    flow_equation_jac_wrt_pressure: np.ndarray,
    flow_equation_jac_wrt_saturation: np.ndarray,
    flow_equation_val: np.ndarray,
) -> None:
    """Test the flow equation; both w.r.t. to :math:`p` and w.r.t. :math:`S`."""
    A, _ = model.equation_system.assemble_subsystem(
        equations=["Flow equation"], variables=[model.primary_pressure_var]
    )
    assert np.allclose(A.todense(), flow_equation_jac_wrt_pressure)
    A, b = model.equation_system.assemble_subsystem(
        equations=["Flow equation"], variables=[model.saturation_var]
    )
    assert np.allclose(A.todense(), flow_equation_jac_wrt_saturation)
    assert np.allclose(b, -flow_equation_val)


@pytest.fixture
def transport_equation_val(
    model: BL_HomogeneousInitialSaturationBuckleyLeverett,
) -> np.ndarray:
    """Residual of the flow equation at t=0. Calculated by hand.

    The residual equals the influx (Neumann bc).

    """
    b = np.zeros(model._grid_size)
    b[-1] = model._influx
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
    model: BL_HomogeneousInitialSaturationBuckleyLeverett,
) -> np.ndarray:
    """Jacobian of the transport equation w.r.t. to :math:`S_w` at t=0. Calculated by
    hand."""
    # Pressure gradient is zero, hence no influence of .
    A_1 = np.zeros(model._grid_size)
    # Timestep
    A_2 = (
        model._porosity(model.mdg.subdomains()[0])[0]
        / model._time_step
        * np.eye(model._grid_size)
    )
    # The domain's porosity is homogeneous and equals 1.
    return A_1 + A_2


def test_transport_equation(
    model: BL_HomogeneousInitialSaturationBuckleyLeverett,
    flow_equation_val: np.ndarray,
    transport_equation_jac_wrt_pressure: np.ndarray,
    transport_equation_jac_wrt_saturation: np.ndarray,
) -> None:
    """Test the transport equation; both w.r.t. to :math:`p` and w.r.t. :math:`S`."""
    A, b = model.equation_system.assemble_subsystem(
        equations=["Transport equation"], variables=[model.primary_pressure_var]
    )
    assert np.allclose(A.todense(), transport_equation_jac_wrt_pressure)
    A, b = model.equation_system.assemble_subsystem(
        equations=["Transport equation"], variables=[model.saturation_var]
    )
    # Add some tolerance, as the model divides by :math:`\lambda_t + 1e-7`, but
    # multiplies by :math:`\lambda_t`. We skip this step in the exact calculation.
    assert np.allclose(A.todense(), transport_equation_jac_wrt_saturation, atol=1e-5)
    assert np.allclose(b, -flow_equation_val)


class BL_LinearInitialSaturationBuckleyLeverett(BuckleyLeverettSetup):
    def initial_condition(self) -> None:
        """Residual nonwetting saturation in the left side of the domain. Residual
        wetting saturation in the right side of the domain. A transition zone in the
        middle."""
        sd = self.mdg.subdomains()[0]
        self.equation_system.set_variable_values(
            np.full(sd.num_cells, 0.0),
            [self._ad.pressure_w],
            time_step_index=self.time_manager.time_index,
        )
        self.equation_system.set_variable_values(
            np.full(sd.num_cells, 0.0),
            [self._ad.pressure_n],
            time_step_index=self.time_manager.time_index,
        )
        initial_saturation = np.linspace(
            self._residual_saturation_w,
            1 - self._residual_saturation_n,
            self._grid_size,
        )
        self.equation_system.set_variable_values(
            initial_saturation,
            [self._ad.saturation],
            time_step_index=self.time_manager.time_index,
        )


@pytest.fixture(scope="module")
def model_for_frac_flow_testing() -> BL_LinearInitialSaturationBuckleyLeverett:
    model = BL_LinearInitialSaturationBuckleyLeverett(
        {"formulation": "n_pressure_w_saturation"}
    )
    model._grid_size = 20
    model._phys_size = 1
    model.prepare_simulation()
    model.before_newton_loop()
    model.before_newton_iteration()
    return model


def test_fractional_flow(
    model_for_frac_flow_testing: BL_LinearInitialSaturationBuckleyLeverett,
):
    subdomains = model_for_frac_flow_testing.mdg.subdomains()
    mobility_w = model_for_frac_flow_testing._mobility_w(subdomains=subdomains)
    mobility_t = model_for_frac_flow_testing._mobility_t(subdomains=subdomains)
    fractional_flow = mobility_w / mobility_t
    fractional_flow_system = fractional_flow.evaluate(
        model_for_frac_flow_testing.equation_system
    )
    assert np.all(fractional_flow_system.val == 1)
