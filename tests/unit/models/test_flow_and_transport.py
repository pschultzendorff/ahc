from typing import Any

import numpy as np
import porepy as pp
import pytest
from ahc.models.flow_and_transport import TwoPhaseFlow
from ahc.models.phase import FluidPhase
from ahc.models.protocol import TPFProtocol
from ahc.utils.constants_and_typing import NONWETTING, WETTING

# TODO Change the residual saturation of one phase and adjust the tests.


class ModifiedGeometry(TPFProtocol):
    def set_domain(self) -> None:
        """2x2 domain."""
        size = 2 / self.units.m
        self._domain = pp.applications.md_grids.domains.nd_cube_domain(2, size)

    def meshing_arguments(self) -> dict[str, float]:
        """2x2 grid cells."""
        default_meshing_args: dict[str, float] = {"cell_size": 1.0}
        return self.params.get("meshing_arguments", default_meshing_args)


class TwoPhaseFlowEqationsSource(TPFProtocol):
    def _source_w(self, g: pp.Grid) -> np.ndarray:
        """Volumetric wetting source.

        In the default model there is no source term.

        SI Units: m^d/s
        """
        source = np.zeros(g.num_cells)
        source[0] = 1
        return source

    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Zero vector source (gravity). Corresponds to the wetting buoyancy flow."""
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Zero vector volume source (gravity). Corresponds to the nonwetting buoyancy
        flow."""
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        return vals.ravel()


class TwoPhaseFlowEqationsSourceandGravity(TPFProtocol):
    def _vector_source_w(self, g: pp.Grid) -> np.ndarray:
        """Vector volume source (gravity). Corresponds to the wetting buoyancy flow."""
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * self.wetting.density
        return vals.ravel()

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Vector volume source (gravity). Corresponds to the nonwetting buoyancy
        flow."""
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        vals[:, -1] = pp.GRAVITY_ACCELERATION * self.nonwetting.density
        return vals.ravel()


class TwoPhaseFlowModifiedSetup(TwoPhaseFlowEqationsSource, TwoPhaseFlow): ...


class TwoPhaseFlowModifiedSetupGravity(
    TwoPhaseFlowEqationsSourceandGravity, TwoPhaseFlow()
): ...


@pytest.fixture(scope="module")
def model() -> TwoPhaseFlowModifiedSetup:
    model = TwoPhaseFlowModifiedSetup({"formulation": "n_pressure_w_saturation"})
    model.cap_press.cap_pressure_model = "linear"
    model._rel_perm_model = "Brooks-Corey"
    model._limit_rel_perm = "False"
    model.grid_type = "simplex"
    model.phys_size = 2
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    return model


@pytest.fixture(scope="module")
def model_with_gravity() -> TwoPhaseFlowModifiedSetupGravity:
    model = TwoPhaseFlowModifiedSetupGravity(
        {"formulation": "n_pressure_w_saturation", "density_w": 1.0, "density_n": 2.0}
    )
    model._cap_pressure_model = "linear"
    model._rel_perm_model = "Brooks-Corey"
    model._limit_rel_perm = "False"
    model._grid_size = 2
    model._phys_size = 2
    model.prepare_simulation()
    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    return model


def residual_saturation(phase_name: str) -> float:
    if phase_name == WETTING:
        return 0.1
    elif phase_name == NONWETTING:
        return 0.1


@pytest.fixture
def params() -> dict[str, Any]:
    return {
        "material_constants": {
            "wetting": FluidPhase(
                {"name": WETTING, "residual_saturation": residual_saturation(WETTING)}
            ),
            "nonwetting": FluidPhase(
                {
                    "name": NONWETTING,
                    "residual_saturation": residual_saturation(NONWETTING),
                }
            ),
        }
    }


@pytest.fixture
def phase(request) -> FluidPhase:
    return FluidPhase(
        {
            "name": request.param,
            "residual_saturation": residual_saturation(request.param),
        }
    )


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
    phase: FluidPhase,
    saturation: pp.ad.Operator,
    expected: np.ndarray,
):
    assert tpf_model.wetting.residual_saturation == 0.1
    assert tpf_model.nonwetting.residual_saturation == 0.1

    normalized_saturation: pp.ad.Operator = tpf_model.normalize_saturation(
        saturation, phase=phase
    )
    assert np.allclose(
        normalized_saturation.value(tpf_model.equation_system),
        expected,
    )


@pytest.fixture
def source_w() -> np.ndarray:
    return np.array([1, 0, 0, 0])


def test_source(model: TwoPhaseFlowModifiedSetup, source_w):
    """Test that the source term is implemented correctly.

    As the nonwetting source is zero, the total source and wetting source are equal.

    """
    source_ad_t = pp.ad.DenseArray(model._source_t(model.mdg.subdomains()[0]))
    source_system = source_ad_t.evaluate(model.equation_system)
    assert np.all(source_system == source_w)


@pytest.fixture
def saturation_w_init() -> np.ndarray:
    return np.array([0.5] * 4)


@pytest.fixture
def normalized_saturation_w_init(
    saturation_w_init: np.ndarray, model: TwoPhaseFlowModifiedSetup
) -> np.ndarray:
    return (saturation_w_init - model._residual_saturation_w) / (
        1 - model._residual_saturation_w - model._residual_saturation_n
    )


@pytest.fixture
def pressure_w_init() -> np.ndarray:
    return np.array([0.0] * 4)


@pytest.fixture
def s_normalized_jac(saturation_w_init, model: TwoPhaseFlowModifiedSetup) -> np.ndarray:
    """Normalized saturation Jacobian w.r.t. saturation."""
    A = np.eye(saturation_w_init.shape[0]) / (
        1 - model._residual_saturation_w - model._residual_saturation_n
    )
    return A


def test_normalized_s(
    model: TwoPhaseFlowModifiedSetup, s_normalized_jac: np.ndarray
) -> None:
    """The model uses the normalized saturation for the rel. perm. and cap. pressure
    functions. We check that its Jacobian is calculated correctly."""
    s_normalized_system = model._s_normalized().evaluate(model.equation_system)
    assert np.allclose(s_normalized_system.jac.todense()[:, -4:], s_normalized_jac)


@pytest.fixture
def cap_pressure_val(
    normalized_saturation_w_init: np.ndarray, model: TwoPhaseFlowModifiedSetup
) -> np.ndarray:
    """Linear cap. pressure model residual."""
    return normalized_saturation_w_init * model._cap_pressure_linear_param


@pytest.fixture
def cap_pressure_jac(
    saturation_w_init: np.ndarray, model: TwoPhaseFlowModifiedSetup
) -> np.ndarray:
    """Linear cap. pressure model Jacobian."""
    A_wrt_saturation = (
        np.eye(saturation_w_init.shape[0])
        / (1 - model._residual_saturation_w - model._residual_saturation_n)
        * model._cap_pressure_linear_param
    )
    # Add zero Jacobians for both pressure variables
    A = np.concatenate(
        [
            np.zeros_like(A_wrt_saturation),
            np.zeros_like(A_wrt_saturation),
            A_wrt_saturation,
        ],
        axis=-1,
    )
    return A


def test_cap_pressure_function(
    model: TwoPhaseFlowModifiedSetup,
    cap_pressure_val: np.ndarray,
    cap_pressure_jac: np.ndarray,
) -> None:
    """The model uses a linear cap. presure model. We compare the values with the ones
    calculated by hand.

    Parameters:
        model: _description_

    """
    p_cap_system = model._cap_pressure().evaluate(model.equation_system)
    A, b = p_cap_system.jac, p_cap_system.val
    assert np.allclose(b, cap_pressure_val)
    assert np.allclose(A.todense()[-4:], cap_pressure_jac)


@pytest.fixture
def rel_perm_w(normalized_saturation_w_init: np.ndarray) -> np.ndarray:
    """Brooks-Corey rel. perm model."""
    n_1 = 2.0
    n_2 = 3.0
    n_3 = 1.0
    return normalized_saturation_w_init ** (n_1 + n_2 * n_3)


@pytest.fixture
def rel_perm_n(normalized_saturation_w_init: np.ndarray) -> np.ndarray:
    """Brooks-Corey rel. perm model."""
    n_1 = 2.0
    n_2 = 3.0
    n_3 = 1.0
    return ((1 - normalized_saturation_w_init) ** n_1) * (
        (1 - normalized_saturation_w_init**n_2) ** n_3
    )


@pytest.fixture
def mobility_t(
    model: TwoPhaseFlowModifiedSetup,
    rel_perm_w: np.ndarray,
    rel_perm_n: np.ndarray,
) -> np.ndarray:
    """Total mobility for all domain interfaces at t=0. Calculated by hand.

    We differentiate between inside interfaces, boundary interfaces with homogeneous
    Dirichlet bc and boundary interfaces with homogeneous Neumann bc.

    """
    domain = model.mdg.subdomains()[0]
    # Homogeneous Neumann bc.
    mobility_t = np.full(domain.num_faces, 0.0)
    # Dirichlet bc. Applies at the bottom, i.e., interfaces 6 and 7.
    # NOTE: When wetting outflow occurs, the mobility equals the one in the inside cell.
    mobility_t[6:8] = 0.5
    # Inside interfaces. Calculated based on the initial pressure :math:`S_0=0.5`.
    mobility_t[
        [
            1,
            4,
            8,
            9,
        ]
    ] = rel_perm_w / model._viscosity_w + rel_perm_n / model._viscosity_n
    # The :math:`\epsilon=1e-7` is added in the model to prevent division by zero, hence
    # we add it here as well.
    return mobility_t + 1e-7


def test_mobility_t(model: TwoPhaseFlowModifiedSetup, mobility_t: np.ndarray) -> None:
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
def mobility_w(model: TwoPhaseFlowModifiedSetup, rel_perm_w: np.ndarray) -> np.ndarray:
    """Wetting mobility for all domain interfaces at t=0. Calculated by hand.

    We differentiate between inside interfaces, boundary interfaces with homogeneous
    Dirichlet bc and boundary interfaces with homogeneous Neumann bc.

    Note that the value at the Dirichlet boundaries is scaled with the flux.
    TODO: Find out why this makes sense!

    """
    domain = model.mdg.subdomains()[0]
    # Homogeneous Neumann bc.
    mobility_w = np.full(domain.num_faces, 0.0)
    # Dirichlet bc. Applies at the bottom, i.e., interfaces 6 and 7.
    # NOTE: When wetting outflow occurs, the mobility equals the one in the inside cell.
    mobility_w[6:8] = 0.25
    # Inside interfaces. Calculated based on the initial pressure :math:`S_0=0.5`.
    mobility_w[
        [
            1,
            4,
            8,
            9,
        ]
    ] = rel_perm_w / model._viscosity_w

    return mobility_w


def test_mobility_w(model: TwoPhaseFlowModifiedSetup, mobility_w: np.ndarray) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    assert np.allclose(
        model._mobility_w(subdomains).evaluate(model.equation_system).val,
        mobility_w,
    )


@pytest.fixture
def mobility_n(model: TwoPhaseFlowModifiedSetup, rel_perm_n: np.ndarray) -> np.ndarray:
    """Nonwetting mobility for all domain interfaces at t=0. Calculated by hand.

    We differentiate between inside interfaces, boundary interfaces with homogeneous
    Dirichlet bc and boundary interfaces with homogeneous Neumann bc.

    Note that the value at the Dirichlet boundaries is scaled with the flux.
    TODO: Find out why this makes sense!

    """
    domain = model.mdg.subdomains()[0]
    # Homogeneous Neumann bc
    mobility_n = np.zeros(domain.num_faces)
    # Dirichlet bc. Applies at the bottom, i.e., interfaces 6 and 7. This equals the
    # fixed boundary value we set in ``_bc_values_mobility_t`` in the model.
    mobility_n[6:8] = 0.25
    # Inside interfaces. Calculated based on the initial pressure :math:`S_0=0.5`.
    mobility_n[
        [
            1,
            4,
            8,
            9,
        ]
    ] = rel_perm_n / model._viscosity_n
    return mobility_n


def test_mobility_n(model: TwoPhaseFlowModifiedSetup, mobility_n: np.ndarray) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    assert np.allclose(
        model._mobility_n(subdomains).evaluate(model.equation_system).val,
        mobility_n,
    )


@pytest.fixture
def tpfa_array() -> np.ndarray:
    """Array of the divergence of the TPFA array, including Dirichlet bc at the bottom."""
    return np.asarray([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 2, -1], [0, -1, -1, 2]])


def test_tpfa(tpfa_array: np.ndarray, model: TwoPhaseFlowModifiedSetup) -> None:
    """Test that TPFA works correctly."""
    subdomains = model.mdg.subdomains()
    tpfa = pp.ad.TpfaAd(model.w_flux_key, subdomains)
    div = pp.ad.Divergence(subdomains)
    tpfa_system = (div * tpfa.flux).evaluate(model.equation_system)
    assert np.allclose(tpfa_system.todense(), tpfa_array)


@pytest.fixture
def flux_pressure_val() -> np.ndarray:
    """Residual of the divergence of the pressure flux at t=0. Calculated by hand."""
    b = np.zeros(4)
    return b


@pytest.fixture
def flux_pressure_jac(tpfa_array: np.ndarray) -> np.ndarray:
    """Jacobian of the divergence of the pressure flux at t=0. Calculated by hand."""
    A_wrt_pressure_n = tpfa_array
    A = np.concatenate(
        [
            np.zeros_like(A_wrt_pressure_n),
            A_wrt_pressure_n,
            np.zeros_like(A_wrt_pressure_n),
        ],
        axis=-1,
    )
    return A


def test_flux_n(
    model: TwoPhaseFlowModifiedSetup,
    flux_pressure_val: np.ndarray,
    flux_pressure_jac: np.ndarray,
) -> None:
    """Test the residual and Jacobian of the divergence of the nonwetting flux (i.e.,
    flux dependent on nonwetting pressure).

    NOTE: We ignore mobility for this test.

    """
    subdomains = model.mdg.subdomains()
    div = pp.ad.Divergence(subdomains)
    div_flux_system = (div @ model._flux_n(subdomains)).evaluate(model.equation_system)
    A, b = div_flux_system.jac, div_flux_system.val
    assert np.allclose(A.todense(), flux_pressure_jac)
    assert np.allclose(b, -flux_pressure_val)


@pytest.fixture
def flux_cap_jac_wrt_saturation(
    tpfa_array: np.ndarray, model: TwoPhaseFlowModifiedSetup
) -> np.ndarray:
    """Jac of div of flux induced by cap. pressure w.r.t. saturation.

    NOTE: Calculated without any mobilities.

    """
    A_1 = (
        model._cap_pressure_linear_param
        / (1 - model._residual_saturation_w - model._residual_saturation_n)
    ) * tpfa_array
    A_2 = (
        model._cap_pressure_linear_param
        / (1 - model._residual_saturation_w - model._residual_saturation_n)
    ) * np.asarray([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    return A_1 - A_2


def test_flux_cap_pressure(
    model: TwoPhaseFlowModifiedSetup,
    flux_cap_jac_wrt_saturation: np.ndarray,
) -> None:
    """Test the Jacobian w.r.t. saturation of the divergence of the capillary flux.

    NOTE: We ignore mobility for this test.

    """
    subdomains = model.mdg.subdomains()
    cap_flux_tpfa = pp.ad.TpfaAd(model.cap_flux_key, subdomains)
    div = pp.ad.Divergence(subdomains)
    div_flux_system = (div @ cap_flux_tpfa.flux @ model._cap_pressure()).evaluate(
        model.equation_system
    )
    A, _ = div_flux_system.jac, div_flux_system.val
    assert np.allclose(A.todense()[:, -4:], flux_cap_jac_wrt_saturation)


@pytest.fixture
def flow_equation_val(
    source_w: np.ndarray,
) -> np.ndarray:
    """Residual of the flow equation at t=0. Calculated by hand.

    The residual equals the negative source.

    """
    return -1 * source_w


@pytest.fixture
def flow_equation_jac_wrt_pressure(
    tpfa_array: np.ndarray, mobility_t: np.ndarray
) -> np.ndarray:
    """Jacobian of the flow equation w.r.t. to :math:`p_n` at t=0. Calculated by
    hand."""
    # Total flow induced inside the domain. The total mobility is takem from one of the
    # inner interfaces (they are equal for all).
    A_1 = mobility_t[1] * tpfa_array
    # Total flow induced by the Dirichlet bc for the southern two cells. The total
    # mobility is takem from one of the two southern boundaries (they are equal for
    # both). We substract the total mobility of an inner interface, to not count the
    # flow twice (it is already included in ``A_1``).
    A_2 = (mobility_t[6] - mobility_t[1]) * np.asarray(
        [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    return A_1 + A_2


@pytest.fixture
def flow_equation_jac_wrt_saturation(
    tpfa_array: np.ndarray, mobility_w: np.ndarray, model: TwoPhaseFlowModifiedSetup
) -> np.ndarray:
    """Jacobian of the flow equation w.r.t. to :math:`S_w` at t=0. Calculated by
    hand."""
    A_1 = (
        -1
        * mobility_w[1]
        * (
            model._cap_pressure_linear_param
            / (1 - model._residual_saturation_w - model._residual_saturation_n)
        )
        * tpfa_array
    )
    A_2 = (
        mobility_w[1]
        * (
            model._cap_pressure_linear_param
            / (1 - model._residual_saturation_w - model._residual_saturation_n)
        )
        * np.asarray([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )
    return A_1 + A_2


def test_flow_equation(
    model: TwoPhaseFlowModifiedSetup,
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
def transport_equation_val(source_w: np.ndarray) -> np.ndarray:
    """Residual of the transport equation at t=0. Calculated by hand.

    The residual equals the negative source.

    """
    val = -source_w
    return val


@pytest.fixture
def transport_equation_jac_wrt_pressure(
    tpfa_array: np.ndarray, mobility_w: np.ndarray
) -> np.ndarray:
    """Jacobian of the transport equation w.r.t. to :math:`p_n` at t=0. Calculated by
    hand."""
    # Total flow induced inside the domain. The fractional flow is takem from one of the
    # inner interfaces (they are equal for all).
    A_1 = mobility_w[1] * tpfa_array
    # Total flow induced by the Dirichlet bc for the southern two cells. The total
    # mobility is takem from one of the two southern boundaries (they are equal for
    # both). We substract the total mobility of an inner interface, to not count the
    # flow twice (it is already included in ``A_1``).
    A_2 = (mobility_w[6] - mobility_w[1]) * np.asarray(
        [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    return A_1 + A_2


@pytest.fixture
def transport_equation_jac_wrt_saturation(
    tpfa_array: np.ndarray,
    mobility_t: np.ndarray,
    mobility_w: np.ndarray,
    model: TwoPhaseFlowModifiedSetup,
) -> np.ndarray:
    """Jacobian of the transport equation w.r.t. to :math:`S_w` at t=0. Calculated by
    hand."""
    A_1 = (
        model._cap_pressure_linear_param
        / (1 - model._residual_saturation_w - model._residual_saturation_n)
        * -1
        * mobility_w[1]
        * tpfa_array
    )
    # The capillary pressure induces no flow at the southern boundary (homogeneous
    # Neumann bc), hence we add the corresponding term again (TPFA calculates it).
    A_2 = (
        model._cap_pressure_linear_param
        / (1 - model._residual_saturation_w - model._residual_saturation_n)
        * mobility_w[1]
        * np.asarray([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    )
    # NOTE: The domain's porosity is homogeneous and equals 1.
    return (1.0 / model.time_manager.dt) * np.eye(A_1.shape[0]) + (A_1 + A_2)


def test_transport_equation(
    model: TwoPhaseFlowModifiedSetup,
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


# Now we repeat the last two tests with gravity.


def test_transport_equation_w_gravity(
    model_with_gravity: TwoPhaseFlowModifiedSetupGravity,
    flow_equation_val: np.ndarray,
    transport_equation_jac_wrt_pressure: np.ndarray,
    transport_equation_jac_wrt_saturation: np.ndarray,
) -> None:
    """Test the transport equation; both w.r.t. to :math:`p` and w.r.t. :math:`S`."""
    A, b = model_with_gravity.equation_system.assemble_subsystem(
        equations=["Transport equation"],
        variables=[model_with_gravity.primary_pressure_var],
    )
    assert np.allclose(A.todense(), transport_equation_jac_wrt_pressure)
    A, b = model_with_gravity.equation_system.assemble_subsystem(
        equations=["Transport equation"], variables=[model_with_gravity.saturation_var]
    )
    # Add some tolerance, as the model divides by :math:`\lambda_t + 1e-7`, but
    # multiplies by :math:`\lambda_t`. We skip this step in the exact calculation.
    assert np.allclose(A.todense(), transport_equation_jac_wrt_saturation, atol=1e-5)
    assert np.allclose(b, -flow_equation_val)
