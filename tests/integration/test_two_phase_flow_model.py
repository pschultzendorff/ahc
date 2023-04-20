"""
Test the two-phase flow model in the nonwetting pressure - wetting saturation
formulation.

The following parts of the equations are tested separately:
    - total mobility
    - wetting/nonwetting mobility
    - cap. pressure
    - total flux
    - wetting/nonwetting flux
    - flow/pressure equation
    - transport/saturation equation

Everything is tested on a 2x2 grid. The boundary conditions are homogeneous Neumann bc
at the top, right and left and homogeneous Dirichlet bc (i.e. :math:`p=0,\lambda_t=0.5`)
at the bottom. As initial conditions, the pressure is set to :math:`p=0` and the
saturation to :math:`S=0.5` on the entire grid.

NOTE: As of now, everything is tested without gravity.

NOTE: In these tests we use ``A`` to denote the Jacobian of an equation system and ``b``
to denote its residual. ``PorePy`` provides them in the form :math:`A=J` and
:math:`b=-r`, s.t. the problem can be immediately solved by :math:`A\Delta x=b`.
Thus, we need to always multiply the residual by -1.

"""


from functools import partial

import numpy as np
import porepy as pp
import pytest
import scipy.sparse as sps
from porepy.models.run_models import run_time_dependent_model
from porepy.numerics.ad.forward_mode import Ad_array

from src.tpf_lab.models.two_phase_flow import TwoPhaseFlow
from src.tpf_lab.numerics.ad import functions as af
from src.tpf_lab.numerics.ad.functions import pow


class TwoPhaseFlow_with_source(TwoPhaseFlow):
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
        return vals  #

    def _vector_source_n(self, g: pp.Grid) -> np.ndarray:
        """Zero vector volume source (gravity). Corresponds to the nonwetting buoyancy
        flow."""
        vals = np.zeros((self.mdg.dim_max(), g.num_cells))
        return vals


@pytest.fixture(scope="module")
def model() -> TwoPhaseFlow_with_source:
    model = TwoPhaseFlow_with_source({"formulation": "n_pressure_w_saturation"})
    model._cap_pressure_model = "linear"
    model._rel_perm_model = "Brooks-Corey"
    model._limit_rel_perm = "False"
    model._grid_size = 2
    model._phys_size = 2
    model.prepare_simulation()
    model.before_newton_loop()
    model.before_newton_iteration()
    return model


@pytest.fixture
def source_w() -> np.ndarray:
    return np.array([1, 0, 0, 0])


def test_source(model: TwoPhaseFlow_with_source, source_w):
    """Test that the source term is implemented correctly.

    As the nonwetting source is zero, the total source and wetting source are equal.

    """
    source_ad_t = pp.ad.ParameterArray(
        model.params_key, "source_t", model.mdg.subdomains()
    )
    source_system = source_ad_t.evaluate(model.equation_system)
    assert np.all(source_system == source_w)


@pytest.fixture
def saturation_w_init() -> np.ndarray:
    return np.array([0.5] * 4)


@pytest.fixture
def normalized_saturation_w_init(
    saturation_w_init: np.ndarray, model: TwoPhaseFlow_with_source
) -> np.ndarray:
    return (saturation_w_init - model._residual_saturation_w) / (
        1 - model._residual_saturation_w - model._residual_saturation_n
    )


@pytest.fixture
def pressure_w_init() -> np.ndarray:
    return np.array([0.0] * 4)


@pytest.fixture
def cap_pressure_val(
    normalized_saturation_w_init: np.ndarray, model: TwoPhaseFlow_with_source
) -> np.ndarray:
    """Linear cap. pressure model residual."""
    return normalized_saturation_w_init * model._cap_pressure_linear_param


@pytest.fixture
def cap_pressure_jac(
    saturation_w_init: np.ndarray, model: TwoPhaseFlow_with_source
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
    model: TwoPhaseFlow_with_source,
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
    model: TwoPhaseFlow_with_source,
    rel_perm_w: np.ndarray,
    rel_perm_n: np.ndarray,
) -> np.ndarray:
    """Total mobility for all domain interfaces at t=0. Calculated by hand.

    We differentiate between inside interfaces, boundary interfaces with homogeneous
    Dirichlet bc and boundary interfaces with homogeneous Neumann bc.

    """
    domain = model.mdg.subdomains()[0]
    # Neumann bc
    mobility_t = np.zeros(domain.num_faces)
    # Dirichlet bc. Applies at the bottom, i.e., interfaces 6 and 7. This equals double
    # the fixed boundary value we set in ``_bc_values_mobility_t`` in the model. As the
    # boundary condition is taken into account by both the wetting/nonwetting mobility.
    mobility_t[6:8] = 0.5
    # Inside interfaces. Calculated based on the initial pressure :math:`S_0=0.5`.
    # The :math:`\epsilon=1e-7` is added in the model to prevent division by zero, hence
    # we add it here as well.
    mobility_t[
        [
            1,
            4,
            8,
            9,
        ]
    ] = (
        rel_perm_w / model._viscosity_w + rel_perm_n / model._viscosity_n
    )
    return mobility_t


def test_mobility_t(model: TwoPhaseFlow_with_source, mobility_t: np.ndarray) -> None:
    """Test that the model's total mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    # Set ``atol`` high s.t. the epsilon in the total mobility does not influence the
    # comparison.
    assert np.allclose(
        model._mobility_t(subdomains).evaluate(model.equation_system).val,
        mobility_t,
        atol=1e-6,
    )


@pytest.fixture
def mobility_w(model: TwoPhaseFlow_with_source, rel_perm_w: np.ndarray) -> np.ndarray:
    """Wetting mobility for all domain interfaces at t=0. Calculated by hand.

    We differentiate between inside interfaces, boundary interfaces with homogeneous
    Dirichlet bc and boundary interfaces with homogeneous Neumann bc.

    """
    domain = model.mdg.subdomains()[0]
    # Neumann bc
    mobility_w = np.zeros(domain.num_faces)
    # Dirichlet bc. Applies at the bottom, i.e., interfaces 6 and 7. This equals the
    # fixed boundary value we set in ``_bc_values_mobility_t`` in the model.
    mobility_w[6:8] = 0.25
    # Inside interfaces. Calculated based on the initial pressure :math:`S_0=0.5`.
    mobility_w[
        [
            1,
            4,
            8,
            9,
        ]
    ] = (
        rel_perm_w / model._viscosity_w
    )

    return mobility_w


def test_mobility_w(model: TwoPhaseFlow_with_source, mobility_w: np.ndarray) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    assert np.allclose(
        model._mobility_w(subdomains).evaluate(model.equation_system).val,
        mobility_w,
    )


@pytest.fixture
def mobility_n(model: TwoPhaseFlow_with_source, rel_perm_n: np.ndarray) -> np.ndarray:
    """Nonwetting mobility for all domain interfaces at t=0. Calculated by hand.

    We differentiate between inside interfaces, boundary interfaces with homogeneous
    Dirichlet bc and boundary interfaces with homogeneous Neumann bc.

    """
    domain = model.mdg.subdomains()[0]
    # Neumann bc
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
    ] = (
        rel_perm_n / model._viscosity_n
    )
    return mobility_n


def test_mobility_n(model: TwoPhaseFlow_with_source, mobility_n: np.ndarray) -> None:
    """Test that the model's wetting mobility coincides with the one calculated by
    hand."""
    subdomains = model.mdg.subdomains()
    assert np.allclose(
        model._mobility_n(subdomains).evaluate(model.equation_system).val,
        mobility_n,
    )


@pytest.fixture
def flux_pressure_val() -> np.ndarray:
    """Residual of the divergence of the pressure flux at t=0. Calculated by hand."""
    b = np.zeros(4)
    return b


@pytest.fixture
def tpfa_array() -> np.ndarray:
    """Array of the divergence of the TPFA array, including Dirichlet bc at the bottom."""
    return np.asarray([[4, -1, -1, 0], [-1, 4, 0, -1], [-1, 0, 2, -1], [0, -1, -1, 2]])


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
    model: TwoPhaseFlow_with_source,
    flux_pressure_val: np.ndarray,
    flux_pressure_jac: np.ndarray,
) -> None:
    """Test the residual and Jacobian of the divergence of the nonwetting flux (i.e.,
    flux dependent on nonwetting pressure).

    NOTE: We ignore mobility for this test.

    """
    subdomains = model.mdg.subdomains()
    div = pp.ad.Divergence(subdomains)
    div_flux_system = (div * model._flux_n(subdomains)).evaluate(model.equation_system)
    A, b = div_flux_system.jac, div_flux_system.val
    assert np.allclose(A.todense(), flux_pressure_jac)
    assert np.allclose(b, -flux_pressure_val)


def test_vector_source(
    model: TwoPhaseFlow_with_source, flux_pressure_jac: np.ndarray
) -> None:
    subdomains = model.mdg.subdomains()
    vector_source_w = pp.ad.ParameterArray(
        model.w_flux_key, "vector_source", subdomains
    )
    div = pp.ad.Divergence(subdomains)
    flux_mpfa = pp.ad.MpfaAd(
        model.n_flux_key,
        subdomains,
    )
    buyoyancy_flux_w = flux_mpfa.vector_source * vector_source_w
    div_buyoyancy_flux_w = div * buyoyancy_flux_w
    assert True


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
    """Jacobian of the flow equation w.r.t. to :math:`p_n` at t=0. Calculated by hand."""
    # Total flow induced inside the domain. The total mobility is takem from one of the
    # inner interfaces (they are equal for all).
    A_1 = tpfa_array * mobility_t[1]
    # Total flow induced by the Dirichlet bc for the southern two cells. The total
    # mobility is takem from one of the two southern boundaries (they are equal for
    # both). We substract the total mobility of an inner interface, to not count the
    # flow twice (it is already included in ``A_1``).
    A_2 = np.asarray([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) * (
        mobility_t[6] - mobility_t[1]
    )
    return A_1 + A_2


@pytest.fixture
def flow_equation_jac_wrt_saturation() -> np.ndarray:
    """Jacobian of the flow equation w.r.t. to :math:`S_w` at t=0. Calculated by hand."""
    A = np.zeros((4, 4))
    return A


def test_flow_equation(
    model: TwoPhaseFlow_with_source,
    flow_equation_jac_wrt_pressure: np.ndarray,
    flow_equation_jac_wrt_saturation: np.ndarray,
    flow_equation_val: np.ndarray,
) -> None:
    """Test the flow equation."""
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

    Parameters:
        source_w: _description_

    """
    val = -source_w
    return val


@pytest.fixture
def transport_equation_jac_wrt_pressure(
    tpfa_array: np.ndarray, mobility_t: np.ndarray, mobility_w: np.ndarray
) -> np.ndarray:
    """Jacobian of the transport equation w.r.t. to :math:`p_n` at t=0. Calculated by
    hand."""
    # Total flow induced inside the domain. The fractional flow is takem from one of the
    # inner interfaces (they are equal for all).
    A_1 = tpfa_array * (mobility_w[1] / mobility_t[1])
    # Total flow induced by the Dirichlet bc for the southern two cells. The total
    # mobility is takem from one of the two southern boundaries (they are equal for
    # both). We substract the total mobility of an inner interface, to not count the
    # flow twice (it is already included in ``A_1``).
    A_2 = np.asarray([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) * (
        (mobility_w[6] / mobility_t[6]) - (mobility_w[1] / mobility_t[1])
    )
    return A_1 + A_2


@pytest.fixture
def transport_equation_jac_wrt_saturation(
    tpfa_array: np.ndarray, mobility_t: np.ndarray, mobility_w: np.ndarray
) -> np.ndarray:
    """Jacobian of the transport equation w.r.t. to :math:`S_w` at t=0. Calculated by
    hand."""
    # Total flow induced inside the domain. The fractional flow is takem from one of the
    # inner interfaces (they are equal for all).
    A_1 = tpfa_array * (mobility_w[1] / mobility_t[1])
    # Total flow induced by the Dirichlet bc for the southern two cells. The total
    # mobility is takem from one of the two southern boundaries (they are equal for
    # both). We substract the total mobility of an inner interface, to not count the
    # flow twice (it is already included in ``A_1``).
    A_2 = np.asarray([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) * (
        (mobility_w[6] / mobility_t[6]) - (mobility_w[1] / mobility_t[1])
    )
    return A_1 + A_2


def test_transport_equation(
    model: TwoPhaseFlow_with_source,
    flow_equation_val: np.ndarray,
    transport_equation_jac_wrt_pressure: np.ndarray,
    transport_equation_jac_wrt_saturation: np.ndarray,
) -> None:
    """Test the transport equation, both w.r.t. to :math:`p` and w.r.t. :math:`S`."""
    A, b = model.equation_system.assemble_subsystem(
        equations=["Transport equation"], variables=[model.primary_pressure_var]
    )
    assert np.allclose(A.todense(), transport_equation_jac_wrt_pressure)
    A, _ = model.equation_system.assemble_subsystem(
        equations=["Transport equation"], variables=[model.saturation_var]
    )
    # assert np.allclose(A.todense(), transport_equation_jac_wrt_saturation)
    assert np.allclose(b, -flow_equation_val)
