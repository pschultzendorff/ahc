import numpy as np
import pytest
from tpf_lab.models.reconstructions import (
    EquilibratedFluxMixin,
    PressureReconstructionMixin,
)


def test_reconstruction():
    """Check if mass conservation is satisfied on a cell basis, in order to do
    this, we check on a local basis, if the divergence of the flux equals
    the sum of internal and external source terms

    """
    full_flux_local_div = (sign_normals_cell * flux[faces_cell]).sum(axis=1)
    external_src = d[pp.PARAMETERS][self.kw]["source"]
    np.testing.assert_allclose(
        full_flux_local_div,
        external_src + mortar_jump,
        rtol=1e-6,
        atol=1e-3,
        err_msg="Error estimates only valid for local mass-conservative methods.",
    )
