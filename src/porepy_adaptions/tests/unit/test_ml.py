import numpy as np
import porepy as pp
import scipy.sparse as sps
import torch
import torch.nn as nn

from porepy_adaptions.ml.ml_ad import nn_wrapper

# from src.porepy_adaptions.ml.nn import BaseNN


class TestNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def test_nn_wrapper_scalar():
    model = TestNN()
    # TODO: Replace ``detach()`` after finding out why ``force=True`` does not work.
    # Right now one gets:
    # TypeError: Parameter.numpy() takes no keyword arguments
    fc1 = list(model.fc1.parameters())[0].detach().numpy()
    fc2 = list(model.fc2.parameters())[0].detach().numpy()

    model_ad = nn_wrapper(model)
    a = pp.ad.Ad_array(3, 0)
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    assert b.jac == 0


def test_nn_wrapper_advar():
    model = TestNN()
    # TODO: Replace ``detach()`` after finding out why ``force=True`` does not work.
    # Right now one gets:
    # TypeError: Parameter.numpy() takes no keyword arguments
    fc1 = list(model.fc1.parameters())[0].detach().numpy()
    fc2 = list(model.fc2.parameters())[0].detach().numpy()

    model_ad = nn_wrapper(model)
    a = pp.ad.Ad_array(3, 2)
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    # ``np.allclose`` gives an error on a sparse matrix.
    assert np.allclose(b.jac.todense(), np.multiply(fc2 @ fc1, a.jac))


def test_nn_wrapper_vector():
    model = TestNN()
    # TODO: Replace ``detach()`` after finding out why ``force=True`` does not work.
    # Right now one gets:
    # TypeError: Parameter.numpy() takes no keyword arguments
    fc1 = list(model.fc1.parameters())[0].detach().numpy()
    fc2 = list(model.fc2.parameters())[0].detach().numpy()

    model_ad = nn_wrapper(model)
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = pp.ad.Ad_array(val, sps.csc_matrix(J))
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    assert np.allclose(b.jac.todense(), np.multiply(fc2 @ fc1, a.jac.todense()))
