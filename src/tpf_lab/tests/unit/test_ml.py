import numpy as np
import porepy as pp
import scipy.sparse as sps
import torch
import torch.nn as nn

from src.tpf_lab.ml.ml_ad import nn_wrapper

# from src.porepy_adaptions.ml.nn import BaseNN


class TestNN_Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class TestNN_NonLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_nn_wrapper_scalar():
    model = TestNN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    a = pp.ad.Ad_array(3, 0)
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    assert b.jac == 0


def test_nn_wrapper_advar():
    model = TestNN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    a = pp.ad.Ad_array(3, 2)
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    # ``np.allclose`` gives an error on a sparse matrix.
    assert np.allclose(b.jac, np.multiply(fc2 @ fc1, a.jac))


def test_nn_wrapper_vector():
    model = TestNN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = pp.ad.Ad_array(val, sps.csc_matrix(J))
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    # Since fc2 @ fc1 has shape (1, 1), we can use np.multiply instead of matrix
    # multiplication. In a strict mathematical sense, the jacobian of the nn w.r.t.
    # multiple inputs is a diagonal matrix with fc2 @ fc1 as values.
    assert np.allclose(b.jac.todense(), np.multiply(fc2 @ fc1, J))


def test_nn_wrapper_vector_nonlinear():
    model = TestNN_NonLinear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = pp.ad.Ad_array(val, sps.csc_matrix(J))
    b = model_ad(a)
    assert np.allclose(b.val, fc2 @ (1 / (1 + np.exp(-fc1 @ np.expand_dims(a.val, 0)))))
    a_jac_expand = np.expand_dims(a.jac.todense(), 1)
    # See https://en.wikipedia.org/wiki/Backpropagation#Matrix_multiplication for the
    # backpropagation algorithm.
    fc1_jac = fc1.T
    fc2_jac = fc2.T
    sigmoid_jac = (
        np.exp(-fc1 @ np.expand_dims(a.val, 0))
        / (1 + np.exp(-fc1 @ np.expand_dims(a.val, 0))) ** 2
    )
    nn_jac = np.squeeze(fc1_jac @ (sigmoid_jac * fc2_jac))
    assert np.allclose(
        b.jac.todense(),
        np.diag(nn_jac) @ J,
    )
