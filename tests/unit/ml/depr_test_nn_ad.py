"""This test file is deprecated until the ``tpf.ml`` functionality is updated."""

import numpy as np
import pytest
import scipy.sparse as sps
import torch
import torch.nn as nn
from porepy.numerics.ad.forward_mode import AdArray
from tpf.ml.nn_ad import nn_wrapper


class NN_Linear(nn.Module):
    """Simple two-layer linear neural network for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class NN_NonLinear(nn.Module):
    """Simple two-layer neural network for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(1, 2, bias=False)
        self.fc2 = nn.Linear(2, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_nn_wrapper_scalar():
    """Test ``nn_wrapper`` for inputs of type ``Ad_array(int, 0)``."""
    model = NN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    a = AdArray(np.array([3]), np.array([[0]]))
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    assert b.jac == 0


def test_nn_wrapper_advar():
    """Test ``nn_wrapper`` for inputs of type ``Ad_array(int, int)``."""
    model = NN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    a = AdArray(np.array([3]), np.array([[2]]))
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    # ``np.allclose`` gives an error on a sparse matrix.
    assert np.allclose(b.jac, np.multiply(fc2 @ fc1, a.jac))


def test_nn_wrapper_advar_additional_args():
    """Test ``nn_wrapper`` for inputs of type ``Ad_array(int, int)`` with optional
    ``*args``."""
    # TODO: Write this test.
    model = NN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    a = AdArray(np.array([3]), np.array([[2]]))
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    # ``np.allclose`` gives an error on a sparse matrix.
    assert np.allclose(b.jac, np.multiply(fc2 @ fc1, a.jac))


def test_nn_wrapper_vector():
    """Test ``nn_wrapper`` for inputs of type ``Ad_array(np.ndarray, np.ndarray)``."""
    model = NN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    # Since fc2 @ fc1 has shape (1, 1), we can use np.multiply instead of matrix
    # multiplication. In a strict mathematical sense, the jacobian of the nn w.r.t.
    # multiple inputs is a diagonal matrix with fc2 @ fc1 as values.
    assert np.allclose(b.jac.todense(), np.multiply(fc2 @ fc1, J))


def test_nn_wrapper_vector_optional_args():
    """Test ``nn_wrapper`` for inputs of type ``Ad_array(np.ndarray, np.ndarray)`` with
    optional ``*args``."""
    # TODO: Write this test.
    model = NN_Linear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
    b = model_ad(a)
    assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
    # Since fc2 @ fc1 has shape (1, 1), we can use np.multiply instead of matrix
    # multiplication. In a strict mathematical sense, the jacobian of the nn w.r.t.
    # multiple inputs is a diagonal matrix with fc2 @ fc1 as values.
    assert np.allclose(b.jac.todense(), np.multiply(fc2 @ fc1, J))


def test_nn_wrapper_vector_nonlinear():
    """Test ``nn_wrapper`` for inputs of type ``Ad_array(np.ndarray, np.ndarray)``.

    In comparison to the previous two tests, the wrapped neural networks has a nonlinear
    activation function in this test.

    """
    model = NN_NonLinear()
    fc1 = list(model.fc1.parameters())[0].numpy(force=True)
    fc2 = list(model.fc2.parameters())[0].numpy(force=True)

    model_ad = nn_wrapper(model)
    val = np.array([1, 2, 3])
    J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
    a = AdArray(val, sps.csc_matrix(J))
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
