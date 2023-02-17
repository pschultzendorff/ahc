import numpy as np
import porepy as pp
import torch
import torch.nn as nn
import scipy.sparse as sps


def inner(network, var):
    if isinstance(var, pp.ad.Ad_array):
        # ? Does ``.from_numpy`` set ``requires_grad`` to true?
        if isinstance(var.val, np.ndarray):
            # Unsqueeze to transform into a batch.
            var_tensor = torch.from_numpy(var.val).to(dtype=torch.float).unsqueeze(-1)
            var_tensor.requires_grad = True
            val = network(var_tensor)
            # ! ``.backward()`` can only be called if the output is a scalar. Hence, we
            # sum over the output vector and use the fact that by the chain rule
            # .. math:
            #       \frac{\sum_i\partial\mathcal{N}(x_i)}{\partial
            #       x_j}=\frac{\sum_i\partial\mathcal{N}(x_i)}{\partial
            #       \mathcal{N}(x_i)}\frac{\partial\mathcal{N}(x_i)}{\partial
            #       x_j}=\frac{\partial\mathcal{N}(x_i)}{\partial x_j},
            #
            # the derivative of the sum of all outputs w.r.t. to a single input is the
            # same as the derivative of one output w.r.t. to that input as a workaround.
            torch.sum(val).backward()
        else:
            # ? Can ``var.val`` also be a list, or is it always a float value if
            # it's not an ``ndarrray``?
            var_tensor = torch.tensor([var.val], dtype=torch.float, requires_grad=True)
            foo = var_tensor.unsqueeze(-1)
            val = network(foo)
            val.backward()
        der_tensor = var_tensor.grad.squeeze().numpy(force=True)
        der = var.diagvec_mul_jac(der_tensor)
        return pp.ad.Ad_array(val.squeeze().numpy(force=True), der)
    else:
        # No gradient tracking needed if we are not using it anyways.
        with torch.no_grad():
            # TODO: Fix ``force=True`` in the numpy call. See above!
            return (
                network(torch.from_numpy(var.val).to(dtype=torch.float).unsqueeze(-1))
                .squeeze()
                .numpy(force=True)
            )


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


model = TestNN_Linear()
fc1 = list(model.fc1.parameters())[0].numpy(force=True)
fc2 = list(model.fc2.parameters())[0].numpy(force=True)


a = pp.ad.Ad_array(3, 0)
b = inner(model, a)
assert np.allclose(b.val, np.multiply(fc2 @ fc1, a.val))
assert b.jac == 0

model = TestNN_NonLinear()
fc1 = list(model.fc1.parameters())[0].numpy(force=True)
fc2 = list(model.fc2.parameters())[0].numpy(force=True)

val = np.array([1, 2, 3])
J = np.array([[3, 2, 1], [5, 6, 1], [2, 3, 5]])
a = pp.ad.Ad_array(val, sps.csc_matrix(J))
b = inner(model, a)

assert np.allclose(
    b.val,
    fc2 @ (1 / (1 + np.exp(-fc1 @ np.expand_dims(a.val, 0)))),
)
