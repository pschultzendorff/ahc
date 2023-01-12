import numpy as np
import porepy as pp
import torch
import torch.nn as nn

# The former import works for testing, the latter for running this file. Is there a way
# to combine both?
# import porepy_adaptions.ml.pp_nn as pp_nn

# import pp_nn


def nn_wrapper(network: nn.Module):
    """Wraps a neural network s.t. it can be used as a ``porepy.ad.functions`` function.

    Parameters:
        network: Neural network. The ``ad`` module does not check that the input and
        output sizes are correct, hence this needs to be ensured by the user.
    """

    def inner(var):
        if isinstance(var, pp.ad.Ad_array):
            if isinstance(var.val, np.ndarray):
                var_tensor = torch.from_numpy(var.val).to(dtype=torch.float)
                var_tensor.requires_grad = True
            else:
                # ? Can ``var.val`` also be a list, or is it always a float value if
                # it's not an ``ndarrray``?
                var_tensor = torch.tensor(
                    [var.val], dtype=torch.float, requires_grad=True
                )
            # Unsqueeze to transform into a batch.
            val = network(var_tensor.unsqueeze(-1))
            # ! ``.backward()`` can only be called if the output is a scalar. Hence, we
            # sum over the output vector and use the fact that by the chain rule
            # .. math:
            #       \frac{\partial\sum_i\mathcal{N}(x_i)}{\partial
            #       x_j}=\frac{\partial\sum_i\mathcal{N}(x_i)}{\partial
            #       \mathcal{N}(x_i)}\frac{\partial\mathcal{N}(x_i)}{\partial
            #       x_j}=\frac{\partial\mathcal{N}(x_i)}{\partial x_j},
            #
            # the derivative of the sum of all outputs w.r.t. to a single input is the
            # same as the derivative of one output w.r.t. to that input as a workaround.
            torch.sum(val).backward()
            nn_der = var_tensor.grad.squeeze().numpy(force=True)
            full_der = var.diagvec_mul_jac(nn_der)
            return pp.ad.Ad_array(val.squeeze().numpy(force=True), full_der)
        else:
            # No gradient tracking needed if we are not using it anyways.
            with torch.no_grad():
                return (
                    network(
                        torch.from_numpy(var.val).to(dtype=torch.float).unsqueeze(-1)
                    )
                    .squeeze()
                    .numpy(force=True)
                )

    return inner


# def model_prep(network: pp_nn.BaseNN):
#     """Prepare the model for fast evaluation.

#     Change the network to ``eval`` mode and turn off ``requires_grad`` for its
#     parameters.

#     Parameters:
#         network: _description_
#     """
#     network.eval()
#     for params in network.parameters():
#         params.requires_grad = False
