"""Machinery to include ``pytorch`` neural networks in ``PorePy``."""


from typing import Any, Callable, TypeVar

import numpy as np
import porepy as pp
import torch
import torch.nn as nn

InputArrayType = TypeVar("InputArrayType", bound=pp.ad.AdArray | np.ndarray)


def nn_wrapper(
    network: nn.Module,
) -> Callable[[InputArrayType, list[Any]], InputArrayType]:
    """Wraps a neural network s.t. it can be used as a ``porepy.ad.functions`` function.

    The wrapped network gets passed ``var`` (which is an ``pp.ad.AdArray``), which is
    then converted to a ``torch.Tensor`` and passed to the network. The output and
    Jacobian are computed, converted back to an ``pp.ad.AdArray`` and returned.

    Additional inputs to the network can be passed as ``*args``. These are concatenated
    with ``var.val`` and passed to the network. The Jacobian of the network is
    calculated only w.r.t. to ``var.val``.

    Note: It is the users responsibility of the user to ensure that any ``np.ndarrays``
        or ``torch.Tensors`` in ``*args`` have the correct shape.

    Parameters:
        network: Neural network. The ``ad`` module does not check that the input and
        output sizes are correct, hence this needs to be ensured by the user.

    """

    def inner(var: InputArrayType, *args) -> InputArrayType:
        # Prepare ``*args``.
        args_tensors: list[torch.Tensor] = []
        for arg in args:
            args_tensors.append(torch.as_tensor(arg))

        if isinstance(var, pp.ad.AdArray):
            if isinstance(var.val, np.ndarray):
                var_tensor = torch.from_numpy(var.val).to(dtype=torch.float)
            else:
                # ? Can ``var.val`` also be a list, or is it always a float value if
                # it's not an ``ndarrray``?
                var_tensor = torch.as_tensor([var.val], dtype=torch.float)
            var_tensor.requires_grad = True

            # Try to concatenate with additional inputs. Catch if the tensor shapes
            # do not align.
            # Unsqueeze to transform into a batch of size 1.
            args_tensors.insert(0, var_tensor)
            try:
                input = torch.cat(args_tensors).unsqueeze(-1)
            except Exception as e:
                # TODO: Fix this exception.
                pass

            val = network(input)

            # ! ``.backward()`` can only be called if the output is a scalar. Hence, we
            # sum over the output vector and use the fact that by the chain rule
            #
            # .. math:
            #       \frac{\partial\sum_i\mathcal{N}(x_i)}{\partial
            #       x_j}=\frac{\partial\sum_i\mathcal{N}(x_i)}{\partial
            #       \mathcal{N}(x_i)}\frac{\partial\mathcal{N}(x_i)}{\partial
            #       x_j}=\frac{\partial\mathcal{N}(x_i)}{\partial x_j},
            #
            # the derivative of the sum of all outputs w.r.t. to a single input is the
            # same as the derivative of one output w.r.t. to that input as a workaround.
            torch.sum(val).backward()

            if var_tensor.grad is not None:
                nn_der = var_tensor.grad.squeeze().numpy(force=True)
                full_der = var._diagvec_mul_jac(nn_der)
            else:
                # In case something goes wrong with the autograd.
                full_der = np.zeros((var.val.shape[0], val.shape[-1]))
            return pp.ad.AdArray(val.squeeze().numpy(force=True), full_der)  # type: ignore
        else:
            # No gradient tracking needed if we are not using it anyways.
            with torch.no_grad():
                #
                var_tensor = torch.from_numpy(var).to(dtype=torch.float)
                args_tensors.insert(0, var_tensor)

                # Try to concatenate with additional inputs. Catch if the tensor shapes
                # do not align.
                # Unsqueeze to transform into a batch of size 1.
                try:
                    input = torch.cat(args_tensors).unsqueeze(-1)
                except Exception as e:
                    # TODO: Fix this exception.
                    pass
                return network(input).squeeze().numpy(force=True)

    return inner


def set_nn_to_eval(network: nn.Module) -> None:
    """Prepare the network for fast evaluation.

    Change the network to ``eval`` mode and turn off ``requires_grad`` for its
    parameters.

    Parameters:
        network: _description_

    """
    network.eval()
    for params in network.parameters():
        params.requires_grad = False
