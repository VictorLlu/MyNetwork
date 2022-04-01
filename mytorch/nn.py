from .cuda_ops import CudaOps
from .tensor_functions import rand, Function
from . import operators


max_reduce = CudaOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, [dim])
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        # "Forward of max should be max reduction"
        if dim is not None:
            out = max_reduce(input, [dim])
            # dim = [dim]
        else:
            out = max_reduce(input, list(input.dims)).view(1)
            dim = list(input.dims)[0]
        ctx.save_for_backward(input, dim)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, dim = ctx.saved_values
        # am = argmax(inp, dim)
        rand_t = rand(inp.shape)
        return argmax(rand_t + inp, dim) * grad_output


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    x_softmax = input.exp() / input.exp().sum(dim=dim)
    return x_softmax


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    sum_exp = input.exp().sum(dim=dim)
    log_sum_exp = sum_exp.log()
    return input - log_sum_exp
