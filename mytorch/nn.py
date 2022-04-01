from .cuda_ops import CudaOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    a1 = input.contiguous()
    i1 = a1.view(batch, channel, height, new_width, kw)
    i2 = i1.permute(0, 1, 3, 2, 4)
    a2 = i2.contiguous()
    i3 = a2.view(batch, channel, new_height, new_width, kh * kw)

    return i3, new_height, new_width


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    input, new_h, new_w = tile(input, kernel)
    out = input.mean(dim=4)

    return out.view(batch, channel, new_h, new_w)


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


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape

    inp, new_h, new_w = tile(input, kernel)
    out = max(inp, 4)

    return out.view(batch, channel, new_h, new_w)


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with random positions dropped out
    """

    if ignore:
        return input
    else:
        keep = 1 - rate
        mask = rand(input.shape)
        return input * (mask < keep)
