import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def tensor_map(fn):
    """
    Higher-order tensor map function ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        out_index = np.zeros(MAX_DIMS, np.int32)
        in_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            count(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return _map


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      b = fn_map(a)


    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Higher-order tensor zipWith (or map2) function. ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)


    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):

        out_index = np.zeros(MAX_DIMS, np.int32)
        a_index = np.zeros(MAX_DIMS, np.int32)
        b_index = np.zeros(MAX_DIMS, np.int32)

        for i in range(len(out)):
            count(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return _zip


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def get_current_idx(out_index, target_dim, num):
    current_idx = []
    for i, v in enumerate(out_index):
        if i != target_dim:
            current_idx.append(out_index[i])
        else:
            current_idx.append(num)
    return current_idx


def tensor_reduce(fn):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = tensor_reduce(fn)
      c = fn_reduce(out, ...)

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,
        reduce_size,
    ):
        out_index = np.zeros(MAX_DIMS, np.int32)
        a_index = np.zeros(MAX_DIMS, np.int32)

        for i in range(len(out)):
            count(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            for s in range(reduce_size):
                count(s, reduce_shape, a_index)
                for w in range(len(reduce_shape)):
                    if reduce_shape[w] != 1:
                        out_index[w] = a_index[w]

                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[j])

    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      reduced = fn_reduce(a, dims)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`TensorData`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`TensorData`, optional): tensor to reduce into

    Returns:
        :class:`TensorData` : new tensor data
    """

    f = tensor_reduce(fn)
    def ret(a, dims=None, out=None):
        old_shape = None
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
        else:
            old_shape = out.shape
            diff = len(a.shape) - len(out.shape)
            out = out.view(*([1] * diff + list(old_shape)))

        assert len(out.shape) == len(a.shape)
        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if out.shape[i] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)
        f(*out.tuple(), *a.tuple(), reduce_shape, reduce_size)

        if old_shape is not None:
            out = out.view(*old_shape)
        return out

    return ret


def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    iteration_n = a_shape[-1]

    for i in range(len(out)):
        out_index = np.zeros(MAX_DIMS, np.int32)
        count(i, out_shape, out_index)
        o = index_to_position(out_index, out_strides)
        a_index = np.copy(out_index)
        b_index = np.zeros(MAX_DIMS, np.int32)
        a_index[len(out_shape) - 1] = 0
        b_index[len(out_shape) - 2] = 0
        b_index[len(out_shape) - 1] = out_index[len(out_shape) - 1]
        temp_sum = 0
        for w in range(iteration_n):
            a_index[len(out_shape) - 1] = w
            b_index[len(out_shape) - 2] = w

            j = index_to_position(a_index, a_strides)
            m = index_to_position(b_index, b_strides)
            temp_sum = temp_sum + a_storage[j] * b_storage[m]

        out[o] = temp_sum


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Create out shape
    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    out = a.zeros(tuple(ls))
    # Call main function
    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    return out


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
