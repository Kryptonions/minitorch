"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from .. import init
import numpy

from ..backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return node.inputs[0]**(self.scalar - 1) * out_grad * self.scalar



def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad * 1 / rhs, out_grad * lhs * (-1) / (rhs * rhs)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar,)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            ax0, ax1 = self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        permute_axes = list(range(a.ndim))
        permute_axes[ax0], permute_axes[ax1] = ax1, ax0
        return a.permute(permute_axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        input, = node.inputs
        return reshape(out_grad, input.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input, = node.inputs
        input_shape = input.shape
        output_shape = self.shape

        expand = len(output_shape) - len(input_shape)
        for i in range(expand):
            out_grad = summation(out_grad, axes=(0))
        axes = []
        for i in range(len(input_shape)):
            if input_shape[i] != output_shape[i + expand]:
                axes.append(i)
        return reshape(summation(out_grad, axes=tuple(axes)), input_shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes, )
        self.axes = axes

    """
    def compute(self, a):
        return a.sum(self.axes)
    """ 
    def compute(self, a):
        n = len(a.shape)
        axes = []
        # 处理多维度求和
        if not isinstance(self.axes, tuple):
            ori_axes = self.axes,
        else:
            ori_axes = self.axes
        for axis in ori_axes:
            # 处理负数情形
            if isinstance(axis, int):
                if axis < 0:
                    axes.append(axis + n)
                else:
                    axes.append(axis)
            else:
                axes.append(axis)
        # 降序排列
        axes = sorted(axes, reverse=True)
        for axis in axes:
            a = array_api.sum(a, axis)
        
        return a

    def gradient(self, out_grad, node):
        input, = node.inputs
        input_shape = input.shape
        output_shape = out_grad.shape
        new_shape = list(input_shape)
        n = len(input.shape)
        if self.axes is None:
            axes = list(range(len(input_shape)))
        else:
            axes = self.axes
        for i in range(len(axes)):
            new_shape[axes[i] if axes[i] >= 0 else axes[i] + n] = 1
        out_grad = reshape(out_grad, tuple(new_shape))
        return broadcast_to(out_grad, input_shape)

    """
    def gradient(self, out_grad, node):
        input, = node.inputs
        # 使坐标为正并且从小到大排列
        if self.axes == None:
            axes = input.shape
            grad_shape = []
        else:
            axes = self.axes
            grad_shape = list(out_grad.shape)

        n = len(input.shape)
        new_axes = []
        for x in axes:
            if x >= 0:
                new_axes.append(x)
            else:
                new_axes.append(x + n)
        new_axes = sorted(new_axes)
        # 恢复grad_shape, 使grad_shape的维度和input.shape的维度相同
        for axis in new_axes:
            grad_shape.insert(axis, 1)

        return broadcast_to(reshape(out_grad, grad_shape), input.shape)
    """
def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b  # (a, b, c) x (d, a, c, f) => (d, a, b, f)
        ###          # (a, b, c, d) x (b, d, f) => (a, b, c, f)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        dlhs = matmul(out_grad, transpose(rhs))
        drhs = matmul(transpose(lhs), out_grad)
        if dlhs.shape != lhs.shape:
            dlhs = summation(dlhs, tuple(range(len(dlhs.shape) - len(lhs.shape))))
        if drhs.shape != rhs.shape:
            drhs = summation(drhs, tuple(range(len(drhs.shape) - len(rhs.shape))))
        assert (dlhs.shape == lhs.shape)
        assert (drhs.shape == rhs.shape)
        return dlhs, drhs


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -1 * a

    def gradient(self, out_grad, node):
        return mul_scalar(out_grad, -1)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        input, = node.inputs
        return out_grad / input


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        input, = node.inputs
        return out_grad * exp(input)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        out = node.realize_cached_data()
        return out_grad * Tensor(out > 0, device=out_grad.device)



def relu(a):
    return ReLU()(a)



class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        input, = node.inputs
        tmp = tanh(input)
        return out_grad * (1 - tmp * tmp)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        assert len(args) > 0, "Stack needs at least one array!"
        shape = args[0].shape
        for a in args:
            assert shape == a.shape, "All arrays need to be of the same size!"
        n = len(args)
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        out = array_api.empty(new_shape, device=args[0].device)
        slices = [slice(0, s) for s in new_shape]
        for i, arr in enumerate(args):
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr
        return out

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)

def stack(args, axis):
    return Stack(axis)(make_tuple(*args))
    
class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []
        for i in range(n):
            slices[self.axis] = slice(i, i+1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)

    def gradient(self, out_grad, node):
        return stack(out_grad, self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        new_shape = tuple(new_shape)
        arr = a.device.full(new_shape, 0)
        slices = [slice(0, n) for n in arr.shape]
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, arr.shape[axis], self.dilation + 1)
        arr[tuple(slices)] = a
        return arr

    def gradient(self, out_grad, node):
        return UnDilate(self.axes, self.dilation)(out_grad)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        slices = [slice(0, n) for n in a.shape]
        for axis in self.axes:
            if axis >= len(a.shape):
                continue
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        return a[tuple(slices)].compact()

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        A = A.pad(((0,0), (self.padding, self.padding), (self.padding, self.padding),(0,0)))
        N, H, W, C_in = A.shape
        K, K_, C_in_, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        assert K == K_, "Conv kernel should be a square tensor"
        assert C_in == C_in_, "Conv kernel and input are not compatible"
        
        inner_dim = K * K * C_in
        out_H, out_W = (H-K+1)//self.stride, (W-K+1)//self.stride
        # compact() cannot be deleted 
        im2col = A.as_strided(shape=(N, out_H, out_W, K, K, C_in),
                              strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs))\
                              .compact()\
                              .reshape((N*out_H*out_W, inner_dim))
        # we need to compact B, othereise will raise error
        out = im2col @ B.compact().reshape((inner_dim, C_out))
        return out.reshape((N, out_H, out_W, C_out))

    def gradient(self, out_grad, node):
        X, W = node.inputs
        K, _, _, _ = W.shape

        if self.stride > 1:
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        W_permute = transpose(flip(W, (0, 1)), (2, 3)) # K * K * C_out * C_in
        # out_grad: # N * (H+2P-K+1) * (W+2P-K+1) * C_out
        X_grad = conv(out_grad, W_permute, padding=K-1-self.padding)

        X_permute = transpose(X, (0, 3)) # C_in * H * W * N
        grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2)) # (H+2P-K+1) * (W+2P-K+1) * N * C_out
        W_grad = conv(X_permute, grad_permute, padding=self.padding) # C_in * H * W * C_out
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2)) # H * W * C_in * C_out

        return X_grad, W_grad


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
