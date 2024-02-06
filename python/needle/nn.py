"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.use_bias = bias
        w = init.kaiming_uniform(in_features, out_features)
        self.weight = Parameter(w, device=device, dtype=dtype)
        if self.use_bias:
            b = ops.reshape(init.kaiming_uniform(out_features, 1), (1, out_features))
            self.bias = Parameter(b, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = ops.matmul(X, self.weight)
        if self.use_bias:
            # 只能用Y.shape, 因为可能高维
            bias = ops.broadcast_to(self.bias, Y.shape)
            Y += bias
        return Y
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        return ops.reshape(X, (batch_size, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        input = x
        for layer in self.modules:
            output = layer.forward(input)
            input = output
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        n = 1.0 * logits.shape[0]
        m = logits.shape[-1]
        y_one_hot = init.one_hot(m, y)
        z_y = ops.summation(ops.multiply(logits, y_one_hot), axes=(-1,))

        return ops.divide_scalar(ops.summation(ops.logsumexp(logits, axes=(-1,)) - z_y), float(n))
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        self.running_mean = Parameter(init.zeros(self.dim))
        self.running_var = Parameter(init.ones(self.dim))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feat_dim = x.shape
        if self.training:
            mean = ops.summation(x, axes=(0,)) / batch_size
            mean_reshape = ops.reshape(mean, (1, feat_dim))
            mean_broadcast = ops.broadcast_to(mean_reshape, x.shape)
            var = ops.summation(ops.power_scalar(x - mean_broadcast, 2), (0,)) / batch_size
            var_reshape = ops.reshape(var, (1, feat_dim))
            var_broadcast = ops.broadcast_to(var_reshape, x.shape)

            w = ops.broadcast_to(self.weight, x.shape)
            b = ops.broadcast_to(self.bias, x.shape)

            y = w * (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5) + b

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

            mean_reshape = ops.reshape(mean, (1, feat_dim))
            var_reshape = ops.reshape(var, (1, feat_dim))

            mean_broadcast = ops.broadcast_to(mean_reshape, x.shape)
            var_broadcast = ops.broadcast_to(var_reshape, x.shape)

            w = ops.broadcast_to(self.weight, x.shape)
            b = ops.broadcast_to(self.bias, x.shape)
            y = w * (x - mean) / ops.power_scalar(var + self.eps, 0.5) + b
        return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        #self.weight = Parameter(init.costant(self.dim, c=1.0))  # why error?
        #self.bias = Parameter(init.constant(self.dim, c=0.0))
        self.weight = Parameter(init.ones(self.dim))
        self.bias = Parameter(init.zeros(self.dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        m, n = x.shape

        mean = ops.reshape(ops.summation(x, axes=(1,)) / n, (m, 1))
        mean = ops.broadcast_to(mean, x.shape)

        var = ops.reshape(ops.summation(ops.power_scalar((x - mean), 2), axes=(1,)) / n, (m, 1))
        var = ops.broadcast_to(var, x.shape)

        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)
        y = w * (x - mean) / ops.power_scalar(var + self.eps, 0.5) + b
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION