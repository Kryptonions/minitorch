"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # print(param)
            if param.grad is None:
                continue
            grad = self.u.get(param, 0) * self.momentum + (1 - self.momentum) * (
                        param.grad.detach() + self.weight_decay * param.detach())
            grad = ndl.Tensor(grad, dtype=param.dtype)
            self.u[param] = grad
            param.data -= self.lr * grad
            ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
            self,
            params,
            lr=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data + self.weight_decay * p.data
            v = self.beta1 * self.v.get(p, 0) + (1 - self.beta1) * grad
            m = self.beta2 * self.m.get(p, 0) + (1 - self.beta2) * grad * grad
            self.v[p] = ndl.Tensor(v, dtype=p.dtype)
            self.m[p] = ndl.Tensor(m, dtype=p.dtype)
            # bias correction
            v = v / (1 - self.beta1 ** self.t)
            m = m / (1 - self.beta2 ** self.t)
            update = ndl.ops.divide(v, ndl.ops.power_scalar(m, 0.5) + self.eps)
            p.data = p.data - self.lr * ndl.Tensor(update, dtype=p.dtype)
