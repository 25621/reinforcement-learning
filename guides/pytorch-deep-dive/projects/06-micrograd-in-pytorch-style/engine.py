"""A scalar autograd engine, written the way PyTorch's is organised.

Every `Value` holds one number and remembers which operation produced it and
from which inputs. That record IS the computation graph: it is built as a side
effect of doing the forward arithmetic, and it is walked backwards to get
gradients.

The design mirrors PyTorch:

    Value.data        <->  Tensor  (the number itself)
    Value.grad        <->  Tensor.grad
    Value._backward   <->  the Node's backward function (grad_fn)
    Value._prev       <->  grad_fn.next_functions (the edges)
    Value._op         <->  the name shown by grad_fn ("AddBackward0", ...)
    Value.backward()  <->  Tensor.backward()

Imported by 07-11 as a reference implementation is NOT the point; this file
exists so project 06 can compare it against torch, number for number.
"""

import math


class Value:
    """One scalar, plus the edge back to whatever produced it."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0             # d(final output) / d(self), filled by backward()
        self._backward = lambda: None   # how to push grad from self to _prev
        self._prev = tuple(_children)   # the inputs this node was built from
        self._op = _op                  # what operation made it (for printing)
        self.label = label

    # ---------------- forward ops: each one records its own backward -------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  ->  the upstream grad passes through
            # unchanged to BOTH inputs. Note "+=", never "=": see accumulate().
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a  -> each input's grad is scaled by
            # the OTHER input's value. This is why forward values must be kept
            # alive until backward runs (PyTorch: ctx.save_for_backward).
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only scalar exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            # d(exp(x))/dx = exp(x) = the OUTPUT. Saving the output instead of
            # recomputing exp is the same trick project 08 uses for sigmoid.
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # The derivative at exactly 0 does not exist. PyTorch picks 0 here;
            # we match it so the comparison in run.py is exact.
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ---------------- sugar built out of the ops above ---------------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f}, op={self._op or 'leaf'})"

    # ---------------- backward ---------------------------------------------

    def topo_order(self):
        """Every node this one depends on, inputs always before outputs.

        Depth-first post-order: a node is appended only after all of ITS inputs
        have been appended, so the returned list has inputs before outputs.
        Reversing it gives outputs before inputs -- the order backward needs.

        Written with an explicit stack rather than recursion on purpose. A
        recursive version is three lines shorter and dies with RecursionError on
        a graph deeper than ~1000 nodes (Python's default limit), which a plain
        `for` loop summing 20,000 terms reaches easily. PyTorch's engine keeps
        its own worklist in C++ for the same reason.
        """
        order, seen = [], set()
        stack = [(self, False)]         # (node, have its children been queued?)
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)      # all children already appended
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))  # revisit after the children
            for child in node._prev:
                stack.append((child, False))
        return order

    def backward(self, order=None):
        """Fill .grad on every node this one depends on."""
        topo = self.topo_order() if order is None else order
        self.grad = 1.0             # d(self)/d(self) = 1: the seed
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        for node in self.topo_order():
            node.grad = 0.0
