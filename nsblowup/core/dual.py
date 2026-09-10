"""Forward-mode automatic differentiation with dual numbers (vectorised).

A `Dual` carries a value and a derivative with respect to one seed (here: time).
Every operation below propagates the derivative exactly, so `d_t u_ref` for the
reference construction is computed analytically instead of by a finite
difference: no step-size epsilon, no truncation error, no cancellation noise in
one of the largest terms of the momentum balance.

Only the operations needed by the construction are implemented:
+ - * / ** (constant exponent), exp, sin, cos, and exp(-1/x) (flat-at-zero).
"""

from __future__ import annotations

import numpy as np


class Dual:
    __slots__ = ("v", "d")

    def __init__(self, v, d=0.0):
        self.v = v
        self.d = d

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _v(other):
        return other.v if isinstance(other, Dual) else other

    @staticmethod
    def _d(other):
        return other.d if isinstance(other, Dual) else 0.0

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other):
        return Dual(self.v + self._v(other), self.d + self._d(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Dual(self.v - self._v(other), self.d - self._d(other))

    def __rsub__(self, other):
        return Dual(self._v(other) - self.v, self._d(other) - self.d)

    def __mul__(self, other):
        return Dual(self.v * self._v(other),
                    self.d * self._v(other) + self.v * self._d(other))

    __rmul__ = __mul__

    def __truediv__(self, other):
        ov, od = self._v(other), self._d(other)
        return Dual(self.v / ov, (self.d * ov - self.v * od) / ov ** 2)

    def __rtruediv__(self, other):
        ov, od = self._v(other), self._d(other)
        return Dual(ov / self.v, (od * self.v - ov * self.d) / self.v ** 2)

    def __neg__(self):
        return Dual(-self.v, -self.d)

    def __pow__(self, power):
        """Constant real exponent (bases are strictly positive here)."""
        value = self.v ** power
        return Dual(value, power * self.v ** (power - 1.0) * self.d)

    # -- elementary functions ---------------------------------------------
    def exp(self):
        value = np.exp(self.v)
        return Dual(value, value * self.d)

    def sin(self):
        return Dual(np.sin(self.v), np.cos(self.v) * self.d)

    def cos(self):
        return Dual(np.cos(self.v), -np.sin(self.v) * self.d)

    def exp_neg_inv(self):
        """phi(x) = exp(-1/x) for x > 0, 0 otherwise (C-infinity, flat at 0+)."""
        positive = np.asarray(self.v) > 0.0
        safe = np.where(positive, self.v, 1.0)
        inv = 1.0 / safe
        value = np.where(positive, np.exp(-inv), 0.0)
        deriv = np.where(positive, value * inv * inv, 0.0) * self.d
        return Dual(value, deriv)


def smooth_step(x: Dual) -> Dual:
    """S(x) = phi(x)/(phi(x) + phi(1-x)): 0 for x <= 0, 1 for x >= 1, C-infinity."""
    left = x.exp_neg_inv()
    right = (1.0 - x).exp_neg_inv()
    return left / (left + right + 1e-300)
