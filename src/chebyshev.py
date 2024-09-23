import torch
import numpy as np
from dataclasses import dataclass
import functools
import warnings


@dataclass
class ChebyshevMagicNumbers:
    r"""
    Magic numbers used in order to iteratively construct a desirable Chebyshev
    polynomial

    Let $T_n(x)$ be the $n$th-order Chebyshev polynomial of the first kind,
    defined recursively as follows:
        $T_0(x) = 1$
        $T_1(x) = x$
        $T_n(x) = 2 x T_{n-1}(x) - T_{n-2}(x) ~ \forall n \geq 2$

    Let $\alpha \in (0, 1)$ be some chosen threshold, $r_n :=
    \cos(\frac{\pi}{2 n})$ be the largest root of $T_n$, and $z_n := \frac{1 +
    r_n}{\alpha}$. Let $f_n(y)$ be defined then as follows:
        $f_0(y) = 1$
        $f_n(y) = \frac{T_n(z_n y - r_n)}{T_n(z_n - r_n)} ~ \forall n \geq 1$

    The functions $f_n$ may be (very-good-approximately) constructed
    iteratively using the magic numbers (a_n, b_n, c_n) as follows:
        $f_0(y) = 1$
        $f_1(y) = y$
        $f_n(y) \approx a_n ((y + b_n) f_{n-1}(y) + c_n f_{n-2}(y))
                ~ \forall n \geq 2$

    The magic numbers (a_n, b_n, c_n) are chosen such that the approximate
    iterated polynomial $f_n$ matches the original polynomial exactly in its 3
    highest degree terms (i.e. $y^n$, $y^{n-1}$, and $y^{n-2}$). For example,
    $f_2$ is perfectly reconstructed and $f_3$ is the first actual
    approximation.

    """
    alpha : float
    def __post_init__(self):
        if not isinstance(self.alpha, float):
            raise ValueError(
                f'{alpha=} is not a float. alpha must be a float to ensure'
                ' 64-bit type for numerical stability'
            )
        lower_lim = 1e-6
        if lower_lim > self.alpha:
            warnings.warn(
                'ChebyshevMagicNumbers: Due to numerical instabilities in the'
                ' computation of magic number a, alpha values less than'
                f' {lower_lim} are rounded up'
            )
            self.alpha = lower_lim
    def __hash__(self):
        return hash(self.alpha)
    def __eq__(self, other):
        return self.alpha == other.alpha
    def get_root_arr(self, n):
        n_arr = torch.arange(n)
        root_arr = torch.cos((torch.pi / n) * (n_arr + 0.5))
        root_arr = (root_arr + root_arr[0]) / (1 + root_arr[0])
        root_arr *= self.alpha
        return root_arr
    @staticmethod
    @functools.cache
    def r0(n):
        if 1 > n:
            raise ValueError('n for r0 must be geq 1')
        return np.cos(np.pi / (2 * n))
    @functools.cache
    def z(self, n):
        if 1 > n:
            raise ValueError('n for z must be geq 1')
        return (1 + self.r0(n)) / self.alpha
    @functools.cache
    def t(self, n, x=None):
        if x is None:
            x = self.z(n) - self.r0(n)
        match n:
            case 0:
                return 1
            case 1:
                return x
            case _:
                return (2 * x * self.t(n-1, x=x)) - self.t(n-2, x=x)
    @functools.cache
    def g(self, n):
        if 0 == n:
            return 1
        return self.t(n) / (self.z(n) ** n)
    @functools.cache
    def q(self, n):
        if 1 > n:
            raise ValueError('n for q must be geq 1')
        return (1 / self.z(n)) * (-self.r0(n))
    @functools.cache
    def a(self, n):
        if 1 > n:
            raise ValueError('n for a must be geq 1')
        upper_lim = 32
        if n > upper_lim:
            warnings.warn(
                'ChebyshevMagicNumbers: Indexing magic number a beyond its'
                ' numerically stable domain; an approximation will be given'
                ' from here on'
            )
            return self.a(upper_lim)
        return 2 * (self.g(n-1) / self.g(n))
    @functools.cache
    def b(self, n):
        if 2 > n:
            raise ValueError('n for b must be geq 2')
        return (n * self.q(n)) - ((n-1) * self.q(n-1))
    @functools.cache
    def bp(self, n):
        return self.a(n) * self.b(n)
    @functools.cache
    def c(self, n):
        if 2 > n:
            raise ValueError('n for c must be geq 2')
        term0 = 2 * n * (n-1) * ((self.q(n) - self.q(n-1)) ** 2)
        term1 = n * ((1 / self.z(n)) ** 2)
        if 2 < n:
            term1 -= (n-1) * ((1 / self.z(n-1)) ** 2)
        c = 0.25 * self.a(n-1) * (term0 - term1)
        if 2 == n:
            c /= 2
        return c
    @functools.cache
    def cp(self, n):
        return self.a(n) * self.c(n)
