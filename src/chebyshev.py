from abc import ABC, abstractmethod
import torch
import numpy as np
from dataclasses import dataclass
import functools
import warnings


@dataclass
class AbstractChebyshevMagicNumbers(ABC):
    """
    Magic numbers used in order to iteratively construct a desirable Chebyshev
    polynomial
    """
    alpha : float
    def __hash__(self):
        return hash(self.alpha)
    def __eq__(self, other):
        return self.alpha == other.alpha
    @abstractmethod
    def visualize_polynomial(self, n, *, exact=False, samples=None,
            focus_roots=False, yy=None):
        pass


@dataclass
class ChebyshevMagicNumbers(AbstractChebyshevMagicNumbers):
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
        $f_n(y) = \frac{T_n(z_n y - r_n)}{T_n(z_n - r_n)} ~ \forall n \geq 0$

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
    __hash__ = super.__hash__
    def __post_init__(self):
        if not isinstance(self.alpha, float):
            raise ValueError(
                f'{self.alpha=} is not a float. alpha must be a float to'
                ' ensure 64-bit type for numerical stability'
            )
        lower_lim = 1e-6
        if lower_lim > self.alpha:
            warnings.warn(
                'ChebyshevMagicNumbers: Due to numerical instabilities in the'
                ' computation of magic number a, alpha values less than'
                f' {lower_lim} are rounded up'
            )
            self.alpha = lower_lim
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
    def visualize_polynomial(self, n, *, which_poly='exact', samples=None,
            focus_roots=False, yy=None):
        options_which_poly = ('exact', 'iterated', 'both')
        if which_poly not in options_which_poly:
            raise ValueError(
                f'which_poly must be one of the following:'
                ' {options_which_poly}'
            )
        plot_exact = which_poly in ('exact', 'both')
        plot_iter = which_poly in ('iterated', 'both')
        import matplotlib.pyplot as plt
        if yy is None:
            yy = torch.linspace(0, 1, 8192)
            if focus_roots:
                yy *= self.alpha
        yy = torch.cat((torch.tensor([self.alpha]), yy))
        if samples is not None:
            yy = torch.cat((samples, yy))
        poly = torch.ones_like(yy)
        if plot_exact:
            poly_exact = poly.clone()
            for root in self.get_root_arr(n):
                poly_exact *= (yy - root) / (1 - root)
            poly_exact = poly_exact.abs()
        if plot_iter:
            poly_iter = poly
            for i in range(1, n+1):
                if 1 == i:
                    next_poly_iter = yy
                else:
                    a = self.a(i)
                    b = self.b(i)
                    c = self.c(i)
                    next_poly_iter = a * (
                        (yy * poly_iter) + (b * poly_iter)
                        + (c * prev_poly_iter)
                    )
                prev_poly_iter = poly_iter
                poly_iter = next_poly_iter
            poly_iter = poly_iter.abs()
        if samples is not None:
            if plot_exact:
                f_exact_samples = poly_exact[:samples.numel()]
                poly_exact = poly_exact[samples.numel():]
            if plot_iter:
                f_iter_samples = poly_iter[:samples.numel()]
                poly_iter = poly_iter[samples.numel():]
            yy = yy[samples.numel():]
        max_poly = 0.
        if plot_exact:
            f_exact_alpha = poly_exact[0].item()
            poly_exact = poly_exact[1:]
            max_poly = max(max_poly, poly_exact.max().item())
        if plot_iter:
            f_iter_alpha = poly_iter[0].item()
            poly_iter = poly_iter[1:]
            max_poly = max(max_poly, poly_iter.max().item())
        yy = yy[1:]
        if samples is not None:
            plt.subplot(1, 2, 1)
        if plot_exact:
            plt.plot(yy, poly_exact, 'b', label='exact')
            plt.plot(
                [0, self.alpha, self.alpha],
                [f_exact_alpha, f_exact_alpha, 0],
                'b--'
            )
            if samples is not None:
                plt.plot(samples, f_exact_samples, 'bx')
        if plot_iter:
            plt.plot(yy, poly_iter, 'r', label='iterated')
            plt.plot(
                [0, self.alpha, self.alpha],
                [f_iter_alpha, f_iter_alpha, 0],
                'r--'
            )
            if samples is not None:
                plt.plot(samples, f_iter_samples, 'rx')
        plt.title(fr'$|f_n(y)|$ for $\alpha={self.alpha}$')
        plt.xlabel(r'$y$')
        plt.xlim([yy[0], yy[-1]])
        plt.ylim([0, max_poly])
        plt.legend()
        if samples is not None:
            plt.subplot(1, 2, 2)
            max_new_samples = 0.
            if plot_exact:
                new_samples_exact = torch.cat(
                    (torch.tensor([0.]), f_exact_samples.sort().values)
                )
                plt.plot(
                    new_samples_exact,
                    torch.linspace(0, 1, new_samples_exact.numel()),
                    'b-x',
                    label='exact'
                )
                max_new_samples = max(
                    max_new_samples,
                    new_samples_exact[-1].item()
                )
            if plot_iter:
                new_samples_iter = torch.cat(
                    (torch.tensor([0.]), f_iter_samples.sort().values)
                )
                plt.plot(
                    new_samples_iter,
                    torch.linspace(0, 1, new_samples_iter.numel()),
                    'r-x',
                    label='iterated'
                )
                max_new_samples = max(
                    max_new_samples,
                    new_samples_iter[-1].item()
                )
            plt.title(r'CDF of $|f_n(\text{samples})|$')
            plt.xlabel(r'$|f_n(\text{samples})|$')
            _g = 20
            x_upper = int((max_new_samples * _g) + 1) / _g
            plt.xlim([0, x_upper])
            plt.ylim([0, 1])
            plt.legend()
        plt.show()


@dataclass
class BalancedChebyshevMagicNumbers(AbstractChebyshevMagicNumbers):
    r"""
    Magic numbers used in order to iteratively construct a desirable Chebyshev
    polynomial

    Let $T_n(x)$ be the $n$th-order Chebyshev polynomial of the first kind,
    defined recursively as follows:
        $T_0(x) = 1$
        $T_1(x) = x$
        $T_n(x) = 2 x T_{n-1}(x) - T_{n-2}(x) ~ \forall n \geq 2$

    Let $\alpha \in (0, 1)$ be some chosen threshold and $z :=
    \frac{1}{\alpha}$. Let $f_n(y)$ be defined then as follows:
        $f_n(y) = \frac{T_n(z_n y)}{T_n(z_n)} ~ \forall n \geq 0$

    The functions $f_n$ may be constructed iteratively using the magic numbers
    (a_n, c_n) as follows:
        $f_0(y) = 1$
        $f_1(y) = y$
        $f_n(y) = a_n (y f_{n-1}(y) + c_n f_{n-2}(y)) ~ \forall n \geq 2$
    """
    __hash__ = super.__hash__
    def get_root_arr(self, n):
        n_arr = torch.arange(n)
        root_arr = torch.cos((torch.pi / n) * (n_arr + 0.5))
        root_arr *= self.alpha
        return root_arr
    @functools.cache
    def h(self, n):
        if 1 > n:
            raise ValueError('n for h must be geq 1')
        if 1 == n:
            return self.alpha
        return 1 / ((2 / self.alpha) - self.h(n-1))
    @functools.cache
    def a(self, n):
        if 2 > n:
            raise ValueError('n for a must be geq 2')
        return (2 / self.alpha) * self.h(n)
    @functools.cache
    def c(self, n):
        if 2 > n:
            raise ValueError('n for c must be geq 2')
        return -(self.alpha / 2) * self.h(n-1)
    @functools.cache
    def cp(self, n):
        return self.a(n) * self.c(n)
    def visualize_polynomial(self, n, *, which_poly='exact', samples=None,
            focus_roots=False, yy=None):
        options_which_poly = ('exact', 'iterated', 'both')
        if which_poly not in options_which_poly:
            raise ValueError(
                f'which_poly must be one of the following:'
                ' {options_which_poly}'
            )
        plot_exact = which_poly in ('exact', 'both')
        plot_iter = which_poly in ('iterated', 'both')
        import matplotlib.pyplot as plt
        if yy is None:
            yy = torch.linspace(-1, 1, 8192)
            if focus_roots:
                yy *= self.alpha
        yy = torch.cat((torch.tensor([self.alpha]), yy))
        if samples is not None:
            yy = torch.cat((samples, yy))
        poly = torch.ones_like(yy)
        if plot_exact:
            poly_exact = poly.clone()
            for root in self.get_root_arr(n):
                poly_exact *= (yy - root) / (1 - root)
            poly_exact = poly_exact.abs()
        if plot_iter:
            poly_iter = poly
            for i in range(1, n+1):
                if 1 == i:
                    next_poly_iter = yy
                else:
                    a = self.a(i)
                    c = self.c(i)
                    print(f'{a=}')
                    print(f'{c=}')
                    next_poly_iter = a * (
                        (yy * poly_iter) + (c * prev_poly_iter)
                    )
                prev_poly_iter = poly_iter
                poly_iter = next_poly_iter
            poly_iter = poly_iter.abs()
        if samples is not None:
            if plot_exact:
                f_exact_samples = poly_exact[:samples.numel()]
                poly_exact = poly_exact[samples.numel():]
            if plot_iter:
                f_iter_samples = poly_iter[:samples.numel()]
                poly_iter = poly_iter[samples.numel():]
            yy = yy[samples.numel():]
        max_poly = 0.
        if plot_exact:
            f_exact_alpha = poly_exact[0].item()
            poly_exact = poly_exact[1:]
            max_poly = max(max_poly, poly_exact.max().item())
        if plot_iter:
            f_iter_alpha = poly_iter[0].item()
            poly_iter = poly_iter[1:]
            max_poly = max(max_poly, poly_iter.max().item())
        yy = yy[1:]
        if samples is not None:
            plt.subplot(1, 2, 1)
        if plot_exact:
            plt.plot(yy, poly_exact, 'b', label='exact')
            plt.plot(
                [-self.alpha, self.alpha, self.alpha],
                [f_exact_alpha, f_exact_alpha, 0],
                'b--'
            )
            if samples is not None:
                plt.plot(samples, f_exact_samples, 'bx')
        if plot_iter:
            plt.plot(yy, poly_iter, 'r', label='iterated')
            plt.plot(
                [-self.alpha, self.alpha, self.alpha],
                [f_iter_alpha, f_iter_alpha, 0],
                'r--'
            )
            if samples is not None:
                plt.plot(samples, f_iter_samples, 'rx')
        plt.title(fr'$|f_n(y)|$ for $\alpha={self.alpha}$')
        plt.xlabel(r'$y$')
        plt.xlim([yy[0], yy[-1]])
        plt.ylim([0, max_poly])
        plt.legend()
        if samples is not None:
            plt.subplot(1, 2, 2)
            max_new_samples = 0.
            if plot_exact:
                new_samples_exact = torch.cat(
                    (torch.tensor([0.]), f_exact_samples.sort().values)
                )
                plt.plot(
                    new_samples_exact,
                    torch.linspace(0, 1, new_samples_exact.numel()),
                    'b-x',
                    label='exact'
                )
                max_new_samples = max(
                    max_new_samples,
                    new_samples_exact[-1].item()
                )
            if plot_iter:
                new_samples_iter = torch.cat(
                    (torch.tensor([0.]), f_iter_samples.sort().values)
                )
                plt.plot(
                    new_samples_iter,
                    torch.linspace(0, 1, new_samples_iter.numel()),
                    'r-x',
                    label='iterated'
                )
                max_new_samples = max(
                    max_new_samples,
                    new_samples_iter[-1].item()
                )
            plt.title(r'CDF of $|f_n(\text{samples})|$')
            plt.xlabel(r'$|f_n(\text{samples})|$')
            _g = 20
            x_upper = int((max_new_samples * _g) + 1) / _g
            plt.xlim([0, x_upper])
            plt.ylim([0, 1])
            plt.legend()
        plt.show()
