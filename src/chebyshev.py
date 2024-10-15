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
    alpha : torch.Tensor
    def __post_init__(self):
        if not isinstance(self.alpha, torch.Tensor):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float64)
        if 0 < len(self.alpha.shape):
            raise ValueError(f'{self.alpha=} is a non-scalar tensor')
        if not torch.is_floating_point(self.alpha):
            raise ValueError(f'{self.alpha.dtype=} is not a float type')
    def __hash__(self):
        return hash(self.alpha.item())
    def __eq__(self, other):
        return self.alpha.item() == other.alpha.item()
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
        super().__post_init__()
        if 0.99 < self.alpha:
            warnings.warn(
                'ChebyshevMagicNumbers: For numerical purposes, it is not'
                ' recommended to set alpha > 0.99 (which is'
                f' {self.alpha.item()} < 0.99)'
            )
        if self.alpha < torch.finfo(self.alpha.dtype).resolution:
            _RES = torch.finfo(self.alpha.dtype).resolution
            warnings.warn(
                'ChebyshevMagicNumbers: For numerical purposes, it is not'
                ' recommended to set alpha < FP_RESOLUTION'
                f' (which is {self.alpha.item()} < {_RES})'
            )
    def get_root_arr(self, n):
        n_arr = torch.arange(n, dtype=self.alpha.dtype)
        root_arr = torch.cos((torch.pi / n) * (n_arr + 0.5))
        root_arr = (root_arr + root_arr[0]) / (1 + root_arr[0])
        root_arr *= self.alpha
        return root_arr
    @staticmethod
    @functools.cache
    def r0(n, *, dtype):
        if 1 > n:
            raise ValueError('n for r0 must be geq 1')
        return torch.cos(torch.tensor(torch.pi / (2 * n), dtype=dtype))
        # grows from 0 to 1
    @classmethod
    @functools.cache
    def z_(cls, n, *, dtype):
        if 1 > n:
            raise ValueError('n for z_ must be geq 1')
        return 1 + cls.r0(n, dtype=dtype)
        # grows from 1 to 2
    @classmethod
    @functools.cache
    def w0(cls, n, *, dtype):
        if 1 > n:
            raise ValueError('n for w0 must be geq 1')
        if 1 == n:
            return torch.tensor(1., dtype=dtype)
        z__n = cls.z_(n, dtype=dtype)
        z__nm1 = cls.z_(n-1, dtype=dtype)
        return ((z__n / z__nm1)**(n-1)) * z__n
        # w0(1)=1, then goes from (3/2)+(2**0.5) to 2
    @classmethod
    @functools.cache
    def q_(cls, n, *, dtype):
        if 1 > n:
            raise ValueError('n for q_ must be geq 1')
        return -cls.r0(n, dtype=dtype) / cls.z_(n, dtype=dtype)
        # decreases from 0 to -0.5
    @classmethod
    @functools.cache
    def diff_q_(cls, n, *, dtype):
        if 2 > n:
            raise ValueError('n for diff_q_ must be geq 2')
        diff = cls.q_(n, dtype=dtype) - cls.q_(n-1, dtype=dtype)
        if torch.abs(diff) < torch.finfo(dtype).resolution:
            # Needed for stability as diff decreases O(n^{-3}) and is only ever
            # multiplied by O(n)
            diff.fill_(0)
        return diff
        # goes from 1-2**0.5 to 0
    @classmethod
    @functools.cache
    def b_pre(cls, n, *, dtype):
        if 2 > n:
            raise ValueError('n for b_pre must be geq 2')
        return (n * cls.diff_q_(n, dtype=dtype)) + cls.q_(n-1, dtype=dtype)
        # goes from 2(1-2**0.5) to -0.5
    @classmethod
    @functools.cache
    def diff_inv_sq_z_(cls, n, *, dtype):
        if 2 > n:
            raise ValueError('n for diff_inv_sq_z_ must be geq 2')
        term0 = ((1 / cls.z_(n, dtype=dtype)) ** 2)
        term1 = ((1 / cls.z_(n-1, dtype=dtype)) ** 2)
        diff = term0 - term1
        if torch.abs(diff) < torch.finfo(dtype).resolution:
            # Needed for stability as diff decreases O(n^{-3}) and is only ever
            # multiplied by O(n)
            diff.fill_(0)
        return diff
        # goes from 5-(4*(2**0.5)) to 0
    @classmethod
    @functools.cache
    def c_pre(cls, n, *, dtype):
        if 2 > n:
            raise ValueError('n for c_pre must be geq 2')
        if 2 == n:
            return torch.tensor(0, dtype=dtype)
        term0 = 2 * n * (n-1) * (cls.diff_q_(n, dtype=dtype) ** 2)
        term1 = -n * cls.diff_inv_sq_z_(n, dtype=dtype)
        term2 = -((1 / cls.z_(n-1, dtype=dtype)) ** 2)
        return 0.25 * (term0 + term1 + term2)
        # goes from 0 to -0.0625
    @functools.cache
    def z(self, n):
        return self.z_(n, dtype=self.alpha.dtype) / self.alpha
        # grows from 1/alpha to 2/alpha
    @functools.cache
    def t(self, n, x=None):
        if (x is None) and (0 < n):
            if not torch.isfinite(self.t(n-1)):
                return self.t(n-1)
            x = self.z(n) - self.r0(n, dtype=self.alpha.dtype)
        match n:
            case 0:
                return torch.tensor(1, dtype=self.alpha.dtype)
            case 1:
                return x
            case _:
                return (2 * x * self.t(n-1, x=x)) - self.t(n-2, x=x)
    @functools.cache
    def w1_infty(self):
        term0 = 2 / self.alpha
        return 2 / (((term0**0.5) + ((term0-2)**0.5))**2)
    @functools.cache
    def w1(self, n):
        if 1 > n:
            raise ValueError('n for w1 must be geq 1')
        t_n = self.t(n)
        t_nm1 = self.t(n-1)
        if not torch.isfinite(t_n):
            return self.w1_infty()
        return self.t(n-1) / self.t(n)
    @functools.cache
    def a(self, n):
        if 1 > n:
            raise ValueError('n for a must be geq 1')
        #return 2 * (self.g(n-1) / self.g(n))
        w0 = self.w0(n, dtype=self.alpha.dtype)
        w1 = self.w1(n)
        return 2 * w0 * (1 / self.alpha) * w1
    @functools.cache
    def b(self, n):
        if 2 > n:
            raise ValueError('n for b must be geq 2')
        return self.alpha * self.b_pre(n, dtype=self.alpha.dtype)
    @functools.cache
    def bp(self, n):
        return self.a(n) * self.b(n)
    @functools.cache
    def c(self, n):
        if 2 > n:
            raise ValueError('n for c must be geq 2')
        c_pre = self.c_pre(n, dtype=self.alpha.dtype)
        c = self.a(n-1) * (self.alpha ** 2) * c_pre
        return c
    @functools.cache
    def cp(self, n):
        return self.a(n) * self.c(n)
    def visualize_polynomial(self, n, *, which_poly='exact', samples=None,
            focus_roots=False, yy=None):
        dtype = self.alpha.dtype
        alpha = self.alpha.to(dtype)
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
            yy = torch.linspace(0, 1, 8192, dtype=dtype)
            if focus_roots:
                yy *= alpha
        yy = torch.cat((alpha[None], yy))
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
            plt.plot(yy, poly_exact, 'b', alpha=0.5, label='exact')
            plt.plot(
                [0, alpha, alpha],
                [f_exact_alpha, f_exact_alpha, 0],
                'b--',
                alpha=0.5,
            )
            if samples is not None:
                plt.plot(samples, f_exact_samples, 'bx', alpha=0.5)
        if plot_iter:
            plt.plot(yy, poly_iter, 'r', alpha=0.5, label='iterated')
            plt.plot(
                [0, alpha, alpha],
                [f_iter_alpha, f_iter_alpha, 0],
                'r--',
                alpha=0.5,
            )
            if samples is not None:
                plt.plot(samples, f_iter_samples, 'rx', alpha=0.5)
        plt.title(
            fr'$|f_n(y)|$ for $\alpha={self.alpha.item()}$'
            f' (dtype={self.alpha.dtype})'
        )
        plt.xlabel(r'$y$')
        plt.xlim([yy[0], yy[-1]])
        plt.ylim([0, max_poly])
        plt.legend()
        if samples is not None:
            plt.subplot(1, 2, 2)
            max_new_samples = 0.
            if plot_exact:
                new_samples_exact = torch.cat((
                    torch.tensor([0.], dtype=dtype),
                    f_exact_samples.sort().values
                ))
                plt.plot(
                    new_samples_exact,
                    torch.linspace(
                        0, 1, new_samples_exact.numel(), dtype=dtype
                    ),
                    'b-x',
                    alpha=0.5,
                    label='exact'
                )
                max_new_samples = max(
                    max_new_samples,
                    new_samples_exact[-1].item()
                )
            if plot_iter:
                new_samples_iter = torch.cat((
                    torch.tensor([0.], dtype=dtype),
                    f_iter_samples.sort().values
                ))
                plt.plot(
                    new_samples_iter,
                    torch.linspace(
                        0, 1, new_samples_iter.numel(), dtype=dtype
                    ),
                    'r-x',
                    alpha=0.5,
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
        n_arr = torch.arange(n, dtype=self.alpha.dtype)
        root_arr = torch.cos((torch.pi / n) * (n_arr + 0.5))
        root_arr *= self.alpha
        return root_arr
    @functools.cache
    def h(self, n):
        if 1 > n:
            raise ValueError('n for h must be geq 1')
        if 1 == n:
            return self.alpha.clone()
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
        dtype = self.alpha.dtype
        alpha = self.alpha.to(dtype)
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
            yy = torch.linspace(-1, 1, 8192, dtype=dtype)
            if focus_roots:
                yy *= alpha
        yy = torch.cat((alpha[None], yy))
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
            plt.plot(yy, poly_exact, 'b', alpha=0.5, label='exact')
            plt.plot(
                [-alpha, alpha, alpha],
                [f_exact_alpha, f_exact_alpha, 0],
                'b--',
                alpha=0.5,
            )
            if samples is not None:
                plt.plot(samples, f_exact_samples, 'bx', alpha=0.5)
        if plot_iter:
            plt.plot(yy, poly_iter, 'r', alpha=0.5, label='iterated')
            plt.plot(
                [-alpha, alpha, alpha],
                [f_iter_alpha, f_iter_alpha, 0],
                'r--',
                alpha=0.5,
            )
            if samples is not None:
                plt.plot(samples, f_iter_samples, 'rx', alpha=0.5)
        plt.title(
            fr'$|f_n(y)|$ for $\alpha={self.alpha.item()}$'
            f' (dtype={self.alpha.dtype})'
        )
        plt.xlabel(r'$y$')
        plt.xlim([yy[0], yy[-1]])
        plt.ylim([0, max_poly])
        plt.legend()
        if samples is not None:
            plt.subplot(1, 2, 2)
            max_new_samples = 0.
            if plot_exact:
                new_samples_exact = torch.cat((
                    torch.tensor([0.], dtype=dtype),
                    f_exact_samples.sort().values
                ))
                plt.plot(
                    new_samples_exact,
                    torch.linspace(
                        0, 1, new_samples_exact.numel(), dtype=dtype
                    ),
                    'b-x',
                    alpha=0.5,
                    label='exact'
                )
                max_new_samples = max(
                    max_new_samples,
                    new_samples_exact[-1].item()
                )
            if plot_iter:
                new_samples_iter = torch.cat((
                    torch.tensor([0.], dtype=dtype),
                    f_iter_samples.sort().values
                ))
                plt.plot(
                    new_samples_iter,
                    torch.linspace(
                        0, 1, new_samples_iter.numel(), dtype=dtype
                    ),
                    'r-x',
                    alpha=0.5,
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

