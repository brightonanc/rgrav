import torch
from types import SimpleNamespace
import numpy as np
from dataclasses import dataclass
import functools
import warnings

from .algorithm_base import GrassmannianAveragingAlgorithm, \
        DecentralizedConsensusAlgorithm
from . import util


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
                'ChebyshevMagicNumber: Due to numerical instabilities in the'
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
                'ChebyshevMagicNumber: Indexing magic number a beyond its'
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



def get_rgrav_roots(N, lo):
    def get_chebyshev_roots(N):
        n_arr = torch.arange(N)
        roots = torch.cos((torch.pi/N)*(0.5+n_arr))
        return roots
    rgrav_roots = lo * 0.5 * (1 + get_chebyshev_roots(N))
    return rgrav_roots


class RGrAv(GrassmannianAveragingAlgorithm):

    def __init__(self, num_iter, lo, mode='qr-stable', ortho_scheduler=None):
        self.num_iter = num_iter
        self.root_arr = get_rgrav_roots(num_iter, lo)
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = util.get_standard_basis_like(U_arr[0])
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            root_it = self.root_arr[it-1] if (it-1) < self.num_iter else 0.
            tmp0 = (1 / (1 - root_it)) * (U_arr @ (U_arr.mT @ U))
            tmp1 = (root_it / (1 - root_it)) * U
            Z = tmp0 - tmp1
            Z_hat = Z.mean(0)
            do_ortho = self.ortho_scheduler(it)
            if do_ortho:
                U = util.get_orthobasis(Z_hat, mode=self.mode)
            else:
                U = Z_hat
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            if it > self.num_iter:
                iter_frame.over_iteration = True
            yield iter_frame

class RGrAv2(GrassmannianAveragingAlgorithm):

    def __init__(self, lo, mode='qr-stable', ortho_scheduler=None):
        self.lo = lo
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = util.get_standard_basis_like(U_arr[0])
        gamma = (2 / self.lo) - 1
        a = [1, gamma]
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            if 1 == it:
                fac0 = a[-2] / a[-1]
                tmp0 = (fac0 * (2 / self.lo)) * (U_arr @ (U_arr.mT @ U))
                tmp1 = fac0 * U
                Z = tmp0 - tmp1
            else:
                a.append((2 * gamma * a[-1]) - a[-2])
                a = a[-3:]
                fac0 = 2 * (a[-2] / a[-1])
                tmp0 = (fac0 * (2 / self.lo)) * (U_arr @ (U_arr.mT @ U))
                tmp1 = fac0 * U
                fac1 = a[-3] / a[-1]
                tmp2 = fac1 * prev_U
                Z = (tmp0 - tmp1) - tmp2
            Z_hat = Z.mean(0)
            do_ortho = self.ortho_scheduler(it)
            prev_U = U
            if do_ortho:
                U = util.get_orthobasis(Z_hat, mode=self.mode)
            else:
                U = Z_hat
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame


class FiniteRGrAv(GrassmannianAveragingAlgorithm):

    def __init__(self, alpha, num_iter, zero_first=False, mode='qr-stable',
             ortho_scheduler=None):
        self.num_iter = num_iter
        self.root_arr = ChebyshevMagicNumbers(alpha).get_root_arr(num_iter)
        if zero_first:
            self.root_arr = self.root_arr.flip(0)
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = util.get_standard_basis_like(U_arr[0])
        iter_frame.U = U
        iter_frame.within_num_iter = True
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            within_num_iter = (it-1) < self.num_iter
            if not within_num_iter:
                warnings.warn(
                    'FiniteRGrAv: algorithm is now running longer than'
                    ' originally specified by num_iter'
                )
            root = self.root_arr[it-1] if within_num_iter else 0.
            fac0 = 1 / (1 - root)
            fac1 = root / (1 - root)
            term0 = fac0 * (U_arr @ (U_arr.mT @ U))
            term1 = fac1 * U
            Z = term0 - term1
            Z_hat = Z.mean(0)
            do_ortho = self.ortho_scheduler(it)
            if do_ortho:
                U = util.get_orthobasis(Z_hat, mode=self.mode)
            else:
                U = Z_hat
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            iter_frame.within_num_iter = within_num_iter
            yield iter_frame


class AsymptoticRGrAv(GrassmannianAveragingAlgorithm):

    def __init__(self, alpha, mode='qr-stable', ortho_scheduler=None):
        self.cmn = ChebyshevMagicNumbers(alpha)
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = util.get_standard_basis_like(U_arr[0])
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            if 1 == it:
                Z = U_arr @ (U_arr.mT @ U)
            else:
                a = self.cmn.a(it)
                b = self.cmn.b(it)
                c = self.cmn.c(it)
                Z = a * ((U_arr @ (U_arr.mT @ U)) + (b * U) + (c * prev_U))
            Z_hat = Z.mean(0)
            do_ortho = self.ortho_scheduler(it)
            prev_U = U
            if do_ortho:
                U = util.get_orthobasis(Z_hat, mode=self.mode)
            else:
                U = Z_hat
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame


class DRGrAv(GrassmannianAveragingAlgorithm):

    def __init__(self, num_iter, lo, comm_W, cons_rounds=8, mode='qr-stable',
            ortho_scheduler=None):
        self.num_iter = num_iter
        self.root_arr = get_rgrav_roots(num_iter, lo)
        self.comm_W = comm_W
        self.cons_rounds = cons_rounds
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler
        # Precompute eta
        lamb2 = torch.linalg.eigvalsh(self.comm_W)[-2].abs()
        tmp = (1 - (lamb2**2))**0.5
        self.eta = (1 - tmp) / (1 + tmp)

    def _fast_mix(self, S):
        M, N, K = S.shape
        S_ = S.view(M, N*K)
        prev_S_ = S_.clone()
        for _ in range(self.cons_rounds):
            next_S_ = ((1+self.eta) * self.comm_W @ S_) - (self.eta * prev_S_)
            prev_S_ = S_
            S_ = next_S_
        S = S_.view(M, N, K)
        return S
        #M, N, K = S.shape
        #S_ = S.view(M, N*K)
        #root_arr = [-0.5, 0.5, 0.]
        #for i in range(self.cons_rounds):
        #    root = root_arr[i]
        #    fac = 1. / (1 - root)
        #    cons = self.comm_W @ (fac * S_)
        #    S_ = cons - (fac * root * S_)
        #S = S_.view(M, N, K)
        #return S


    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = util.get_standard_basis_like(U_arr)
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            root_it = self.root_arr[it-1] if (it-1) < self.num_iter else 0.
            tmp0 = (1 / (1 - root_it)) * (U_arr @ (U_arr.mT @ U))
            tmp1 = (root_it / (1 - root_it)) * U
            if 1 == it:
                Z = tmp0 - tmp1
            else:
                root_itm1 = self.root_arr[it-2] if (it-2) < self.num_iter else 0.
                tmp2 = (1 / (1 - root_itm1)) * (U_arr @ (U_arr.mT @ prev_U))
                tmp3 = (root_itm1 / (1 - root_itm1)) * prev_U
                Z = Z_mixed + (tmp0 - tmp1) - (tmp2 - tmp3)
            Z_mixed = self._fast_mix(Z)
            do_ortho = self.ortho_scheduler(it)
            prev_U = U
            if do_ortho:
                U = util.get_orthobasis(Z_mixed, mode=self.mode)
            else:
                U = Z_mixed
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame


class DRGrAv2(GrassmannianAveragingAlgorithm):

    def __init__(self, num_iter, lo, comm_W, cons_rounds=8, mode='qr-stable',
            ortho_scheduler=None):
        self.num_iter = num_iter
        self.root_arr = get_rgrav_roots(num_iter, lo)
        self.comm_W = comm_W
        self.cons_rounds = cons_rounds
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler
        # Precompute eta
        lamb2 = torch.linalg.eigvalsh(self.comm_W)[-2].abs()
        tmp = (1 - (lamb2**2))**0.5
        self.eta = (1 - tmp) / (1 + tmp)

    def _fast_mix(self, S):
        M, N, K = S.shape
        S_ = S.view(M, N*K)
        prev_S_ = S_.clone()
        for _ in range(self.cons_rounds):
            next_S_ = ((1+self.eta) * self.comm_W @ S_) - (self.eta * prev_S_)
            prev_S_ = S_
            S_ = next_S_
        S = S_.view(M, N, K)
        return S
        #M, N, K = S.shape
        #S_ = S.view(M, N*K)
        #root_arr = [-0.5, 0.5, 0.]
        #for i in range(self.cons_rounds):
        #    root = root_arr[i]
        #    fac = 1. / (1 - root)
        #    cons = self.comm_W @ (fac * S_)
        #    S_ = cons - (fac * root * S_)
        #S = S_.view(M, N, K)
        #return S


    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = util.get_standard_basis_like(U_arr)
        prev_upd = torch.zeros_like(U)
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            root_it = self.root_arr[it-1] if (it-1) < self.num_iter else 0.
            tmp0 = (1 / (1 - root_it)) * (U_arr @ (U_arr.mT @ U))
            tmp1 = (root_it / (1 - root_it)) * U
            if 1 == it:
                Z = tmp0 - tmp1
            else:
                root_itm1 = self.root_arr[it-2] if (it-2) < self.num_iter else 0.
                tmp2 = (1 / (1 - root_itm1)) * (U_arr @ (U_arr.mT @ prev_U))
                tmp3 = (root_itm1 / (1 - root_itm1)) * prev_U
                Z = Z_mixed + (tmp0 - tmp1) - (tmp2 - tmp3)
            Z_mixed = self._fast_mix(Z)
            do_ortho = self.ortho_scheduler(it)
            prev_U = U
            if do_ortho:
                U = util.get_orthobasis(Z_mixed, mode=self.mode)
            else:
                U = Z_mixed
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame

