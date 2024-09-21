import torch
from types import SimpleNamespace

from .algorithm_base import GrassmannianAveragingAlgorithm, \
        DecentralizedConsensusAlgorithm
from . import util


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

