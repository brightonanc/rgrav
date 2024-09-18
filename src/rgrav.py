import torch
from types import SimpleNamespace

from .algorithm_base import GrassmannianAveragingAlgorithm, \
        DecentralizedConsensusAlgorithm
from . import util


class RGrAv(GrassmannianAveragingAlgorithm):

    def __init__(self, mode='qr-stable', ortho_scheduler=None):
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
            Z = U_arr @ (U_arr.mT @ U)
            Z_hat = Z.mean(0)
            do_ortho = self.ortho_scheduler(it)
            if do_ortho:
                U = util.get_orthobasis(Z_hat, mode=self.mode)
            else:
                U = Z_hat
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame


class DRGrAv(GrassmannianAveragingAlgorithm):

    def __init__(self, comm_W, cons_rounds=8, mode='qr-stable',
            ortho_scheduler=None):
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

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = util.get_standard_basis_like(U_arr)
        S = U.clone()
        Z = U.clone()
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            tmp = U_arr @ (U_arr.mT @ U)
            S += tmp - Z
            Z = tmp
            S = self._fast_mix(S)
            do_ortho = self.ortho_scheduler(it)
            if do_ortho:
                U = util.get_orthobasis(S, mode=self.mode)
            else:
                U = S
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame
