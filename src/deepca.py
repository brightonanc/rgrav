import torch
from types import SimpleNamespace

from .algorithm_base import GrassmannianAveragingAlgorithm, \
        DecentralizedConsensusAlgorithm
from . import util


class EPCA(GrassmannianAveragingAlgorithm):
    """ A centralized variant of DeEPCA """

    def _sign_adjust(self, W, W0):
        tmp = (W.mT[..., None, :] @ W0.mT[..., :, None]).squeeze((-1,-2))
        W *= tmp.sign()[..., None, :]
        return W

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        W0 = util.get_standard_basis_like(U_arr[0])
        W = W0
        S = W0.clone()
        AW = W0.clone()
        iter_frame.U = W
        iter_frame.err_criterion = torch.nan
        yield iter_frame
        while True:
            iter_frame = SimpleNamespace()
            tmp = U_arr @ (U_arr.mT @ W)
            tmp = tmp.mean(0)
            S = S + tmp - AW
            AW = tmp
            prev_W = W
            W = torch.linalg.qr(S).Q
            W = self._sign_adjust(W, W0)
            iter_frame.U = W
            iter_frame.err_criterion = (W - prev_W).norm().item()
            yield iter_frame


class DeEPCA(GrassmannianAveragingAlgorithm):

    def __init__(self, comm_W, cons_rounds=8):
        self.comm_W = comm_W
        self.cons_rounds = cons_rounds
        # Precompute eta
        lamb2 = torch.linalg.eigvalsh(self.comm_W)[-2].abs()
        tmp = (1 - (lamb2**2))**0.5
        self.eta = (1 - tmp) / (1 + tmp)

    def _fast_mix(self, W):
        M, N, K = W.shape
        W_ = W.view(M, N*K)
        prev_W_ = W_.clone()
        for _ in range(self.cons_rounds):
            next_W_ = ((1+self.eta) * self.comm_W @ W_) - (self.eta * prev_W_)
            prev_W_ = W_
            W_ = next_W_
        W = W_.view(M, N, K)
        return W

    def _sign_adjust(self, W, W0):
        tmp = (W.mT[..., None, :] @ W0.mT[..., :, None]).squeeze((-1,-2))
        W *= tmp.sign()[..., None, :]
        return W

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        W0 = util.get_standard_basis_like(U_arr)
        W = W0
        S = W0.clone()
        AW = W0.clone()
        iter_frame.U = W
        yield iter_frame
        while True:
            iter_frame = SimpleNamespace()
            tmp = U_arr @ (U_arr.mT @ W)
            S += tmp - AW
            AW = tmp
            S = self._fast_mix(S)
            W = torch.linalg.qr(S).Q
            W = self._sign_adjust(W, W0)
            iter_frame.U = W
            yield iter_frame

