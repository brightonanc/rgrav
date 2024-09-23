import torch
from types import SimpleNamespace

from .algorithm_base import GrassmannianAveragingAlgorithm, \
        DecentralizedConsensusAlgorithm
from .consensus import FastMixDeEPCA
from . import util


class DeEPCA(DecentralizedConsensusAlgorithm):
    """ See https://arxiv.org/abs/2102.03990 """

    @classmethod
    def CanonicalForm(cls, comm_W, cons_rounds=8):
        return cls(FastMixDeEPCA(comm_W, cons_rounds))

    def _sign_adjust(self, W, W0):
        tmp = (W.mT[..., None, :] @ W0.mT[..., :, None]).squeeze((-1,-2))
        W *= tmp.sign()[..., None, :]
        return W

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        W0 = self.get_U0(U_arr)
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
            S = self.consensus(S)
            W = torch.linalg.qr(S).Q
            W = self._sign_adjust(W, W0)
            iter_frame.U = W
            yield iter_frame

