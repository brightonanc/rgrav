import torch
from types import SimpleNamespace

from .algorithm_base import DecentralizedConsensusAlgorithm


class SarletteSepulchre(DecentralizedConsensusAlgorithm):
    """ See https://arxiv.org/abs/0811.4275 """

    def __init__(self, A, eta):
        super().__init__(None)
        self.A = A
        self.eta = eta

    def mix(self, X):
        return (self.A[:, :, None, None] * X).sum(0)

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        Y = U_arr.clone()
        iter_frame.U = Y
        yield iter_frame
        while True:
            iter_frame = SimpleNamespace()
            M = Y[:, None].mT @ Y[None, :]
            term0 = Y[:, None] @ M
            term1 = Y[None, :] @ M.mT @ M
            term2 = self.mix(term0 - term1)
            alpha = self.eta
            Y = Y + (4 * alpha * term2)
            iter_frame.U = Y
            yield iter_frame

