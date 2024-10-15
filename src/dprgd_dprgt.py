import torch
from types import SimpleNamespace

from .algorithm_base import DecentralizedConsensusAlgorithm
from . import util


class DPRGD(DecentralizedConsensusAlgorithm):
    """ See https://arxiv.org/abs/2304.08241 """

    def __init__(self, consensus, eta):
        super().__init__(consensus)
        self.eta = eta

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = self.get_U0(U_arr)
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            alpha = self.eta / (it**0.5)
            grad = -(U_arr @ (U_arr.mT @ U))
            grad -= U @ (U.mT @ grad)
            term0 = self.consensus(U) - (alpha * grad)
            U = util.get_orthobasis(term0)
            iter_frame.U = U
            yield iter_frame


class DPRGT(DecentralizedConsensusAlgorithm):
    """ See https://arxiv.org/abs/2304.08241 """

    def __init__(self, consensus, eta):
        super().__init__(consensus)
        self.eta = eta

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = self.get_U0(U_arr)
        grad = -(U_arr @ (U_arr.mT @ U))
        grad -= U @ (U.mT @ grad)
        S = grad
        prev_grad = grad
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            V = S - (U @ (U.mT @ S))
            alpha = self.eta
            term0 = self.consensus(U) - (alpha * V)
            U = util.get_orthobasis(term0)
            grad = -(U_arr @ (U_arr.mT @ U))
            grad -= U @ (U.mT @ grad)
            S = self.consensus(S) + grad - prev_grad
            prev_grad = grad
            iter_frame.U = U
            yield iter_frame

