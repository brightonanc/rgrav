import torch
from types import SimpleNamespace

from .algorithm_base import DecentralizedConsensusAlgorithm
from . import util


class DBPM(DecentralizedConsensusAlgorithm):
    """ Decentralized Block Power Method """

    def __init__(self, comm_W, cons_rounds=8, mode='qr', ortho_scheduler=None):
        super().__init__(comm_W, cons_rounds)
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = self.get_U0(U_arr)
        iter_frame.U = U
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            Z = U_arr @ (U_arr.mT @ U)
            Z_hat = self._consensus(Z)
            do_ortho = self.ortho_scheduler(it)
            if do_ortho:
                U = util.get_orthobasis(Z_hat, mode=self.mode)
            else:
                U = Z_hat
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame

