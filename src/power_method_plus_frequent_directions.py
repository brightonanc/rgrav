import torch
from types import SimpleNamespace

from .algorithm_base import GrassmannianAveragingAlgorithm, \
        DecentralizedConsensusAlgorithm
from . import util

class PMFD(GrassmannianAveragingAlgorithm):

    def __init__(self):
        super().__init__()

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        M, N, K = U_arr.shape
        Q = self.get_U0(U_arr)
        B = U_arr.new_zeros(N, 2*K)
        iter_frame.U = Q
        yield iter_frame
        while True:
            iter_frame = SimpleNamespace()
            Z_arr = U_arr @ (U_arr.mT @ Q)
            Z_hat = Z_arr.mean(0)
            U, _, Vh = torch.linalg.svd(Z_hat, full_matrices=False)
            Q = U @ Vh
            B[..., K:] = Q
            U_hat, S_hat, _ = torch.linalg.svd(B, full_matrices=False)
            delta = S_hat[..., [K]]**2
            S_tild = (S_hat[..., :K]**2 - delta)**0.5
            B[..., :K] = U_hat[..., :K] * S_tild[..., None, :]
            B[..., K:] = 0.
            U, _, Vh = torch.linalg.svd(B[..., :K], full_matrices=False)
            iter_frame.U = U @ Vh
            yield iter_frame


class DPMFD(DecentralizedConsensusAlgorithm):

    def __init__(self, comm_W, cons_rounds=8):
        super().__init__(comm_W, cons_rounds)

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        M, N, K = U_arr.shape
        Q = self.get_U0(U_arr)
        B = U_arr.new_zeros(M, N, 2*K)
        iter_frame.U = Q
        yield iter_frame
        while True:
            iter_frame = SimpleNamespace()
            Z_arr = U_arr @ (U_arr.mT @ Q)
            Z_hat = self._consensus(Z_arr)
            U, _, Vh = torch.linalg.svd(Z_hat, full_matrices=False)
            Q = U @ Vh
            B[..., K:] = Q
            U_hat, S_hat, _ = torch.linalg.svd(B, full_matrices=False)
            delta = S_hat[..., [K]]**2
            S_tild = (S_hat[..., :K]**2 - delta)**0.5
            B[..., :K] = U_hat[..., :K] * S_tild[..., None, :]
            B[..., K:] = 0.
            U, _, Vh = torch.linalg.svd(B[..., :K], full_matrices=False)
            iter_frame.U = U @ Vh
            yield iter_frame



