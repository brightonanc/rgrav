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
        root_arr = [None, 0.1]
        #root_arr = [None, 0.]
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            if 1 == it:
                #root_it = root_arr[it]
                #tmp0 = (1 / (1 - root_it)) * U
                #tmp1 = (-root_it / (1 - root_it)) * U
                #Z = (U_arr @ (U_arr.mT @ tmp0)) + tmp1
                #
                #Z = U_arr @ (U_arr.mT @ U)
                #
                root_it = root_arr[it]
                N = U_arr.shape[-2]
                I = U_arr.new_zeros([N]*2)
                I[range(N), range(N)] = 1
                tmp0 = (1 / (1 - root_it)) * ((U_arr @ U_arr.mT) - (root_it * I)) @ U
                Z = tmp0
            else:
                #root_it = root_arr[it] if it < len(root_arr) else 0.
                #root_itm1 = root_arr[it-1] if (it-1) < len(root_arr) else 0.
                #tmp0 = ((1 / (1 - root_it)) * U) \
                #        - ((1 / (1 - root_itm1)) * prev_U)
                #tmp1 = (((1 - (2 * root_it)) / (1 - root_it)) * U) \
                #        + ((root_itm1 / (1 + root_itm1)) * prev_U)
                #Z = (U_arr @ (U_arr.mT @ tmp0)) + tmp1
                #
                #Z = Z_mixed + (U_arr @ (U_arr.mT @ (U - prev_U)))
                #
                root_it = root_arr[it] if it < len(root_arr) else 0.
                root_itm1 = root_arr[it-1] if (it-1) < len(root_arr) else 0.
                N = U_arr.shape[-2]
                I = U_arr.new_zeros([N]*2)
                I[range(N), range(N)] = 1
                tmp0 = (1 / (1 - root_it)) * ((U_arr @ U_arr.mT) - (root_it * I)) @ U
                tmp1 = (1 / (1 - root_itm1)) * ((U_arr @ U_arr.mT) - (root_itm1 * I)) @ prev_U
                Z = Z_mixed + tmp0 - tmp1
            Z_mixed = self._fast_mix(Z)
            do_ortho = self.ortho_scheduler(it)
            prev_U = U
            if do_ortho:
                U = util.get_orthobasis(Z_mixed, mode=self.mode)
            else:
                U = Z_mixed
            #iter_frame.U = U
            iter_frame.U = util.get_orthobasis(Z_mixed, mode=self.mode)
            iter_frame.do_ortho = do_ortho
            yield iter_frame




#            Z1 = U_arr @ (U_arr.mT @ U0)
#            Z1_mixed = self._fast_mix(Z1)
#            prev_U1 = U0
#            U1 = util.get_orthobasis(Z1_mixed, mode=self.mode)
#
#            Z2 = Z1 + (U_arr @ (U_arr.mT @ (U1 - U0)))
#            Z2_mixed = self._fast_mix(Z2)
#            U2 = util.get_orthobasis(Z2_mixed, mode=self.mode)
#
#            Z3 = Z2 + (U_arr @ (U_arr.mT @ (U2 - U1)))
#            Z3_mixed = self._fast_mix(Z3)
#            U3 = util.get_orthobasis(Z3_mixed, mode=self.mode)
#
#            Z += (U_arr @ (U_arr.mT @ (U - prev_U)))
#            Z_mixed = self._fast_mix(Z)
#            U = util.get_orthobasis(Z_mixed, mode=self.mode)
#
#
#        U = U0
#        S = U0
#        AU = U0
#
#            tmp1 = U_arr @ (U_arr.mT @ U0)
#            S1 = S0 + tmp1 - AU0
#            AU1 = tmp1
#            S1_mixed = self._fast_mix(S1)
#            U1 = torch.linalg.qr(S1_mixed).Q
#
#            tmp2 = U_arr @ (U_arr.mT @ U1)
#            S2 = S1_mixed + tmp2 - AU1
#            AU2 = tmp2
#            S2_mixed = self._fast_mix(S2)
#            U2 = torch.linalg.qr(S2_mixed).Q
#
#            tmp3 = U_arr @ (U_arr.mT @ U2)
#            S3 = S2_mixed + tmp3 - AU2
#            AU3 = tmp3
#            S3_mixed = self._fast_mix(S3)
#            U3 = torch.linalg.qr(S3_mixed).Q
#
#            tmp = U_arr @ (U_arr.mT @ U)
#            S = S + tmp - AU
#            AU = tmp
#            S = self._fast_mix(S)
#            U = torch.linalg.qr(S).Q
#
