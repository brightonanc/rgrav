import torch
from types import SimpleNamespace
import warnings

from .algorithm_base import GrassmannianAveragingAlgorithm, \
        DecentralizedConsensusAlgorithm
from .chebyshev import ChebyshevMagicNumbers
from . import util


class FiniteRGrAv(GrassmannianAveragingAlgorithm):

    def __init__(self, alpha, num_iter, zero_first=False, mode='qr-stable',
             ortho_scheduler=None):
        super().__init__()
        self.root_arr = ChebyshevMagicNumbers(alpha).get_root_arr(num_iter)
        self.num_iter = num_iter
        if zero_first:
            self.root_arr = self.root_arr.flip(0)
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = self.get_U0(U_arr)
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
            A = U_arr @ (U_arr.mT @ U)
            Z_hat = (fac0 * A.mean(0)) - (fac1 * U)
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
        super().__init__()
        self.cmn = ChebyshevMagicNumbers(alpha)
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
            A = U_arr @ (U_arr.mT @ U)
            if 1 == it:
                Z_hat = A.mean(0)
            else:
                a = self.cmn.a(it)
                b = self.cmn.b(it)
                c = self.cmn.c(it)
                Z_hat = a * (A.mean(0) + (b * U) + (c * prev_U))
            do_ortho = self.ortho_scheduler(it)
            prev_U = U
            if do_ortho:
                U, (prev_U,) = util.get_orthobasis(
                    Z_hat,
                    mode=self.mode,
                    others_X=(prev_U,)
                )
            else:
                U = Z_hat
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame


class FiniteDRGrAv(DecentralizedConsensusAlgorithm):

    def __init__(self, alpha, num_iter, consensus, zero_first=False,
                mode='qr-stable', ortho_scheduler=None):
        super().__init__(consensus)
        self.root_arr = ChebyshevMagicNumbers(alpha).get_root_arr(num_iter)
        self.num_iter = num_iter
        if zero_first:
            self.root_arr = self.root_arr.flip(0)
        self.mode = mode
        if ortho_scheduler is None:
            ortho_scheduler = lambda it: True
        self.ortho_scheduler = ortho_scheduler

    def algo_iters(self, U_arr):
        it = 0
        iter_frame = SimpleNamespace()
        U = self.get_U0(U_arr)
        iter_frame.U = U
        iter_frame.within_num_iter = True
        yield iter_frame
        while True:
            it += 1
            iter_frame = SimpleNamespace()
            within_num_iter = (it-1) < self.num_iter
            if not within_num_iter:
                warnings.warn(
                    'FiniteDRGrAv: algorithm is now running longer than'
                    ' originally specified by num_iter'
                )
            root = self.root_arr[it-1] if within_num_iter else 0.
            fac0 = 1 / (1 - root)
            fac1 = root / (1 - root)
            term0 = fac0 * (U_arr @ (U_arr.mT @ U))
            term1 = fac1 * U
            Y = term0 - term1
            if 1 == it:
                Z = Y
            else:
                Z = Z_hat - prev_Y + Y
            Z_hat = self.consensus(Z)
            do_ortho = self.ortho_scheduler(it)
            if do_ortho:
                U = util.get_orthobasis(Z_hat, mode=self.mode)
            else:
                U = Z_hat
            prev_Y = Y
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            iter_frame.within_num_iter = within_num_iter
            yield iter_frame


class AsymptoticDRGrAv(DecentralizedConsensusAlgorithm):

    def __init__(self, alpha, consensus, mode='qr-stable',
                ortho_scheduler=None):
        super().__init__(consensus)
        self.cmn = ChebyshevMagicNumbers(alpha)
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
            if 1 == it:
                Y = U_arr @ (U_arr.mT @ U)
                Z = Y
            else:
                a = self.cmn.a(it)
                b = self.cmn.b(it)
                c = self.cmn.c(it)
                Y = a * ((U_arr @ (U_arr.mT @ U)) + (b * U) + (c * prev_U))
                Z = Z_hat - prev_Y + Y
            Z_hat = self.consensus(Z)
            do_ortho = self.ortho_scheduler(it)
            prev_U = U
            if do_ortho:
                U, (prev_U,) = util.get_orthobasis(
                    Z_hat,
                    mode=self.mode,
                    others_X=(prev_U,)
                )
            else:
                U = Z_hat
            prev_Y = Y
            iter_frame.U = U
            iter_frame.do_ortho = do_ortho
            yield iter_frame

