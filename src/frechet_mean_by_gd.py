import torch
from types import SimpleNamespace
import warnings

from .algorithm_base import GrassmannianAveragingAlgorithm
from . import util

class FrechetMeanByGradientDescent(GrassmannianAveragingAlgorithm):
    """ Computes the Frechet Mean by Gradient Descent """

    def __init__(self, eta=1., init_idx=0):
        """
        Parameters
        ----------
        eta : float
            The step size for gradient descent
        init_idx : int
            The index of the initial point used for averaging
        """
        super().__init__()
        self.eta = eta
        self.init_idx = init_idx

    def get_U0(self, U_arr):
        if self.U0 is None:
            self.U0 = U_arr[self.init_idx].clone()
        elif self.init_idx is not None:
            warnings.warn(
                'FrechetMeanByGradientDescent: init_idx was non-None yet an'
                ' initial U0 was provided. The U0 will be used and the'
                ' init_idx will be ignored.'
            )
        return self.U0

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        U = self.get_U0(U_arr)
        iter_frame.U = U
        iter_frame.err_criterion = torch.nan
        yield iter_frame
        while True:
            iter_frame = SimpleNamespace()
            g = util.grassmannian_log(U, U_arr).mean(0)
            U = util.grassmannian_exp(U, self.eta * g)
            iter_frame.U = U
            iter_frame.err_criterion = g.norm()
            yield iter_frame

