import torch
from types import SimpleNamespace

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
        self.eta = eta
        self.init_idx = init_idx

    def algo_iters(self, U_arr):
        iter_frame = SimpleNamespace()
        U = U_arr[self.init_idx].clone()
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

