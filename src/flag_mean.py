import torch
from types import SimpleNamespace

from .util import grassmannian_dist_chordal
from .algorithm_base import GrassmannianAveragingAlgorithm


def get_flagpole(U):
    assert U.ndim == 2
    n, d = U.shape
    flagpole = []
    for rank in range(1, d+1):
        _U = U[:, :rank]
        flagpole.append(_U.clone())
    return flagpole

def flagpole_distance(flagpole1, flagpole2):
    assert len(flagpole1) > 0 and len(flagpole2) > 0
    assert flagpole1[0].shape == flagpole2[0].shape
    rank = min(len(flagpole1), len(flagpole2)) - 1
    dist = grassmannian_dist_chordal(flagpole1[rank], flagpole2[rank])
    return dist


class FlagMean(GrassmannianAveragingAlgorithm):
    def __init__(self):
        super().__init__()

    def algo_iters(self, U_arr):
        '''
        This isn't iterative, just returns a "flagpole" of subspaces
        '''
        it = 0
        iter_frame = SimpleNamespace()
        U_full = torch.stack(U_arr, dim=2)
        print(U_arr.shape, U_full.shape)
        U, _, __ = torch.svd(U_full, full_matrices=False)
        print(U.shape)
        input()
        iter_frame.U = U
        return iter_frame
