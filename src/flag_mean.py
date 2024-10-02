import torch
from types import SimpleNamespace

from .util import grassmannian_dist_chordal
from .algorithm_base import GrassmannianAveragingAlgorithm


def check_if_flagpole(flagpole):
    assert len(flagpole) > 0
    n_dims = flagpole[0].shape[0]
    for k, U in enumerate(flagpole):
        if U.shape != (n_dims, k + 1):
            return False
    return True

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
    assert flagpole1[0].shape == flagpole2[0].shape, f'{flagpole1[0].shape} != {flagpole2[0].shape}'
    assert check_if_flagpole(flagpole1)
    assert check_if_flagpole(flagpole2)
    rank = min(len(flagpole1), len(flagpole2))
    dist = grassmannian_dist_chordal(flagpole1[rank-1], flagpole2[rank-1])
    return dist

def flagpole_subspace_distance(flagpole, subspace):
    if not check_if_flagpole(flagpole):
        # try swapping them
        flagpole, subspace = subspace, flagpole
        if not check_if_flagpole(flagpole):
            raise ValueError('neither argument is not a valid flagpole')

    assert subspace.ndim == 2
    assert flagpole[0].shape[0] == subspace.shape[0]
    subspace_rank = subspace.shape[1]
    flagpole_subspace = flagpole[subspace_rank - 1]
    dist = grassmannian_dist_chordal(flagpole_subspace, subspace)
    return dist


class FlagMean(GrassmannianAveragingAlgorithm):
    def __init__(self):
        super().__init__()

    def algo_iters(self, U_arr):
        '''
        This isn't iterative, just returns a "flagpole" of subspaces
        '''
        iter_frame = SimpleNamespace()
        U_full = torch.cat([U_arr[i, :, :] for i in range(len(U_arr))], dim=1)
        U, _, __ = torch.linalg.svd(U_full, full_matrices=False)
        r = min(U.shape[1], U_arr.shape[2])
        iter_frame.U = get_flagpole(U[:, :r])
        return iter_frame

    def average(self, U_arr):
        '''
        override superclass method
        average should immediately return
        '''
        iter_frame = self.algo_iters(U_arr)
        return iter_frame.U
