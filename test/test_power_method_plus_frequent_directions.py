import torch

from src.power_method_plus_frequent_directions import PMFD, DPMFD
from src import util, HypercubeGraph, SimpleConsensus


def test_PMFD(U_arr):
    U_emp = PMFD().average(U_arr, max_iter=64)
    P_avg = (U_arr @ U_arr.mT).mean(0)
    U_the = torch.linalg.eigh(P_avg).eigenvectors[:, -U_arr.shape[-1]:]
    err = util.grassmannian_dist(U_emp, U_the)**2
    assert 1e-2 > err

def test_DPMFD(U_arr):
    consensus = SimpleConsensus(
        HypercubeGraph.get_optimal_lapl_based_comm_W(
            int(torch.log2(torch.tensor(U_arr.shape[0])).item()),
            dtype=torch.float64
        )
    )
    U_emp = DPMFD(consensus).average(U_arr, max_iter=64)
    P_avg = (U_arr @ U_arr.mT).mean(0)
    U_the = torch.linalg.eigh(P_avg).eigenvectors[:, -U_arr.shape[-1]:]
    err = (util.grassmannian_dist(U_emp, U_the)**2).mean()
    assert 1e-2 > err
