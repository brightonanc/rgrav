import torch

from src.deepca import EPCA, DeEPCA
from src import util, HypercubeGraph

def test_EPCA(U_arr):
    U_emp = EPCA().average(U_arr, max_iter=64)
    P_avg = (U_arr @ U_arr.mT).mean(0)
    U_the = torch.linalg.eigh(P_avg).eigenvectors[:, -U_arr.shape[-1]:]
    err = util.grassmannian_dist(U_emp, U_the)**2
    assert 1e-8 > err

def test_DeEPCA(U_arr):
    comm_W = HypercubeGraph.get_optimal_lapl_based_comm_W(
        int(torch.log2(torch.tensor(U_arr.shape[0])).item()),
        dtype=torch.float64
    )
    U_emp = DeEPCA(comm_W).average(U_arr, max_iter=64)
    P_avg = (U_arr @ U_arr.mT).mean(0)
    U_the = torch.linalg.eigh(P_avg).eigenvectors[:, -U_arr.shape[-1]:]
    err = (util.grassmannian_dist(U_emp, U_the)**2).mean()
    assert 1e-8 > err
