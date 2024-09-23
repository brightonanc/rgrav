import pytest
import torch
import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt

from src.rgrav import ChebyshevMagicNumbers, FiniteRGrAv, AsymptoticRGrAv, FiniteDRGrAv
from src import util, HypercubeGraph

#def test_RGrAv(U_arr):
#    U_emp = RGrAv().average(U_arr, max_iter=64)
#    P_avg = (U_arr @ U_arr.mT).mean(0)
#    U_the = torch.linalg.eigh(P_avg).eigenvectors[:, -U_arr.shape[-1]:]
#    err = util.grassmannian_dist(U_emp, U_the)**2
#    assert 1e-8 > err


def test_FiniteRGrAv(U_arr):
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_the = eigvec[:, -U_arr.shape[-1]:]
    alpha_ideal = eigval[eigval.diff().argmax()].item()
    num_iter = 64
    algo = FiniteRGrAv(alpha_ideal, num_iter)
    U_emp = algo.average(U_arr, max_iter=num_iter)
    err = util.grassmannian_dist(U_emp, U_the)**2
    assert 1e-8 > err

@pytest.mark.filterwarnings('ignore:ChebyshevMagicNumber')
def test_AsymptoticRGrAv(U_arr):
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_the = eigvec[:, -U_arr.shape[-1]:]
    alpha_ideal = eigval[eigval.diff().argmax()].item()
    algo = AsymptoticRGrAv(alpha_ideal)
    U_emp = algo.average(U_arr, max_iter=64)
    err = util.grassmannian_dist(U_emp, U_the)**2
    assert 1e-8 > err

@pytest.mark.parametrize('alpha, std_root_arr', [
    (
        0.07,
        torch.tensor([
            0.,
            0.82842712,
        ]),
    ), (
        0.5,
        torch.tensor([
            0.,
            0.82842712,
        ]),
    ), (
        0.07,
        torch.tensor([
            0.,
            0.46410161,
            0.92820323,
        ]),
    ), (
        0.5,
        torch.tensor([
            0.,
            0.46410161,
            0.92820323,
        ]),
    ), (
        0.07,
        torch.tensor([
            0.,
            0.07538206,
            0.21466993,
            0.39665832,
            0.59364113,
            0.77562951,
            0.91491739,
            0.99029944,
        ]),
    ), (
        0.5,
        torch.tensor([
            0.,
            0.07538206,
            0.21466993,
            0.39665832,
            0.59364113,
            0.77562951,
            0.91491739,
            0.99029944,
        ]),
    ),
])
def test_FiniteRGrAv_root_placement(alpha, std_root_arr):
    np.random.seed(0)
    torch.manual_seed(0)
    M = std_root_arr.numel() + 1
    K = 3
    N = 10 * M * K
    U_arr = torch.from_numpy(ortho_group.rvs(N)[:M*K]).view(M, K, N).mT
    scale = (alpha * std_root_arr)**0.5
    scale = torch.cat((scale, torch.tensor([1.])))
    scale *= M**0.5
    U_arr *= scale[:, None, None]
    P_avg = (U_arr @ U_arr.mT).mean(0)
    # P_avg is now a matrix with eigenvalues at the desired roots
    num_iter = M - 1
    for zero_first in [True, False]:
        algo = FiniteRGrAv(
            alpha,
            num_iter,
            zero_first=zero_first,
            ortho_scheduler=lambda it: False
        )
        prev_norm = None
        for i, iter_frame in enumerate(algo.algo_iters(U_arr)):
            U = iter_frame.U
            cond_arr = 1e-6 > (U_arr[:-1].mT @ U).abs().amax(dim=(-1,-2))
            print(cond_arr)
            if zero_first:
                assert torch.all(cond_arr[:i])
            else:
                assert torch.all(cond_arr[(M-1)-i:])
            if i < 3:
                # This assertion is finicky because the algorithm is just too good
                # - it pushes eigenvalues close to zero pretty quickly even when
                # they're not the roots already applied. So instead of checking
                # this every iteration, only check the first few (3) as a sanity
                # check
                if zero_first:
                    assert not torch.any(cond_arr[max(i,1):])
                else:
                    assert not torch.any(cond_arr[1:(M-1)-i])
            norm = (U_arr[-1].mT @ U).norm()
            if prev_norm is None:
                prev_norm = norm
            assert 1e-6 > torch.abs(norm - prev_norm)
            prev_norm = norm
            if i >= num_iter:
                break

def test_AsymptoticRGrAv_vs_FiniteRGrAv(U_arr):
    #TODO
    pass

def test_FiniteDRGrAv(U_arr):
    comm_W = HypercubeGraph.get_positive_optimal_lapl_based_comm_W(
        int(torch.log2(torch.tensor(U_arr.shape[0])).item()),
        dtype=torch.float64
    )
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_the = eigvec[:, -U_arr.shape[-1]:]
    alpha_ideal = eigval[eigval.diff().argmax()].item()
    num_iter = 64
    algo = FiniteDRGrAv(alpha_ideal, num_iter, comm_W, cons_rounds=100, ortho_scheduler=lambda it: True)
    U_emp = algo.average(U_arr, max_iter=num_iter)
    err = (util.grassmannian_dist(U_emp, U_the)**2).max()
    assert 1e-8 > err

