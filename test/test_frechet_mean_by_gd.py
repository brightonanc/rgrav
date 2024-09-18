import torch
from scipy.stats import ortho_group

from src.frechet_mean_by_gd import FrechetMeanByGradientDescent
from src import util

def test_FrechetMeanByGradientDescent_runs(U_arr):
    FrechetMeanByGradientDescent().average(U_arr, max_iter=64)

def test_FrechetMeanByGradientDescent_is_correct():
    N = 20
    K = 7
    M = 64
    Q_center = torch.from_numpy(ortho_group.rvs(N))
    sigma = (0.99*(0.25*torch.pi)) * torch.rand(M//2, K, dtype=torch.float64)
    c = torch.cos(sigma)
    s = torch.sin(sigma)
    V = torch.from_numpy(ortho_group.rvs(K, size=M//2))
    U = torch.from_numpy(ortho_group.rvs(N-K, size=M//2)[:, :, :K])
    U_arr = torch.empty((M, N, K), dtype=torch.float64)
    U_arr[:M//2] = (Q_center[:, :K] @ (V * c[:, None, :])) \
            + (Q_center[:, K:] @ (U * s[:, None, :]))
    U_arr[M//2:] = (Q_center[:, :K] @ (V * c[:, None, :])) \
            - (Q_center[:, K:] @ (U * s[:, None, :]))

    U_emp = FrechetMeanByGradientDescent().average(U_arr, max_iter=128)
    U_the = Q_center[:, :K]
    err = util.grassmannian_dist(U_emp, U_the)**2
    assert 1e-5 > err
