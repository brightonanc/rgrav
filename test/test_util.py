import pytest
import torch
import numpy as np
import pymanopt
from scipy.stats import ortho_group

from src import util


@pytest.mark.parametrize('mode', [
    'qr',
    'svd',
    'qr-stable',
])
def test_get_orthobasis(mode):
    M = 23
    N = 20
    K = 7
    torch.manual_seed(0)
    X = torch.randn(M, N, K)
    U = util.get_orthobasis(X, mode=mode)
    X_ = U @ (U.mT @ X)
    assert 1e-5 > (X - X_).abs().max()
    others_X = [
        torch.randn(M, N, K),
        torch.randn(M, 13, K),
        torch.randn(M, K, K),
        torch.randn(M, 1, K),
        torch.randn(N, K),
        torch.randn(13, K),
        torch.randn(K, K),
        torch.randn(1, K),
    ]
    U, others_U = util.get_orthobasis(X, mode=mode, others_X=others_X)
    S_inv = U.mT @ X
    X_ = U @ S_inv
    assert 1e-5 > (X - X_).abs().max()
    others_X_ = [other_U @ S_inv for other_U in others_U]
    assert all(
        1e-5 > (other_X - other_X_).abs().max()
        for other_X, other_X_ in zip(others_X, others_X_)
    )

@pytest.fixture(params=[
    (2, 1),
    (13, 1),
    (4, 2),
    (19, 2),
    (32, 10),
])
def GrNK(request):
    N, K = request.param
    return pymanopt.manifolds.Grassmann(N, K)

def test_grassmannian_dist(GrNK):
    np.random.seed(0)
    num_trials = 128
    for _ in range(num_trials):
        U1 = GrNK.random_point()
        U2 = GrNK.random_point()
        ref = GrNK.dist(U1, U2)
        U1 = torch.from_numpy(U1)
        U2 = torch.from_numpy(U2)
        test = util.grassmannian_dist(U1, U2).item()
        assert 1e-10 > abs(test - ref)

def test_grassmannian_log(GrNK):
    np.random.seed(0)
    num_trials = 128
    for _ in range(num_trials):
        U1 = GrNK.random_point()
        U2 = GrNK.random_point()
        ref = torch.from_numpy(GrNK.log(U1, U2))
        U1 = torch.from_numpy(U1)
        U2 = torch.from_numpy(U2)
        test = util.grassmannian_log(U1, U2)
        assert 1e-10 > (test - ref).abs().max()

def test_grassmannian_exp(GrNK):
    np.random.seed(0)
    num_trials = 128
    for _ in range(num_trials):
        U = GrNK.random_point()
        tang = GrNK.random_tangent_vector(U)
        ref = torch.from_numpy(GrNK.exp(U, tang))
        U = torch.from_numpy(U)
        tang = torch.from_numpy(tang)
        test = util.grassmannian_exp(U, tang)
        assert 1e-6 > util.grassmannian_dist(ref, test)

@pytest.mark.parametrize('radius', [
    0.1*(0.25*torch.pi),
    0.99*(0.25*torch.pi),
    0.45*torch.pi,
])
def test_get_random_clustered_grassmannian_points(radius):
    N = 20
    K = 7
    M = 64
    if (0.25*torch.pi) > radius:
        U_arr = util.get_random_clustered_grassmannian_points(
            N=N,
            K=K,
            M=M,
            radius=radius
        )
        dist = util.grassmannian_linfty_dist(U_arr[None, :], U_arr[:, None])
        assert dist.max() < (2 * radius)
    else:
        Q_center = torch.from_numpy(ortho_group.rvs(N))
        U_arr = util.get_random_clustered_grassmannian_points(
            N=N,
            K=K,
            M=M,
            radius=radius,
            Q_center=Q_center,
        )
        dist = util.grassmannian_linfty_dist(U_arr, Q_center)
        assert dist.max() < radius

