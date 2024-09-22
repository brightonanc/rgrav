# script to generate synthetic data for testing
# algorithms:
# 1. fast PCA using RGrAv
# 2. Plain PCA using SVD
# 3. PCA using Randomized SVD

import torch

from src import *
from src.timing import *

def generate_synthetic_data(d, k, n, noise_level=1e-1):
    # d = dimension
    # k = rank
    # n = number of samples
    U = torch.randn(d, k)
    U = torch.linalg.qr(U).Q
    W = torch.randn(k, n)
    N = torch.randn(d, n) * noise_level
    return U @ W + N

def randomized_svd(X, k, p=50):
    m, n = X.shape
    Y = X @ torch.randn(n, k+p)
    Q = torch.linalg.qr(Y).Q
    B = Q.T @ X
    Uhat, s, V = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat
    return U[:, :k], s[:k], V[:k, :]

def rgrav_pca(X, k):
    n_split = X.shape[1] // k
    X_split = torch.split(X, n_split, dim=1)
    U_arr = []
    for X in X_split:
        U, _, __ = torch.linalg.svd(X, full_matrices=False)
        U_arr.append(U[:, :k])
    U_arr = torch.stack(U_arr)
    rgrav = RGrAv()
    U_ave = rgrav.average(U_arr)
    return U_ave

def rgrav_pca_subsample(X, k):
    n_split = X.shape[1] // k * 2
    X_split = torch.split(X, n_split, dim=1)
    U_arr = []
    for X in X_split:
        U, _, __ = torch.linalg.svd(X, full_matrices=False)
        U_pad = torch.zeros((X.shape[0], k))
        U_pad[:, :k//2] = U[:, :k//2]
        U_pad[:, k//2:] = torch.randn(X.shape[0], k//2)
        U_pad = torch.linalg.qr(U_pad).Q
        U_arr.append(U_pad)
    U_arr = torch.stack(U_arr)
    rgrav = RGrAv()
    U_ave = rgrav.average(U_arr)
    return U_ave

def pca_err(U1, U2):
    assert U1.shape == U2.shape, 'Shape mismatch: {} and {}'.format(U1.shape, U2.shape)
    k = U1.shape[1]
    return k - torch.norm(U1.T @ U2) ** 2


if __name__ == '__main__':
    d = 100
    k = 10
    # number of samples should be multiple of k
    n = k * 100
    X = generate_synthetic_data(d, k, n)
    print('data shape: ', X.shape)

    # do timing
    svd_time, svd_result = time_func(torch.linalg.svd, X, full_matrices=False)
    print('svd result shape: ', svd_result[0].shape)
    svd_result = svd_result[0][:, :k]
    print('svd time: ', svd_time)

    rgrav_time, rgrav_result = time_func(rgrav_pca, X, k)
    print('rgrav shape', rgrav_result.shape)
    print('rgrav time: ', rgrav_time, 'error: ', pca_err(svd_result, rgrav_result))

    rgrav_sub_time, rgrav_sub_result = time_func(rgrav_pca_subsample, X, k)
    print('rgrav_sub shape', rgrav_sub_result.shape)
    print('rgrav_sub time: ', rgrav_sub_time, 'error: ', pca_err(svd_result, rgrav_sub_result))

    rsvd_time, rsvd_result = time_func(randomized_svd, X, k)
    rsvd_result = rsvd_result[0][:, :k]
    print('rsvd time: ', rsvd_time, 'error: ', pca_err(svd_result, rsvd_result))


