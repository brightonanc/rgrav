import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from src import *


U_correct = None
V_correct = None

def generate_low_rank_matrix(m, n, rank):
    U = torch.randn(m, rank) / (m ** .5)
    V = torch.randn(n, rank) / (n ** .5)
    global U_correct, V_correct
    U_correct = U.clone()
    V_correct = V.clone()
    return U @ V.T

def mask_matrix(X, mask_ratio=0.5):
    mask = torch.rand_like(X) > mask_ratio
    masked_X = X.clone()
    masked_X[~mask] = 0
    return masked_X, mask

def nuclear_norm(X):
    return torch.sum(torch.svd(X)[1])

def compute_V(U, X, mask):
    m, r = U.shape
    m2, n = X.shape
    
    assert m == m2, "U and M must have the same number of rows"
    assert mask.shape == X.shape, "Mask must have the same shape as M"
    
    V = torch.zeros((r, n))
    
    for j in range(n):
        observed = mask[:, j] > 0
        
        if not torch.any(observed):
            continue
        
        # Extract observed rows of U and M
        U_obs = U[observed, :]
        X_obs = X[observed, j]
        
        UTU = U_obs.T @ U_obs
        
        UTX = U_obs.T @ X_obs
        V[:, j] = torch.linalg.lstsq(UTU, UTX, rcond=None)[0]
    
    return V.T

def update_U_with_gradient_step(U, V, X, mask, step_size=0.1):
    residual = mask * (X - U @ V.T)
    euclidean_grad = -2 * residual @ V
    
    # Project onto the tangent space
    sym_part = (U.T @ euclidean_grad + euclidean_grad.T @ U) / 2
    riemannian_grad = euclidean_grad - U @ sym_part
    
    U_new = U - step_size * riemannian_grad
    
    # Retract back to the manifold using QR decomposition
    svd = torch.linalg.svd(U_new, full_matrices=False)
    U_new = svd[0] @ svd[2]
    
    return U_new

def procrustes_U(U, V, X, mask):
    B = X @ (V @ torch.linalg.pinv(V.T @ V))
    W, _, Z = torch.linalg.svd(B, full_matrices=False)
    return W @ Z.T

def matrix_completion(X_masked, mask, rank, num_iterations=1000):
    U = torch.linalg.qr(torch.randn(X_masked.shape[0], rank)).Q
    losses = []
    U_losses = []

    for _ in tqdm(range(num_iterations)):
        # update V
        V = compute_V(U, X_masked, mask)

        # update U
        U = update_U_with_gradient_step(U, V, X_masked, mask)
        # U = procrustes_U(U, V, X_masked, mask)

        # loss metrics
        loss = torch.linalg.norm(mask * (X_masked - U @ V.T))
        U_loss = rank - torch.linalg.norm(U.T @ U_correct) ** 2
        losses.append(loss)
        U_losses.append(U_loss)

    X_recovered = U @ V.T
    
    return X_recovered, losses, U_losses

def evaluate_recovery(X_true, X_recovered, mask):
    mse = torch.mean((X_true[~mask] - X_recovered[~mask])**2)
    rel_error = torch.norm(X_true[~mask] - X_recovered[~mask]) / torch.norm(X_true[~mask])
    return mse, rel_error

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    m, n = 100, 1000
    rank = 5
    X_true = generate_low_rank_matrix(m, n, rank)

    # Create masked version
    mask_ratio = 0.80
    X_masked, mask = mask_matrix(X_true, mask_ratio)
    
    # distribute the load
    n_workers = 10
    k = n // n_workers
    U_arr = []
    V_arr = []
    X_arr = []
    mask_arr = []
    for i in range(n_workers):
        _U = torch.linalg.qr(torch.randn(m, rank)).Q
        _V = torch.randn(k, rank)
        _X = X_true[:, i*k:(i+1)*k]
        _mask = mask[:, i*k:(i+1)*k]
        U_arr.append(_U)
        V_arr.append(_V)
        X_arr.append(_X)
        mask_arr.append(_mask)
    U_arr = torch.stack(U_arr, dim=0)
    V_arr = torch.stack(V_arr, dim=0)
    X_arr = torch.stack(X_arr, dim=0)
    mask_arr = torch.stack(mask_arr, dim=0)

    # move everything to GPU
    # global U_correct
    # U_correct = U_correct.to('cuda')
    # U_arr = U_arr.to('cuda')
    # V_arr = V_arr.to('cuda')
    # X_arr = X_arr.to('cuda')
    # mask_arr = mask_arr.to('cuda')
    
    # Perform matrix completion
    num_iterations = 10000
    # X_recovered, losses, U_losses = matrix_completion(X_masked, mask, rank, num_iterations)
    
    ave_algo = AsymptoticRGrAv(0.05)

    lr = 1e-1
    # distribute matrix completion across workers
    losses, U_losses = [], []
    for t in tqdm(range(num_iterations)):
        for i in range(n_workers):
            V_arr[i] = compute_V(U_arr[i], X_arr[i], mask_arr[i])
            U_arr[i] = update_U_with_gradient_step(U_arr[i], V_arr[i], X_arr[i], mask_arr[i], lr)
        
        # do consensus on Us
        ave_algo.average(U_arr, max_iter=1)

        loss = sum(torch.linalg.norm(_mask * (_X - _U @ _V.T)) 
                   for _U, _V, _X, _mask in zip(U_arr, V_arr, X_arr, mask_arr))
        U_loss = sum(rank - torch.linalg.norm(_U.T @ U_correct) ** 2 for _U in U_arr)
        losses.append(loss / n_workers)
        U_losses.append(U_loss / n_workers)

    plt.figure()
    plt.imshow(mask)
    plt.figure()
    plt.subplot(121)
    plt.plot(losses)
    plt.subplot(122)
    plt.plot(U_losses)

    pairwise_dists = torch.zeros((n_workers, n_workers))
    for i in range(n_workers):
        for j in range(n_workers):
            U1 = U_arr[i]
            U2 = U_arr[j]
            dist = rank - torch.linalg.norm(U1.T @ U2) ** 2
            pairwise_dists[i, j] = dist
    
    plt.figure()
    plt.imshow(pairwise_dists)

    X_recovered = []
    for i in range(n_workers):
        _Xr = U_arr[i] @ V_arr[i].T
        X_recovered.append(_Xr)
    X_recovered = torch.cat(X_recovered, dim=1)

    # Evaluate results on unobserved entries
    mse, rel_error = evaluate_recovery(X_true, X_recovered, mask)
    print(f"MSE on unobserved entries: {mse:.6f}")
    print(f"Relative error on unobserved entries: {rel_error:.6f}")
    
    # Evaluate on all entries
    mse_all, rel_error_all = evaluate_recovery(X_true, X_recovered, torch.zeros_like(mask).bool())
    print(f"MSE on all entries: {mse_all:.6f}")
    print(f"Relative error on all entries: {rel_error_all:.6f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(X_true.cpu(), cmap='viridis')
    plt.colorbar()
    plt.title('True Matrix')
    
    plt.subplot(132)
    plt.imshow(X_masked.cpu(), cmap='viridis')
    plt.colorbar()
    plt.title('Masked Matrix')
    
    plt.subplot(133)
    plt.imshow(X_recovered.cpu(), cmap='viridis')
    plt.colorbar()
    plt.title('Recovered Matrix')
    
    plt.tight_layout()
    
    # Plot singular values
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(torch.svd(X_true)[1].cpu().numpy(), label='True')
    plt.plot(torch.svd(X_recovered)[1].cpu().numpy(), label='Recovered')
    plt.title('Singular Values')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(torch.svd(X_true)[1].cpu().numpy(), label='True')
    plt.plot(torch.svd(X_recovered)[1].cpu().numpy(), label='Recovered')
    plt.semilogy()
    plt.title('Singular Values (log scale)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
