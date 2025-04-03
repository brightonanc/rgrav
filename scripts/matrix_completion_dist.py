import torch
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm


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

def compute_V(U, M, mask):
    m, r = U.shape
    m2, n = M.shape
    
    assert m == m2, "U and M must have the same number of rows"
    assert mask.shape == M.shape, "Mask must have the same shape as M"
    
    V = torch.zeros((r, n))
    
    for j in range(n):
        observed = mask[:, j] > 0
        
        if not torch.any(observed):
            continue
        
        # Extract observed rows of U and M
        U_obs = U[observed, :]
        M_obs = M[observed, j]
        
        UTU = U_obs.T @ U_obs
        
        UTM = U_obs.T @ M_obs
        V[:, j] = torch.linalg.lstsq(UTU, UTM, rcond=None)[0]
    
    return V.T

def update_U_with_gradient_step(U, V, M, mask, step_size=0.1):
    residual = mask * (M - U @ V.T)
    euclidean_grad = -2 * residual @ V
    
    # Project onto the tangent space
    sym_part = (U.T @ euclidean_grad + euclidean_grad.T @ U) / 2
    riemannian_grad = euclidean_grad - U @ sym_part
    
    U_new = U - step_size * riemannian_grad
    
    # Retract back to the manifold using QR decomposition
    svd = torch.linalg.svd(U_new, full_matrices=False)
    U_new = svd[0] @ svd[2]
    
    return U_new

def procrustes_U(U, V, M, mask):
    B = M @ (V @ torch.linalg.pinv(V.T @ V))
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
    """Evaluate the recovery quality.
    Computes MSE and relative error on specified entries (usually unobserved ones)."""
    mse = torch.mean((X_true[~mask] - X_recovered[~mask])**2)
    rel_error = torch.norm(X_true[~mask] - X_recovered[~mask]) / torch.norm(X_true[~mask])
    return mse, rel_error

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Generate synthetic data
    m, n = 100, 100
    rank = 5
    X_true = generate_low_rank_matrix(m, n, rank)
    
    # Create masked version
    mask_ratio = 0.80
    X_masked, mask = mask_matrix(X_true, mask_ratio)
    
    # Perform matrix completion
    num_iterations = 1000
    
    X_recovered, losses, U_losses = matrix_completion(X_masked, mask, rank, num_iterations)
    
    plt.figure()
    plt.imshow(mask)
    plt.figure()
    plt.subplot(121)
    plt.plot(losses)
    plt.subplot(122)
    plt.plot(U_losses)
    
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
