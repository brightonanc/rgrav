import numpy as np
from scipy.linalg import orth

class GRASTA:
    def __init__(self, n, d, rho=1, max_iterations=20, tol=1e-4):
        self.n = n  # Ambient dimension
        self.d = d  # Subspace dimension
        self.rho = rho  # ADMM parameter
        self.max_iterations = max_iterations  # Max ADMM iterations
        self.tol = tol  # ADMM tolerance
        
        # Initialize subspace estimate
        self.U = orth(np.random.randn(n, d))
        
        # Adaptive step size parameters
        self.eta = 0.1
        self.mu = 3
        self.level = 0
        
        # Previous gradient for adaptive step size
        self.prev_gradient = None

    def soft_threshold(self, x, lambda_):
        return np.sign(x) * np.maximum(0, np.abs(x) - lambda_)

    def _admm_solver(self, v_omega, omega):
        U_omega = self.U[omega, :]
        
        # Initialize variables
        s = np.zeros((len(omega), 1))
        w = np.zeros((self.d, 1))
        y = np.zeros((len(omega), 1))
        
        for _ in range(self.max_iterations):
            # Update w
            w = np.linalg.solve(U_omega.T @ U_omega, U_omega.T @ (v_omega - s + y / self.rho))
            
            # Update s
            s_new = v_omega - U_omega @ w + y
            s_new = self.soft_threshold(s_new, 1 / (1 + self.rho))
            
            # Update y
            y = y + self.rho * (U_omega @ w + s_new - v_omega)
            
            # Check convergence
            primal_residual = np.linalg.norm(U_omega @ w + s_new - v_omega)
            dual_residual = self.rho * np.linalg.norm(s_new - s)
            
            if primal_residual < self.tol and dual_residual < self.tol:
                break
            
            s = s_new
        
        return s, w, y

    def _compute_gradient(self, s, w, y, v_omega, omega):
        U_omega = self.U[omega, :]
        gamma1 = y + self.rho * (U_omega @ w + s - v_omega)
        gamma2 = U_omega.T @ gamma1
        gamma = np.zeros((self.n, 1))
        gamma[omega] = gamma1
        gamma -= self.U @ gamma2
        return gamma, w

    def _update_step_size(self, gamma, w):
        gradient = gamma @ w.T
        if self.prev_gradient is not None:
            inner_product = np.sum(self.prev_gradient * gradient)
            self.mu = max(self.mu + np.tanh(-inner_product), 1)
            
            if self.mu > 15:
                self.level += 1
                self.mu = 3
            elif self.mu < 1:
                self.level = max(0, self.level - 1)
                self.mu = 3
        
        self.eta = 0.1 * (2 ** -self.level) / (1 + self.mu)
        self.prev_gradient = gradient

    def _update_subspace(self, gamma, w):
        gradient = gamma @ w.T
        sigma = np.linalg.norm(gradient) * np.linalg.norm(w)
        w_norm = w / np.linalg.norm(w)
        gam_norm = gamma / np.linalg.norm(gamma)
        U_new = self.U + ((np.cos(self.eta * sigma) - 1) * (self.U @ w_norm) -
                          np.sin(self.eta * sigma) * gam_norm) @ w_norm.T
        self.U = orth(U_new)

    def add_data(self, v_omega, omega):
        # Solve ADMM
        s, w, y = self._admm_solver(v_omega, omega)
        
        # Compute gradient
        gamma, w = self._compute_gradient(s, w, y, v_omega, omega)
        
        # Update step size
        self._update_step_size(gamma, w)
        
        # Update subspace
        self._update_subspace(gamma, w)
        
        return s, w


import numpy as np
from scipy.linalg import orth

# Example usage
n = 500  # Ambient dimension
d = 5    # Subspace dimension

grasta = GRASTA(n, d)

# Generate true subspace
U_true = orth(np.random.randn(n, d))

# Simulation parameters
num_samples = 1000
outlier_fraction = 0.1  # Fraction of entries that are outliers
observation_fraction = 0.3  # Fraction of entries that are observed
noise_std = 1e-3  # Standard deviation of Gaussian noise

U_errs = []
for t in range(num_samples):
    # Generate weight vector
    w_t = np.random.randn(d, 1)
    
    # Generate sparse outlier vector
    s_t = np.zeros((n, 1))
    outlier_indices = np.random.choice(n, int(outlier_fraction * n), replace=False)
    s_t[outlier_indices] = np.random.randn(len(outlier_indices), 1) * np.max(np.abs(U_true @ w_t))
    
    # Generate Gaussian noise vector
    zeta_t = np.random.randn(n, 1) * noise_std
    
    # Generate full vector
    v_t = U_true @ w_t + s_t + zeta_t
    
    # Randomly sample some entries
    omega = np.random.choice(n, int(observation_fraction * n), replace=False)
    v_omega = v_t[omega]
    
    # Update GRASTA with new data
    s_est, w_est = grasta.add_data(v_omega, omega)
    U_errs.append(d - np.linalg.norm(grasta.U.T @ U_true) ** 2)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(U_errs)
plt.show()
