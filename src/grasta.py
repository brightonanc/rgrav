import torch


class GRASTA:

    def __init__(self, rho=None):
        if rho is None:
            self.rho = 10

    def add_data(self, X, Omega):
        pass


if __name__ == '__main__':
    def draw_sample(U, sparsity):
        d, k = U.shape
        w = torch.randn(k, 1)
        s = torch.randn(d, 1)
        eta = torch.zeros(d, 1)
        return U @ w + s + eta
    
    d = 100
    k = 10
    sparsity = 0.05

    U_true = torch.linalg.qr(torch.randn(d, k)).Q


