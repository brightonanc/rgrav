import _script_startup

import torch
import matplotlib.pyplot as plt

from src import ChebyshevMagicNumbers, BalancedChebyshevMagicNumbers, HypercubeGraph

torch.set_default_dtype(torch.float64)


def visualize_hypercube_comm_behavior():
    hc_dim = 10
    cons_rounds = 8
    comm_W_psd = HypercubeGraph.get_positive_optimal_lapl_based_comm_W(hc_dim)
    eigval_psd = torch.linalg.eigvalsh(comm_W_psd)
    assert 1e-14 > (1 - eigval_psd[-1])
    eigval_psd = eigval_psd[:-1]
    cmn = ChebyshevMagicNumbers(eigval_psd.abs().max().item())
    cmn.visualize_polynomial(
        cons_rounds,
        which_poly='both',
        samples=eigval_psd,
        focus_roots=False,
    )
    comm_W_gen = HypercubeGraph.get_optimal_lapl_based_comm_W(hc_dim)
    eigval_gen = torch.linalg.eigvalsh(comm_W_gen)
    assert 1e-14 > (1 - eigval_gen[-1])
    eigval_gen = eigval_gen[:-1]
    cmn = BalancedChebyshevMagicNumbers(eigval_gen.abs().max().item())
    cmn.visualize_polynomial(
        cons_rounds,
        which_poly='both',
        samples=eigval_gen,
        focus_roots=False,
    )

if __name__ == '__main__':
    visualize_hypercube_comm_behavior()
