import _script_startup

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import util, HypercubeGraph, EPCA, RGrAv, RGrAv2

torch.set_default_dtype(torch.float64)


def main():
    seed = np.random.randint(2**32, dtype=np.uint64)
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    hc_dim = 5
    M = 2**hc_dim
    N = 90
    K = 12
    radius_ratio = 2.

    U_arr = util.get_random_clustered_grassmannian_points(
        N=N,
        K=K,
        M=M,
        radius=radius_ratio*(0.25*torch.pi)
    )
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigvals, eigvecs = torch.linalg.eigh(P_avg)
    plt.plot(eigvals, torch.linspace(0, 1, eigvals.numel()))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Problem spectrum')
    print(eigvals)
    plt.show()
    U_the = eigvecs[:, -K:]

    num_iter = 4

    # EPCA
    epca = EPCA()
    epca_loss_hist = []
    epca_gen = epca.algo_iters(U_arr)
    # RGrAv
    rgrav = RGrAv(num_iter=num_iter, lo=0.18, ortho_scheduler=lambda it: True)
    rgrav_loss_hist = []
    rgrav_gen = rgrav.algo_iters(U_arr)
    # RGrAv2
    rgrav2 = RGrAv2(lo=0.18, ortho_scheduler=lambda it: True)
    rgrav2_loss_hist = []
    rgrav2_gen = rgrav2.algo_iters(U_arr)

    for _ in range(num_iter+1):
        plt.gca().clear()

        # EPCA
        epca_iter_frame = next(epca_gen)
        epca_loss_hist.append((util.grassmannian_dist(
            util.get_orthobasis(epca_iter_frame.U, 'qr'),
            U_the
        )**2).mean(0))
        plt.semilogy(epca_loss_hist, '-x', label='epca')
        # RGrAv
        rgrav_iter_frame = next(rgrav_gen)
        rgrav_loss_hist.append((util.grassmannian_dist(
            util.get_orthobasis(rgrav_iter_frame.U, 'qr'),
            U_the
        )**2).mean(0))
        plt.semilogy(rgrav_loss_hist, '-x', label='rgrav')
        # RGrAv2
        rgrav2_iter_frame = next(rgrav2_gen)
        rgrav2_loss_hist.append((util.grassmannian_dist(
            util.get_orthobasis(rgrav2_iter_frame.U, 'qr'),
            U_the
        )**2).mean(0))
        plt.semilogy(rgrav2_loss_hist, '-x', label='rgrav2')

        plt.legend()
        plt.pause(0.01)
    plt.show()

if __name__ == '__main__':
    main()
