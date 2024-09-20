import _script_startup

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import util, HypercubeGraph, DeEPCA, DRGrAv

torch.set_default_dtype(torch.float64)


def main():
    seed = np.random.randint(2**32, dtype=np.uint64)
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    hc_dim = 5
    M = 2**hc_dim
    N = 43
    K = 12
    radius_ratio = 0.9

    U_arr = util.get_random_clustered_grassmannian_points(
        N=N,
        K=K,
        M=M,
        radius=radius_ratio*(0.25*torch.pi)
    )
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigvals, eigvecs = torch.linalg.eigh(P_avg)
    plt.plot(eigvals)
    plt.title('Problem spectrum')
    print(eigvals)
    plt.show()
    U_the = eigvecs[:, -K:]

    comm_W = HypercubeGraph.get_optimal_lapl_based_comm_W(hc_dim)
    cons_rounds = 3
    #print(torch.linalg.eigvalsh(comm_W))
    #exit()


    # DeEPCA
    deepca = DeEPCA(comm_W, cons_rounds)
    deepca_loss_hist = []
    deepca_gen = deepca.algo_iters(U_arr)
    # DRGrAv
    drgrav = DRGrAv(comm_W, cons_rounds, ortho_scheduler=lambda it: True)
    drgrav_loss_hist = []
    drgrav_gen = drgrav.algo_iters(U_arr)

    for _ in range(4):
        plt.gca().clear()

        # DeEPCA
        deepca_iter_frame = next(deepca_gen)
        deepca_loss_hist.append(
            (util.grassmannian_dist(deepca_iter_frame.U, U_the)**2).mean(0)
        )
        plt.semilogy(deepca_loss_hist, '-x', label='deepca')
        # DRGrAv
        drgrav_iter_frame = next(drgrav_gen)
        drgrav_loss_hist.append(
            (util.grassmannian_dist(drgrav_iter_frame.U, U_the)**2).mean(0)
        )
        plt.semilogy(drgrav_loss_hist, '-x', label='drgrav')

        plt.legend()
        plt.pause(0.01)
    plt.show()

if __name__ == '__main__':
    main()
