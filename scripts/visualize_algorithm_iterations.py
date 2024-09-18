import _script_startup

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import util, HypercubeGraph, DBPM, EPCA, DeEPCA, \
        FrechetMeanByGradientDescent, PMFD, DPMFD, RGrAv, DRGrAv

torch.set_default_dtype(torch.float64)


def main():
    seed = np.random.randint(2**32, dtype=np.uint64)
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    hc_dim = 3
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
    U_the = torch.linalg.eigh(P_avg).eigenvectors[:, -K:]

    comm_W = HypercubeGraph.get_optimal_lapl_based_comm_W(hc_dim)
    cons_rounds = 8

    # ============================== Centralized ==============================
    # EPCA
    epca = EPCA()
    epca_loss_hist = []
    epca_gen = epca.algo_iters(U_arr)
    # FMGD
    fmgd = FrechetMeanByGradientDescent()
    fmgd_loss_hist = []
    fmgd_gen = fmgd.algo_iters(U_arr)
    # PMFD
    pmfd = PMFD()
    pmfd_loss_hist = []
    pmfd_gen = pmfd.algo_iters(U_arr)
    # RGrAv
    rgrav = RGrAv()
    rgrav_loss_hist = []
    rgrav_gen = rgrav.algo_iters(U_arr)
    # ============================= Decentralized =============================
    # DBPM
    dbpm = DBPM(comm_W, cons_rounds, mode='qr-stable')
    dbpm_loss_hist = []
    dbpm_gen = dbpm.algo_iters(U_arr)
    # DeEPCA
    deepca = DeEPCA(comm_W, cons_rounds)
    deepca_loss_hist = []
    deepca_gen = deepca.algo_iters(U_arr)
    # DPMFD
    dpmfd = DPMFD(comm_W, cons_rounds)
    dpmfd_loss_hist = []
    dpmfd_gen = dpmfd.algo_iters(U_arr)
    # DRGrAv
    drgrav = DRGrAv(comm_W, cons_rounds)
    drgrav_loss_hist = []
    drgrav_gen = drgrav.algo_iters(U_arr)

    for _ in range(64):
        plt.gca().clear()

        # ============================ Centralized ============================
        # EPCA
        epca_iter_frame = next(epca_gen)
        epca_loss_hist.append(
            util.grassmannian_dist(epca_iter_frame.U, U_the)**2
        )
        plt.semilogy(epca_loss_hist, '-x', label='epca')
        # FMGD
        fmgd_iter_frame = next(fmgd_gen)
        fmgd_loss_hist.append(
            util.grassmannian_dist(fmgd_iter_frame.U, U_the)**2
        )
        plt.semilogy(fmgd_loss_hist, '-x', label='fmgd')
        # PMFD
        pmfd_iter_frame = next(pmfd_gen)
        pmfd_loss_hist.append(
            util.grassmannian_dist(pmfd_iter_frame.U, U_the)**2
        )
        plt.semilogy(pmfd_loss_hist, '-x', label='pmfd')
        # RGrAv
        rgrav_iter_frame = next(rgrav_gen)
        rgrav_loss_hist.append(
            util.grassmannian_dist(rgrav_iter_frame.U, U_the)**2
        )
        plt.semilogy(rgrav_loss_hist, '-x', label='rgrav')
        # =========================== Decentralized ===========================
        # DBPM
        dbpm_iter_frame = next(dbpm_gen)
        dbpm_loss_hist.append(
            (util.grassmannian_dist(dbpm_iter_frame.U, U_the)**2).mean(0)
        )
        plt.semilogy(dbpm_loss_hist, '-x', label='dbpm')
        # DeEPCA
        deepca_iter_frame = next(deepca_gen)
        deepca_loss_hist.append(
            (util.grassmannian_dist(deepca_iter_frame.U, U_the)**2).mean(0)
        )
        plt.semilogy(deepca_loss_hist, '-x', label='deepca')
        # DPMFD
        dpmfd_iter_frame = next(dpmfd_gen)
        dpmfd_loss_hist.append(
            (util.grassmannian_dist(dpmfd_iter_frame.U, U_the)**2).mean(0)
        )
        plt.semilogy(dpmfd_loss_hist, '-x', label='dpmfd')
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
