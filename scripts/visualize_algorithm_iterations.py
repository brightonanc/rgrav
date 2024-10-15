import _script_startup

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import util, HypercubeGraph, ChebyshevConsensus, BPM, DBPM, DeEPCA, \
        FrechetMeanByGradientDescent, PMFD, DPMFD, FiniteRGrAv, \
        AsymptoticRGrAv, FiniteDRGrAv, AsymptoticDRGrAv

torch.set_default_dtype(torch.float64)


def main():
    seed = np.random.randint(2**32, dtype=np.uint64)
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    hc_dim = 3
    M = 2**hc_dim
    N = 200
    K = 12
    radius_ratio = 0.9

    U_arr = util.get_random_clustered_grassmannian_points(
        N=N,
        K=K,
        M=M,
        radius=radius_ratio*(0.25*torch.pi)
    )
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_the = eigvec[:, -K:]

    plt.plot(eigval, (1+torch.arange(eigval.numel()))/eigval.numel(), '-x')
    plt.plot([eigval[-K]]*2, [0, 1], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('CDF of eigenvalues')
    plt.show()

    consensus = ChebyshevConsensus(
        HypercubeGraph.get_positive_optimal_lapl_based_comm_W(hc_dim),
        cons_rounds = 8,
    )

    alpha_ideal = eigval[eigval.diff().argmax()].item()
    alpha = alpha_ideal

    num_iter = 32

    flag_S_experiment = False
    if flag_S_experiment:
        kwargs = dict(
            ortho_scheduler=lambda x: x in [0,1,2,3,4,6,8,12,16],
            use_S=True,
        )
    else:
        kwargs = {}

    # ============================== Centralized ==============================
    # BPM
    bpm = BPM()
    bpm_loss_hist = []
    bpm_gen = bpm.algo_iters(U_arr)
    # FMGD
    fmgd = FrechetMeanByGradientDescent()
    fmgd_loss_hist = []
    fmgd_gen = fmgd.algo_iters(U_arr)
    # PMFD
    pmfd = PMFD()
    pmfd_loss_hist = []
    pmfd_gen = pmfd.algo_iters(U_arr)
    # FiniteRGrAv
    frgrav = FiniteRGrAv(alpha, num_iter, **kwargs)
    frgrav_loss_hist = []
    frgrav_gen = frgrav.algo_iters(U_arr)
    # AsymptoticRGrAv
    argrav = AsymptoticRGrAv(alpha, **kwargs)
    argrav_loss_hist = []
    argrav_gen = argrav.algo_iters(U_arr)
    # ============================= Decentralized =============================
    # DBPM
    dbpm = DBPM(consensus)
    dbpm_loss_hist = []
    dbpm_gen = dbpm.algo_iters(U_arr)
    # DeEPCA
    deepca = DeEPCA(consensus)
    deepca_loss_hist = []
    deepca_gen = deepca.algo_iters(U_arr)
    # DPMFD
    dpmfd = DPMFD(consensus)
    dpmfd_loss_hist = []
    dpmfd_gen = dpmfd.algo_iters(U_arr)
    # FiniteDRGrAv
    fdrgrav = FiniteDRGrAv(alpha, num_iter, consensus, **kwargs)
    fdrgrav_loss_hist = []
    fdrgrav_gen = fdrgrav.algo_iters(U_arr)
    # AsymptoticDRGrAv
    adrgrav = AsymptoticDRGrAv(alpha, consensus, **kwargs)
    adrgrav_loss_hist = []
    adrgrav_gen = adrgrav.algo_iters(U_arr)

    for _ in range(num_iter+1):
        plt.gca().clear()

        # ============================ Centralized ============================
        # BPM
        bpm_iter_frame = next(bpm_gen)
        bpm_loss_hist.append(
            util.grassmannian_dist(bpm_iter_frame.U, U_the)**2
        )
        plt.semilogy(bpm_loss_hist, '-x', label='bpm')
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
        # FiniteRGrAv
        frgrav_iter_frame = next(frgrav_gen)
        frgrav_loss_hist.append(
            util.grassmannian_dist(frgrav_iter_frame.U, U_the)**2
        )
        plt.semilogy(frgrav_loss_hist, '--x', label='frgrav')
        # AsymptoticRGrAv
        argrav_iter_frame = next(argrav_gen)
        argrav_loss_hist.append(
            util.grassmannian_dist(argrav_iter_frame.U, U_the)**2
        )
        plt.semilogy(argrav_loss_hist, '--x', label='argrav')
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
        # FiniteDRGrAv
        fdrgrav_iter_frame = next(fdrgrav_gen)
        fdrgrav_loss_hist.append(
            (util.grassmannian_dist(fdrgrav_iter_frame.U, U_the)**2).mean(0)
        )
        plt.semilogy(fdrgrav_loss_hist, '--x', label='fdrgrav')
        # AsymptoticDRGrAv
        adrgrav_iter_frame = next(adrgrav_gen)
        adrgrav_loss_hist.append(
            (util.grassmannian_dist(adrgrav_iter_frame.U, U_the)**2).mean(0)
        )
        plt.semilogy(adrgrav_loss_hist, '--x', label='adrgrav')

        plt.legend()
        plt.pause(0.01)
    plt.show()

if __name__ == '__main__':
    main()
