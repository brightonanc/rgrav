import _script_startup

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import util, HypercubeGraph, EPCA, FiniteRGrAv, DeEPCA, FiniteDRGrAv, FiniteDRGrAv2, get_constant_period_scheduler, AsymptoticRGrAv, AsymptoticRGrAv2, FiniteDRGrAv3, AsymptoticDRGrAv

torch.set_default_dtype(torch.float64)


def main():
    seed = np.random.randint(2**32, dtype=np.uint64)
    seed = 0
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    hc_dim = 5
    M = 2**hc_dim
    N = 300
    K = 64
    radius_ratio = 0.1

    U_arr = util.get_random_clustered_grassmannian_points(
        N=N,
        K=K,
        M=M,
        radius=radius_ratio*(0.25*torch.pi)
    )
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_the = eigvec[:, -K:]

    comm_W = HypercubeGraph.get_positive_optimal_lapl_based_comm_W(hc_dim)
    cons_rounds = 20

    alpha_ideal = eigval[eigval.diff().argmax()].item()
    alpha = alpha_ideal

    num_iter = 100

    centralized = False
    decentralized = True

    if True:
        print(f'{alpha=}')
        plt.plot(torch.cat((torch.tensor([0]), eigval)), torch.linspace(0, 1, eigval.numel()+1), '-x')
        plt.plot([alpha]*2, [0, 1], '--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(f'{alpha=}')
        plt.show()

    # ============================== Centralized ==============================
    if centralized:
        # EPCA
        epca = EPCA()
        epca_loss_hist = []
        epca_gen = epca.algo_iters(U_arr)
        # FiniteRGrAv
        frgrav = FiniteRGrAv(alpha, num_iter)
        frgrav_loss_hist = []
        frgrav_gen = frgrav.algo_iters(U_arr)
        # AsymptoticRGrAv
        argrav = AsymptoticRGrAv(alpha)
        argrav_loss_hist = []
        argrav_gen = argrav.algo_iters(U_arr)
        # AsymptoticRGrAv2
        argrav2 = AsymptoticRGrAv2(alpha)
        argrav2_loss_hist = []
        argrav2_gen = argrav2.algo_iters(U_arr)
    # ============================= Decentralized =============================
    if decentralized:
        # DeEPCA
        deepca = DeEPCA(comm_W, cons_rounds)
        deepca_loss_hist = []
        deepca_gen = deepca.algo_iters(U_arr)
        # FiniteDRGrAv
        fdrgrav = FiniteDRGrAv(alpha, num_iter, comm_W, cons_rounds)
        fdrgrav_loss_hist = []
        fdrgrav_gen = fdrgrav.algo_iters(U_arr)
        # FiniteDRGrAv1a
        fdrgrav1a = FiniteDRGrAv(alpha, num_iter, comm_W, cons_rounds,
                ortho_scheduler=get_constant_period_scheduler(2))
        fdrgrav1a_loss_hist = []
        fdrgrav1a_gen = fdrgrav1a.algo_iters(U_arr)
        # FiniteDRGrAv1b
        fdrgrav1b = FiniteDRGrAv(alpha, num_iter, comm_W, cons_rounds,
                ortho_scheduler=lambda it: False)
        fdrgrav1b_loss_hist = []
        fdrgrav1b_gen = fdrgrav1b.algo_iters(U_arr)
        # FiniteDRGrAv2
        fdrgrav2 = FiniteDRGrAv2(alpha, num_iter, comm_W, cons_rounds)
        fdrgrav2_loss_hist = []
        fdrgrav2_gen = fdrgrav2.algo_iters(U_arr)
        # FiniteDRGrAv2a
        fdrgrav2a = FiniteDRGrAv2(alpha, num_iter, comm_W, cons_rounds,
                ortho_scheduler=get_constant_period_scheduler(2))
        fdrgrav2a_loss_hist = []
        fdrgrav2a_gen = fdrgrav2a.algo_iters(U_arr)
        # FiniteDRGrAv2zf
        fdrgrav2zf = FiniteDRGrAv2(alpha, num_iter, comm_W, cons_rounds, zero_first=True)
        fdrgrav2zf_loss_hist = []
        fdrgrav2zf_gen = fdrgrav2zf.algo_iters(U_arr)
        # FiniteDRGrAv3
        fdrgrav3 = FiniteDRGrAv3(alpha, num_iter, comm_W, cons_rounds,
                ortho_scheduler=get_constant_period_scheduler(1))
        fdrgrav3_loss_hist = []
        fdrgrav3_gen = fdrgrav3.algo_iters(U_arr)
        # AsymptoticDRGrAv
        adrgrav = AsymptoticDRGrAv(alpha, comm_W, cons_rounds)
        adrgrav_loss_hist = []
        adrgrav_gen = adrgrav.algo_iters(U_arr)

    for _ in range(num_iter+1):
        plt.gca().clear()

        ## ============================ Centralized ============================
        if centralized:
            # EPCA
            epca_iter_frame = next(epca_gen)
            epca_loss_hist.append(
                util.grassmannian_dist(epca_iter_frame.U, U_the)**2
            )
            plt.semilogy(epca_loss_hist, '-x', label='epca')
            # FiniteRGrAv
            frgrav_iter_frame = next(frgrav_gen)
            frgrav_loss_hist.append(
                util.grassmannian_dist(frgrav_iter_frame.U, U_the)**2
            )
            plt.semilogy(frgrav_loss_hist, '-x', label='frgrav')
            # AsymptoticRGrAv
            argrav_iter_frame = next(argrav_gen)
            argrav_loss_hist.append(
                util.grassmannian_dist(argrav_iter_frame.U, U_the)**2
            )
            plt.semilogy(argrav_loss_hist, '-x', label='argrav')
            # AsymptoticRGrAv2
            argrav2_iter_frame = next(argrav2_gen)
            argrav2_loss_hist.append(
                util.grassmannian_dist(argrav2_iter_frame.U, U_the)**2
            )
            plt.semilogy(argrav2_loss_hist, '-x', label='argrav2')
        # =========================== Decentralized ===========================
        if decentralized:
            # DeEPCA
            deepca_iter_frame = next(deepca_gen)
            deepca_loss_hist.append(
                (util.grassmannian_dist(deepca_iter_frame.U, U_the)**2).mean(0)
            )
            plt.semilogy(deepca_loss_hist, '-x', label='deepca')
            # # FiniteDRGrAv
            # fdrgrav_iter_frame = next(fdrgrav_gen)
            # fdrgrav_loss_hist.append(
            #     (util.grassmannian_dist(fdrgrav_iter_frame.U, U_the)**2).mean(0)
            # )
            # plt.semilogy(fdrgrav_loss_hist, '-x', label='fdrgrav')
            # # FiniteDRGrAv1a
            # fdrgrav1a_iter_frame = next(fdrgrav1a_gen)
            # fdrgrav1a_loss_hist.append(
            #     (util.grassmannian_dist(fdrgrav1a_iter_frame.U, U_the)**2).mean(0)
            # )
            # plt.semilogy(fdrgrav1a_loss_hist, '-o', label='fdrgrav1a')
            # # FiniteDRGrAv1b
            # fdrgrav1b_iter_frame = next(fdrgrav1b_gen)
            # fdrgrav1b_loss_hist.append(
            #     (util.grassmannian_dist(fdrgrav1b_iter_frame.U, U_the)**2).mean(0)
            # )
            # plt.semilogy(fdrgrav1b_loss_hist, '-o', label='fdrgrav1b')
            # FiniteDRGrAv2
            fdrgrav2_iter_frame = next(fdrgrav2_gen)
            fdrgrav2_loss_hist.append(
                (util.grassmannian_dist(fdrgrav2_iter_frame.U, U_the)**2).mean(0)
            )
            plt.semilogy(fdrgrav2_loss_hist, '-x', label='fdrgrav2')
            # FiniteDRGrAv2zf
            fdrgrav2zf_iter_frame = next(fdrgrav2zf_gen)
            fdrgrav2zf_loss_hist.append(
                (util.grassmannian_dist(fdrgrav2zf_iter_frame.U, U_the)**2).mean(0)
            )
            plt.semilogy(fdrgrav2zf_loss_hist, '-x', label='fdrgrav2zf')
            # # FiniteDRGrAv2a
            # fdrgrav2a_iter_frame = next(fdrgrav2a_gen)
            # fdrgrav2a_loss_hist.append(
            #     (util.grassmannian_dist(fdrgrav2a_iter_frame.U, U_the)**2).mean(0)
            # )
            # plt.semilogy(fdrgrav2a_loss_hist, '-x', label='fdrgrav2a')
            # # FiniteDRGrAv3
            # fdrgrav3_iter_frame = next(fdrgrav3_gen)
            # fdrgrav3_loss_hist.append(
            #     (util.grassmannian_dist(fdrgrav3_iter_frame.U, U_the)**2).mean(0)
            # )
            # plt.semilogy(fdrgrav3_loss_hist, '-x', label='fdrgrav3')
            # AsymptoticDRGrAv
            adrgrav_iter_frame = next(adrgrav_gen)
            adrgrav_loss_hist.append(
                (util.grassmannian_dist(adrgrav_iter_frame.U, U_the)**2).mean(0)
            )
            plt.semilogy(adrgrav_loss_hist, '-x', label='adrgrav')

        plt.legend()
        plt.pause(0.01)
    #plt.title(title)
    #plt.pause(0.1)
    plt.show()

def main_R():
    seed = np.random.randint(2**32, dtype=np.uint64)
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    hc_dim = 4
    M = 2**hc_dim
    N = 300
    K = 13
    radius_ratio = 200

    U_arr = util.get_random_clustered_grassmannian_points(
        N=N,
        K=K,
        M=M,
        radius=radius_ratio*(0.25*torch.pi)
    )
    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_the = eigvec[:, -K:]

    comm_W = HypercubeGraph.get_positive_optimal_lapl_based_comm_W(hc_dim)
    cons_rounds = 8

    alpha_ideal = eigval[eigval.diff().argmax()].item()
    alpha = alpha_ideal

    num_iter = 16

    if True:
        print(f'{alpha=}')
        plt.plot(torch.cat((torch.tensor([0]), eigval)), torch.linspace(0, 1, eigval.numel()+1), '-x')
        plt.plot([alpha]*2, [0, 1], '--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(f'{alpha=}')
        plt.show()

    # FiniteDRGrAv
    fdrgrav = FiniteDRGrAv(alpha, num_iter, comm_W, cons_rounds, ortho_scheduler=lambda it: False)
    fdrgrav_loss_hist = []
    fdrgrav_gen = fdrgrav.algo_iters(U_arr)

    for it in range(num_iter+1):

        # FiniteDRGrAv
        fdrgrav_iter_frame = next(fdrgrav_gen)
        I = torch.zeros((M, K, K))
        I[:, range(K), range(K)] = 1.
        U, (S,) = util.get_orthobasis(
            fdrgrav_iter_frame.U,
            others_X=(I,),
        )

        tmp = fdrgrav.cons_rounds
        fdrgrav.cons_rounds = 2 * tmp
        cons_S = fdrgrav._fast_mix(S)
        fdrgrav.cons_rounds = tmp

        def proc(x):
            mag = x.mean(0).abs() + 1e-3
            x_linf = (x.amax(0) - x.amin(0)).abs()
            x_l2 = ((x - x.mean(0, keepdims=True))**2).mean(0)**0.5

            ind = K
            mag = mag[:ind, :ind]
            x_linf = x_linf[:ind, :ind]
            x_linf /= mag
            x_l2 = x_l2[:ind, :ind]
            x_l2 /= mag

            triu_ind = torch.triu_indices(*x_linf.shape)
            x_linf_triu = x_linf[triu_ind[0], triu_ind[1]]
            x_mag_triu = mag[triu_ind[0], triu_ind[1]]

            return x_linf, x_l2, x_linf_triu, x_mag_triu
        
        S_linf, S_l2, S_linf_triu, S_mag_triu = proc(S)
        cons_S_linf, cons_S_l2, cons_S_linf_triu, cons_S_mag_triu = proc(cons_S)

        #plt.subplot(1, 2, 1)
        #plt.imshow(S_linf)
        #plt.colorbar()
        #plt.title('linf')
        #plt.subplot(1, 2, 2)
        #plt.imshow(S_l2)
        #plt.colorbar()
        #plt.title('l2')

        plot = plt.semilogy
        plot(S_linf_triu, '-x', label='S')
        #plot(S_mag_triu, '--', label='S')
        plot(cons_S_linf_triu, '-x', label='cons_S')
        #plot(cons_S_mag_triu, '--', label='cons_S')
        plt.legend()

        plt.suptitle(f'{it=}')
        plt.show()

if __name__ == '__main__':
    main()
