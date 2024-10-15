"""


In order to replicate the results of the Decentralized Grassmannian Averaging
Experiments section (Section 5.1 at the time of writing), one need only run
this script with a single command line argument for the save directory of
results.


The plots are saved automatically. For timing data, the stdout stream from the
run of this script may be processed using the following shell one-liner:

# grep ' Being Timed: \|algo_iters' [teed_filename] | awk '{ if($1 ~ /^[0-9]+(\.[0-9]+)?$/) { print "Time_to_Tolerance: "(1000*($4/100)) } else { print $1": "$4 } }'

For this reason, it is recommended that this script be run with tee or a
standard file redirection.

An expected result would then look something like the following

Algorithm: DRGrAv
Tolerance: 0.001
Total_Iterations: 3
Time_to_Tolerance: 35.41
Algorithm: DeEPCA
Tolerance: 0.001
Total_Iterations: 4
Time_to_Tolerance: 35.83
...


"""




import _script_startup

import torch
import numpy as np
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import cProfile
import pstats
from tqdm import tqdm
import signal
def raise_timeout_error(x, y):
    raise TimeoutError
signal.signal(signal.SIGALRM, raise_timeout_error)

from src import util, HypercubeGraph, CycleGraph, \
        BalancedChebyshevConsensus, FastMixDeEPCA, DeEPCA, \
        SarletteSepulchre, DPRGD, DPRGT, GrassmannianGossip, FiniteDRGrAv, \
        AsymptoticDRGrAv

torch.set_default_dtype(torch.float64)


def get_normal_grassmann_points(N, K, M, stddev):
    Q_center = torch.from_numpy(ortho_group.rvs(N))
    theta = stddev * torch.randn(M, K)
    c = torch.cos(theta)
    s = torch.sin(theta)
    if 1 < K:
        V = torch.from_numpy(ortho_group.rvs(K, size=M))
    else:
        V = 1 - (2 * torch.randint(2, (M, 1, 1)).type(torch.float64))
    if 1 < (N-K):
        U = torch.from_numpy(ortho_group.rvs(N-K, size=M)[:, :, :K])
    else:
        U = 1 - (2 * torch.randint(2, (M, 1, 1)).type(torch.float64))
    U_arr = (Q_center[:, :K] @ (V * c[:, None, :])) \
            + (Q_center[:, K:] @ (U * s[:, None, :]))
    return U_arr, Q_center[:, :K]

def get_problem_data_0():
    seed = np.random.randint(2**32, dtype=np.uint64)
    seed = 0
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    hc_dim = 6
    M = 2**hc_dim
    N = 150
    K = 30
    stddev = 0.25*np.pi

    U_arr, U_ctr = get_normal_grassmann_points(
        N=N,
        K=K,
        M=M,
        stddev=stddev
    )

    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_iam = eigvec[:, -K:]

    alpha_ideal = eigval[eigval.diff().argmax()].item()
    alpha = 0.15

    return U_arr, U_iam, alpha, hc_dim, M

def main_0():
    U_arr, U_iam, alpha, hc_dim, M = get_problem_data_0()

    cons_rounds = 10
    consensus = BalancedChebyshevConsensus(
        HypercubeGraph.get_optimal_lapl_based_comm_W(hc_dim),
        cons_rounds=cons_rounds,
    )
    adjacency = HypercubeGraph.get_adjacency(hc_dim)
    edges = HypercubeGraph.get_edges(hc_dim)

    num_iter = 6

    # AsymptoticDRGrAv
    adrgrav = AsymptoticDRGrAv(alpha, consensus)
    adrgrav_loss_hist = []
    adrgrav_disagree_hist = []
    adrgrav_gen = adrgrav.algo_iters(U_arr)
    # DeEPCA
    deepca = DeEPCA(consensus)
    deepca_loss_hist = []
    deepca_disagree_hist = []
    deepca_gen = deepca.algo_iters(U_arr)
    # DPRGD
    dprgd = DPRGD(consensus, eta=1.3)
    dprgd_loss_hist = []
    dprgd_disagree_hist = []
    dprgd_gen = dprgd.algo_iters(U_arr)
    # DPRGT
    dprgt = DPRGT(consensus, eta=8.0e-1)
    dprgt_loss_hist = []
    dprgt_disagree_hist = []
    dprgt_gen = dprgt.algo_iters(U_arr)
    # SarletteSepulchre
    sarsep = SarletteSepulchre(adjacency, eta=4.8e-2)
    sarsep_loss_hist = []
    sarsep_disagree_hist = []
    sarsep_gen = sarsep.algo_iters(U_arr)
    # GrassmannianGossip
    gossip = GrassmannianGossip(edges, a=1.0, b=6.9e-4, rounds=cons_rounds)
    gossip_loss_hist = []
    gossip_disagree_hist = []
    gossip_gen = gossip.algo_iters(U_arr)

    plot_mode = 'loss'

    for it in range(num_iter+1):
        plt.gca().clear()

        # AsymptoticDRGrAv
        adrgrav_iter_frame = next(adrgrav_gen)
        adrgrav_loss_hist.append((util.grassmannian_extrinsic_dist(
            adrgrav_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            adrgrav_disagree_hist.append(np.nan)
        else:
            adrgrav_disagree_hist.append((util.grassmannian_extrinsic_dist(
                adrgrav_iter_frame.U[None, :],
                adrgrav_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(adrgrav_loss_hist, '-x', label='DRGrAv')
        else:
            plt.semilogy(adrgrav_disagree_hist, '-x', label='DRGrAv')
        # DeEPCA
        deepca_iter_frame = next(deepca_gen)
        deepca_loss_hist.append((util.grassmannian_extrinsic_dist(
            deepca_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            deepca_disagree_hist.append(np.nan)
        else:
            deepca_disagree_hist.append((util.grassmannian_extrinsic_dist(
                deepca_iter_frame.U[None, :],
                deepca_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(deepca_loss_hist, '-x', label='DeEPCA')
        else:
            plt.semilogy(deepca_disagree_hist, '-x', label='DeEPCA')
        # DPRGD
        dprgd_iter_frame = next(dprgd_gen)
        dprgd_loss_hist.append((util.grassmannian_extrinsic_dist(
            dprgd_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            dprgd_disagree_hist.append(np.nan)
        else:
            dprgd_disagree_hist.append((util.grassmannian_extrinsic_dist(
                dprgd_iter_frame.U[None, :],
                dprgd_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(dprgd_loss_hist, '-x', label='DPRGD')
        else:
            plt.semilogy(dprgd_disagree_hist, '-x', label='DPRGD')
        # DPRGT
        dprgt_iter_frame = next(dprgt_gen)
        dprgt_loss_hist.append((util.grassmannian_extrinsic_dist(
            dprgt_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            dprgt_disagree_hist.append(np.nan)
        else:
            dprgt_disagree_hist.append((util.grassmannian_extrinsic_dist(
                dprgt_iter_frame.U[None, :],
                dprgt_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(dprgt_loss_hist, '-x', label='DPRGT')
        else:
            plt.semilogy(dprgt_disagree_hist, '-x', label='DPRGT')
        # SarletteSepulchre
        sarsep_iter_frame = next(sarsep_gen)
        sarsep_loss_hist.append((util.grassmannian_extrinsic_dist(
            sarsep_iter_frame.U,
            U_iam
        )**2).mean(0))
        sarsep_disagree_hist.append((util.grassmannian_extrinsic_dist(
            sarsep_iter_frame.U[None, :],
            sarsep_iter_frame.U[:, None]
        )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(sarsep_loss_hist, '-x', label='COM')
        else:
            plt.semilogy(sarsep_disagree_hist, '-x', label='COM')
        # GrassmannianGossip
        gossip_iter_frame = next(gossip_gen)
        gossip_loss_hist.append((util.grassmannian_extrinsic_dist(
            gossip_iter_frame.U,
            U_iam
        )**2).mean(0))
        gossip_disagree_hist.append((util.grassmannian_extrinsic_dist(
            gossip_iter_frame.U[None, :],
            gossip_iter_frame.U[:, None]
        )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(gossip_loss_hist, '-x', label='Gossip')
        else:
            plt.semilogy(gossip_disagree_hist, '-x', label='Gossip')

        plt.legend()
        plt.xlabel('Iteration')
        if 'loss' == plot_mode:
            plt.ylabel('MSE')
        else:
            plt.ylabel('MSD')
        plt.title('Hypercube Graph')
        plt.pause(0.01)
    #plt.show()

    fig_loss = plt.figure(figsize=(6, 4))
    ax_loss = fig_loss.gca()
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), adrgrav_loss_hist, label='DRGrAv', linestyle='-', marker='o', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), deepca_loss_hist, label='DeEPCA', linestyle='--', marker='s', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), dprgd_loss_hist, label='DPRGD', linestyle='-.', marker='^', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), dprgt_loss_hist, label='DPRGT', linestyle=':', marker='d', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), sarsep_loss_hist, label='COM', linestyle='-', marker='x', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), gossip_loss_hist, label='Gossip', linestyle='--', marker='*', markersize=5)
    ax_loss.legend()
    ax_loss.set_xlabel('Communication Rounds')
    ax_loss.set_ylabel('MSE')
    ax_loss.set_title('Hypercube Graph')
    fig_loss.savefig(f'{DIR}/hypercube_mse.pdf')
    fig_disagree = plt.figure(figsize=(6, 4))
    ax_disagree = fig_disagree.gca()
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), adrgrav_disagree_hist, label='DRGrAv', linestyle='-', marker='o', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), deepca_disagree_hist, label='DeEPCA', linestyle='--', marker='s', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), dprgd_disagree_hist, label='DPRGD', linestyle='-.', marker='^', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), dprgt_disagree_hist, label='DPRGT', linestyle=':', marker='d', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), sarsep_disagree_hist, label='COM', linestyle='-', marker='x', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), gossip_disagree_hist, label='Gossip', linestyle='--', marker='*', markersize=5)
    ax_disagree.legend()
    ax_disagree.set_xlabel('Communication Rounds')
    ax_disagree.set_ylabel('MSD')
    ax_disagree.set_title('Hypercube Graph')
    fig_disagree.savefig(f'{DIR}/hypercube_msd.pdf')

def timing_0(which, tol):
    U_arr, U_iam, alpha, hc_dim, M = get_problem_data_0()

    cons_rounds = 10
    consensus = BalancedChebyshevConsensus(
        HypercubeGraph.get_optimal_lapl_based_comm_W(hc_dim),
        cons_rounds=cons_rounds,
    )
    adjacency = HypercubeGraph.get_adjacency(hc_dim)
    edges = HypercubeGraph.get_edges(hc_dim)

    # AsymptoticDRGrAv
    adrgrav = AsymptoticDRGrAv(alpha, consensus)
    adrgrav.cmn.a(32)
    # DeEPCA
    deepca = DeEPCA(consensus)
    # DPRGD
    dprgd = DPRGD(consensus, eta=1.3)
    # DPRGT
    dprgt = DPRGT(consensus, eta=8.0e-2)
    # SarletteSepulchre
    sarsep = SarletteSepulchre(adjacency, eta=4.8e-2)
    # GrassmannianGossip
    gossip = GrassmannianGossip(edges, a=1.0, b=6.9e-4, rounds=cons_rounds)

    def get_mse_loss(U):
        return (util.grassmannian_extrinsic_dist(U, U_iam)**2).mean(0)
    def get_msd_loss(U):
        return (util.grassmannian_extrinsic_dist(U[None, :], U[:, None])**2).sum((0,1)) / (M*(M-1))

    match which:
        case 'DRGrAv':
            algo = adrgrav
            get_loss = get_mse_loss
        case 'DeEPCA':
            algo = deepca
            get_loss = get_mse_loss
        case 'DPRGD':
            algo = dprgd
            get_loss = get_mse_loss
        case 'DPRGT':
            algo = dprgt
            get_loss = get_mse_loss
        case 'COM':
            algo = sarsep
            get_loss = get_msd_loss
        case 'Gossip':
            algo = gossip
            get_loss = get_msd_loss
        case _:
            assert False

    num_trials = 100
    max_runtime = 100000

    prof = cProfile.Profile()
    prof.enable()
    try:
        for _ in tqdm(range(num_trials)):
            signal.alarm(max_runtime)
            for it, algo_iter_frame in enumerate(algo.algo_iters(U_arr)):
                algo_loss = get_loss(algo_iter_frame.U)
                if algo_loss < tol:
                    final_it = it
                    break
    except TimeoutError:
        final_it = 'too_long'
    signal.alarm(0)
    prof.disable()
    stats = pstats.Stats(prof)
    print(f'Algorithm Being Timed: {which}')
    print(f'Tolerance Being Timed: {tol}')
    print(f'Total_Iterations Being Timed: {final_it}')
    stats.strip_dirs().sort_stats('cumtime').print_stats(12)

def get_problem_data_1():
    seed = np.random.randint(2**32, dtype=np.uint64)
    seed = 0
    print(f'{seed=}')
    torch.manual_seed(seed)
    np.random.seed(seed)
    M = 64
    N = 150
    K = 30
    stddev = 0.25*np.pi

    U_arr, U_ctr = get_normal_grassmann_points(
        N=N,
        K=K,
        M=M,
        stddev=stddev
    )

    P_avg = (U_arr @ U_arr.mT).mean(0)
    eigval, eigvec = torch.linalg.eigh(P_avg)
    U_iam = eigvec[:, -K:]

    alpha_ideal = eigval[eigval.diff().argmax()].item()
    alpha = 0.15

    return U_arr, U_iam, alpha, M

def main_1():
    U_arr, U_iam, alpha, M = get_problem_data_1()

    cons_rounds = 50
    consensus = BalancedChebyshevConsensus(
        CycleGraph.get_optimal_lapl_based_comm_W(M),
        cons_rounds=cons_rounds,
    )
    adjacency = CycleGraph.get_adjacency(M)
    edges = CycleGraph.get_edges(M)

    num_iter = 9

    # AsymptoticDRGrAv
    adrgrav = AsymptoticDRGrAv(alpha, consensus)
    adrgrav_loss_hist = []
    adrgrav_disagree_hist = []
    adrgrav_gen = adrgrav.algo_iters(U_arr)
    # DeEPCA
    deepca = DeEPCA(consensus)
    deepca_loss_hist = []
    deepca_disagree_hist = []
    deepca_gen = deepca.algo_iters(U_arr)
    # DPRGD
    dprgd = DPRGD(consensus, eta=1.5)
    dprgd_loss_hist = []
    dprgd_disagree_hist = []
    dprgd_gen = dprgd.algo_iters(U_arr)
    # DPRGT
    dprgt = DPRGT(consensus, eta=0.7)
    dprgt_loss_hist = []
    dprgt_disagree_hist = []
    dprgt_gen = dprgt.algo_iters(U_arr)
    # SarletteSepulchre
    sarsep = SarletteSepulchre(adjacency, eta=9.5e-2)
    sarsep_loss_hist = []
    sarsep_disagree_hist = []
    sarsep_gen = sarsep.algo_iters(U_arr)
    # GrassmannianGossip
    gossip = GrassmannianGossip(edges, a=0.9, b=8.5e-6, rounds=cons_rounds)
    gossip_loss_hist = []
    gossip_disagree_hist = []
    gossip_gen = gossip.algo_iters(U_arr)

    plot_mode = 'loss'

    for it in range(num_iter+1):
        plt.gca().clear()

        # AsymptoticDRGrAv
        adrgrav_iter_frame = next(adrgrav_gen)
        adrgrav_loss_hist.append((util.grassmannian_extrinsic_dist(
            adrgrav_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            adrgrav_disagree_hist.append(np.nan)
        else:
            adrgrav_disagree_hist.append((util.grassmannian_extrinsic_dist(
                adrgrav_iter_frame.U[None, :],
                adrgrav_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(adrgrav_loss_hist, '-x', label='DRGrAv')
        else:
            plt.semilogy(adrgrav_disagree_hist, '-x', label='DRGrAv')
        # DeEPCA
        deepca_iter_frame = next(deepca_gen)
        deepca_loss_hist.append((util.grassmannian_extrinsic_dist(
            deepca_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            deepca_disagree_hist.append(np.nan)
        else:
            deepca_disagree_hist.append((util.grassmannian_extrinsic_dist(
                deepca_iter_frame.U[None, :],
                deepca_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(deepca_loss_hist, '-x', label='DeEPCA')
        else:
            plt.semilogy(deepca_disagree_hist, '-x', label='DeEPCA')
        # DPRGD
        dprgd_iter_frame = next(dprgd_gen)
        dprgd_loss_hist.append((util.grassmannian_extrinsic_dist(
            dprgd_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            dprgd_disagree_hist.append(np.nan)
        else:
            dprgd_disagree_hist.append((util.grassmannian_extrinsic_dist(
                dprgd_iter_frame.U[None, :],
                dprgd_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(dprgd_loss_hist, '-x', label='DPRGD')
        else:
            plt.semilogy(dprgd_disagree_hist, '-x', label='DPRGD')
        # DPRGT
        dprgt_iter_frame = next(dprgt_gen)
        dprgt_loss_hist.append((util.grassmannian_extrinsic_dist(
            dprgt_iter_frame.U,
            U_iam
        )**2).mean(0))
        if 0 == it:
            dprgt_disagree_hist.append(np.nan)
        else:
            dprgt_disagree_hist.append((util.grassmannian_extrinsic_dist(
                dprgt_iter_frame.U[None, :],
                dprgt_iter_frame.U[:, None]
            )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(dprgt_loss_hist, '-x', label='DPRGT')
        else:
            plt.semilogy(dprgt_disagree_hist, '-x', label='DPRGT')
        # SarletteSepulchre
        sarsep_iter_frame = next(sarsep_gen)
        sarsep_loss_hist.append((util.grassmannian_extrinsic_dist(
            sarsep_iter_frame.U,
            U_iam
        )**2).mean(0))
        sarsep_disagree_hist.append((util.grassmannian_extrinsic_dist(
            sarsep_iter_frame.U[None, :],
            sarsep_iter_frame.U[:, None]
        )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(sarsep_loss_hist, '-x', label='COM')
        else:
            plt.semilogy(sarsep_disagree_hist, '-x', label='COM')
        # GrassmannianGossip
        gossip_iter_frame = next(gossip_gen)
        gossip_loss_hist.append((util.grassmannian_extrinsic_dist(
            gossip_iter_frame.U,
            U_iam
        )**2).mean(0))
        gossip_disagree_hist.append((util.grassmannian_extrinsic_dist(
            gossip_iter_frame.U[None, :],
            gossip_iter_frame.U[:, None]
        )**2).sum((0,1)) / (M*(M-1)))
        if 'loss' == plot_mode:
            plt.semilogy(gossip_loss_hist, '-x', label='Gossip')
        else:
            plt.semilogy(gossip_disagree_hist, '-x', label='Gossip')

        plt.legend()
        plt.xlabel('Iteration')
        if 'loss' == plot_mode:
            plt.ylabel('MSE')
        else:
            plt.ylabel('MSD')
        plt.title('Hypercube Graph')
        plt.pause(0.01)
    #plt.show()

    fig_loss = plt.figure(figsize=(6, 4))
    ax_loss = fig_loss.gca()
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), adrgrav_loss_hist, label='DRGrAv', linestyle='-', marker='o', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), deepca_loss_hist, label='DeEPCA', linestyle='--', marker='s', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), dprgd_loss_hist, label='DPRGD', linestyle='-.', marker='^', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), dprgt_loss_hist, label='DPRGT', linestyle=':', marker='d', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), sarsep_loss_hist, label='COM', linestyle='-', marker='x', markersize=4)
    ax_loss.semilogy(cons_rounds*np.arange(num_iter+1), gossip_loss_hist, label='Gossip', linestyle='--', marker='*', markersize=5)
    ax_loss.legend()
    ax_loss.set_xlabel('Communication Rounds')
    ax_loss.set_ylabel('MSE')
    ax_loss.set_title('Cycle Graph')
    fig_loss.savefig(f'{DIR}/cycle_mse.pdf')
    fig_disagree = plt.figure(figsize=(6, 4))
    ax_disagree = fig_disagree.gca()
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), adrgrav_disagree_hist, label='DRGrAv', linestyle='-', marker='o', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), deepca_disagree_hist, label='DeEPCA', linestyle='--', marker='s', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), dprgd_disagree_hist, label='DPRGD', linestyle='-.', marker='^', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), dprgt_disagree_hist, label='DPRGT', linestyle=':', marker='d', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), sarsep_disagree_hist, label='COM', linestyle='-', marker='x', markersize=4)
    ax_disagree.semilogy(cons_rounds*np.arange(num_iter+1), gossip_disagree_hist, label='Gossip', linestyle='--', marker='*', markersize=5)
    ax_disagree.legend()
    ax_disagree.set_xlabel('Communication Rounds')
    ax_disagree.set_ylabel('MSD')
    ax_disagree.set_title('Cycle Graph')
    fig_disagree.savefig(f'{DIR}/cycle_msd.pdf')


if __name__ == '__main__':
    import sys
    if 2 > len(sys.argv):
        print('Must provide save directory as first argument')
        exit(-1)
    DIR = sys.argv[1]
    #main_0()
    #main_1()
    for tol in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
        timing_0('DRGrAv', tol=tol)
        timing_0('DeEPCA', tol=tol)
        timing_0('DPRGD', tol=tol)
        timing_0('DPRGT', tol=tol)
        timing_0('COM', tol=tol)
        timing_0('Gossip', tol=tol)


