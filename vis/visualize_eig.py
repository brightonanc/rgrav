import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import ChebyshevMagicNumbers, BalancedChebyshevMagicNumbers, HypercubeGraph

torch.set_default_dtype(torch.float64)

def get_polynomial_power(cons_rounds):
    return lambda x: x ** cons_rounds

def apply_polynomial(x, n, cmn):
    poly = torch.ones_like(x).double()
    poly_iter = poly
    for i in range(1, n+1):
        if 1 == i:
            next_poly_iter = x
        else:
            a = cmn.a(i)
            b = cmn.b(i)
            c = cmn.c(i)
            next_poly_iter = a * (
                (x * poly_iter) + (b * poly_iter)
                + (c * prev_poly_iter)
            )
        prev_poly_iter = poly_iter
        poly_iter = next_poly_iter
    return poly_iter

def get_polynomial_chebyshev(cons_rounds):
    cmn = ChebyshevMagicNumbers(thresh_level)
    return lambda x: apply_polynomial(x, cons_rounds, cmn)


if __name__ == '__main__':
    eps = 1e-20
    thresh_level = 0.25
    x = torch.linspace(1e-5, 1, 512).double()
    
    # Set up the plot style
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ns = torch.arange(1, 11)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=ns.min(), vmax=ns.max())

    for n in ns:
        power_vals = get_polynomial_power(n)(x).abs()
        chebyshev_vals = get_polynomial_chebyshev(n)(x).abs()
        power_vals += eps
        chebyshev_vals += eps
        
        color = cmap(norm(n))
        ax1.plot(x, power_vals, c=color, label=f'n={n}')
        ax2.plot(x, chebyshev_vals, c=color, label=f'n={n}')

    for ax in (ax1, ax2):
        ax.set_xlim([x.min(), 1])
        ax.set_ylim([1e-16, 1])
        ax.set_yscale('log')
        ax.axvline(x=thresh_level, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('x')
        ax.set_ylabel('|f(x)|')
        ax.grid(True, which="both", ls="-", alpha=0.2)

    ax1.set_title('Power Method')
    ax2.set_title('Chebyshev Recursion')

    thresh_level = 0.25
    cdf_min = 1e-30
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    n_eig = 1000
    eig_low = torch.linspace(0, thresh_level, n_eig)
    eig_high = torch.linspace(0.9, 1, n_eig)
    eig_low = torch.rand(n_eig) * thresh_level
    eig_high = 1 - torch.rand(n_eig) * .01
    eigs = torch.cat((eig_low, eig_high))
    eigs = eig_low
    eigs = torch.linspace(0, thresh_level, 1000)

    def cdf(eigs):
        xx = torch.linspace(cdf_min, 1, 1000)
        xx = torch.from_numpy(np.geomspace(cdf_min, 1, 1000))
        yy = [torch.sum(eigs < x) / eigs.numel() for x in xx]
        return xx, yy

    # even order come down, odd order come up
    ns = [0, *torch.arange(1, 20, 2)]
    ns = torch.tensor(ns)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=ns.min(), vmax=ns.max())
    for i, n in enumerate(ns):
        if n > 0:
            eigs_power = get_polynomial_power(n)(eigs)
            eigs_cheb = get_polynomial_chebyshev(n)(eigs)
        else:
            eigs_power = eigs
            eigs_cheb = eigs
        power_xx, power_yy = cdf(eigs_power)
        cheb_xx, cheb_yy = cdf(eigs_cheb)
        ax1.plot(power_xx, power_yy, c=cmap(norm(n)))
        ax2.plot(cheb_xx, cheb_yy, c=cmap(norm(n)))

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.xlim([cdf_min, 1.0])
        plt.axvline(x=thresh_level, color='red', linestyle='--', alpha=0.7)
        plt.semilogx()
        # plt.semilogy()

    plt.tight_layout()
    plt.show()
