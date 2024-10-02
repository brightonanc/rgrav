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
    eps = 1e-40
    thresh_level = 0.25
    thresh_level = 0.5
    x = torch.linspace(1e-5, 1, 2048).double()
    
    # Set up the plot style
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'mathtext.default': 'regular',
    })
    sns.set_context("paper", font_scale=1.5)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ns = torch.arange(1, 21, 2)
    ns = torch.arange(1, 41, 4)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=ns.min(), vmax=ns.max())

    for n in ns:
        power_vals = get_polynomial_power(n)(x).abs()
        chebyshev_vals = get_polynomial_chebyshev(n)(x).abs()
        power_vals += eps
        chebyshev_vals += eps
        
        color = cmap(norm(n))
        ax1.plot(x, power_vals, c=color)
        ax2.plot(x, chebyshev_vals, c=color)

    for ax in (ax1, ax2):
        ax.set_xlim([x.min(), 1])
        ax.set_ylim([chebyshev_vals.min() / 10, 1])
        ax.set_yscale('log')
        ax.axvline(x=thresh_level, color='red', linestyle='--', alpha=0.7, label='Noise Threshold ($\\alpha$)')
        ax.set_xlabel('$\lambda$')
        ax.grid(True, which="both", ls="-", alpha=0.2)

    ax1.set_ylabel('$|f_t(\lambda)|$')
    ax1.set_title('Power Method')
    ax2.set_title('Chebyshev Recursion')
    ax2.legend()

    # Adjust layout to make space for colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.90)

    # Add colorbar
    cbar_ax = fig.add_axes([0.93, 0.125, 0.02, 0.795])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, label='Polynomial Order ($t$)', aspect=30)
    cbar.set_ticks(ns)
    cbar.ax.invert_yaxis()  # Invert the y-axis of the colorbar

    plt.savefig('plots/eig_polynomial.png', dpi=300, bbox_inches='tight')

    thresh_level = 0.25
    cdf_min = 1e-30
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    n_eig = 1000
    eigs = torch.linspace(0, thresh_level, 1000)

    def cdf(eigs):
        xx = torch.from_numpy(np.geomspace(cdf_min, 1, 1000))
        yy = [torch.sum(eigs < x) / eigs.numel() for x in xx]
        return xx, yy

    ns = torch.arange(1, 21, 2)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=ns.min(), vmax=ns.max())
    for i, n in enumerate(ns):
        eigs_power = get_polynomial_power(n)(eigs).abs()
        eigs_cheb = get_polynomial_chebyshev(n)(eigs).abs()
        power_xx, power_yy = cdf(eigs_power)
        cheb_xx, cheb_yy = cdf(eigs_cheb)
        color = cmap(norm(n))
        ax1.plot(power_xx, power_yy, c=color)
        ax2.plot(cheb_xx, cheb_yy, c=color)

    for ax in (ax1, ax2):
        ax.set_xlim([chebyshev_vals.min() / 10, 1.0])
        # ax.set_xlim([cdf_min, 1.0])
        ax.set_ylim([0, 1])
        ax.set_xscale('log')
        ax.axvline(x=thresh_level, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('$\lambda$')
        ax.grid(True, which="both", ls="-", alpha=0.2)

    ax1.set_ylabel(f'CDF( $|f_n(\lambda)|$ )')
    ax1.set_title('Power Method')
    ax2.set_title('Chebyshev Recursion')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], label='n', aspect=30)
    cbar.set_ticks(ns)

    plt.tight_layout()
    plt.savefig('plots/eig_cdf.png', dpi=300, bbox_inches='tight')
