import torch
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

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # cbar = fig.colorbar(sm, ax=[ax1, ax2], label='n', aspect=30)
    # cbar.set_ticks(ns)

    plt.tight_layout()
    plt.show()
